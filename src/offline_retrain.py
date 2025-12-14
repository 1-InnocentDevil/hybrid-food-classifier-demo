# src/offline_retrain.py
# Offline (re)training of text-conditioned PPO using parquet + ingested logs.
# - Rebuilds prototypes if missing
# - Encodes texts with latest BERT checkpoint
# - Env: obs = embedding (D,), action = (D,)
# - Reward: cosine(action, target) in [0,1], gently shaped by user feedback
# - Uses logs/predict.parquet + logs/feedback_ranked.parquet (accepted_rank 0–9)
# - Saves PPO to models/ppo_checkpoints (timestamp + best_model.zip)

from __future__ import annotations
from pathlib import Path
import json, time, math, glob, shutil
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# ───────────────────────────────────────────────────────────────────────────────
# Canonical paths
# ───────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(r"C:\Users\sup028\OneDrive - University of Salford\Hybrid Knowledge-Based Expert System")

PARQUET_BASE       = PROJECT_ROOT / "data" / "RefinedData.parquet"
LOGS_DIR           = PROJECT_ROOT / "logs"
PREDICT_PARQ       = LOGS_DIR / "predict.parquet"
FEEDBACK_PARQ      = LOGS_DIR / "feedback.parquet"           # kept for compatibility (not directly used)
FEEDBACK_RANKED_PARQ = LOGS_DIR / "feedback_ranked.parquet"  # accepted_rank 0–9

CONFIG_DIR   = PROJECT_ROOT / "config"
CLASS_ORDER  = CONFIG_DIR / "class_order.json"
PROTOS_PATH  = CONFIG_DIR / "class_prototypes.npz"

BERT_DIR_ROOT = PROJECT_ROOT / "models" / "bert_checkpoints"
PPO_OUT_DIR   = PROJECT_ROOT / "models" / "ppo_checkpoints"

REPORTS_DIR   = PROJECT_ROOT / "reports"

# ───────────────────────────────────────────────────────────────────────────────
# Knobs
# ───────────────────────────────────────────────────────────────────────────────
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128

# data mixing
LOG_UPWEIGHT = 5         # repeat log rows during training
VAL_FRACTION = 0.10      # (not yet used, but kept for future val split)

# PPO config (aligned with RL script)
POLICY_KW = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])
LEARNING_RATE = 3e-4
N_STEPS       = 1024
BATCH_SIZE    = 64
ENT_COEF      = 0.01
VF_COEF       = 0.5
MAX_GRAD_NORM = 0.5
CLIP_RANGE    = 0.2
TOTAL_TIMESTEPS_MIN = 60_000     # floor; increased automatically with data size

# ───────────────────────────────────────────────────────────────────────────────
# Utils
# ───────────────────────────────────────────────────────────────────────────────
def set_seed(seed=42):
    import random
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def _latest_bert_dir() -> Path:
    runs = sorted(glob.glob(str(BERT_DIR_ROOT / "bert_*")))
    if not runs:
        raise FileNotFoundError(f"No BERT checkpoints under {BERT_DIR_ROOT}")
    return Path(runs[-1])

def normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return (x / n).astype(np.float32)

def timestamp_dir(parent: Path, prefix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    d = parent / f"{prefix}_{ts}"
    d.mkdir(parents=True, exist_ok=False)
    return d

# ───────────────────────────────────────────────────────────────────────────────
# Encoder + prototypes
# ───────────────────────────────────────────────────────────────────────────────
class TextEncoder:
    def __init__(self, ckpt_dir: Path):
        print("[i] Loading BERT from:", ckpt_dir)
        self.tok = AutoTokenizer.from_pretrained(str(ckpt_dir))
        self.bert = AutoModel.from_pretrained(str(ckpt_dir)).to(DEVICE).eval()
        self.D = self.bert.config.hidden_size

    @torch.no_grad()
    def encode_many(self, texts: List[str], bs: int = 64) -> np.ndarray:
        embs = []
        for i in tqdm(range(0, len(texts), bs), desc="Encode", unit="batch"):
            batch = [t if t else "" for t in texts[i:i+bs]]
            enc = self.tok(batch, truncation=True, padding=True,
                           max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
            out = self.bert(**enc, return_dict=True)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                e = out.pooler_output
            else:
                e = out.last_hidden_state.mean(dim=1)
            e = F.normalize(e, p=2, dim=1)
            embs.append(e.detach().cpu().numpy())
        return np.concatenate(embs, axis=0) if embs else np.zeros((0, self.D), dtype=np.float32)

def load_class_order() -> List[str]:
    return json.loads(CLASS_ORDER.read_text(encoding="utf-8"))

def recompute_and_save_prototypes(order: List[str], enc: TextEncoder) -> np.ndarray:
    df = pd.read_parquet(PARQUET_BASE, engine="pyarrow")
    # harmonise
    ren = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in {"text","description","desc","prompt","input"}: ren[c] = "text"
        if lc in {"label","labels","category","class","food","target"}: ren[c] = "label"
    if ren:
        df = df.rename(columns=ren)
    df["text"]  = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")]
    df = df[df["label"].isin(order)].copy()

    embs = enc.encode_many(df["text"].tolist())
    cls2idx = {c: i for i, c in enumerate(order)}
    C, D = len(order), embs.shape[1]
    sums = np.zeros((C, D), dtype=np.float32)
    cnts = np.zeros((C,), dtype=np.int32)
    for e, y in zip(embs, df["label"].tolist()):
        j = cls2idx[y]
        sums[j] += e
        cnts[j] += 1
    protos = sums / (cnts[:, None] + 1e-9)
    protos = normalize_rows(protos)
    np.savez(PROTOS_PATH, prototypes=protos, class_names=np.array(order, dtype=object))
    print(f"[✓] Prototypes rebuilt → {PROTOS_PATH} shape={protos.shape}")
    return protos

def load_or_build_prototypes(order: List[str], enc: TextEncoder) -> np.ndarray:
    if PROTOS_PATH.exists():
        with np.load(PROTOS_PATH, allow_pickle=True) as z:
            P = z["prototypes"]
            if P.shape[0] != len(order):
                print("[!] Prototype class mismatch; rebuilding…")
                return recompute_and_save_prototypes(order, enc)
            return normalize_rows(np.asarray(P, dtype=np.float32))
    return recompute_and_save_prototypes(order, enc)

# ───────────────────────────────────────────────────────────────────────────────
# Data (base + logs)
# ───────────────────────────────────────────────────────────────────────────────
def dedup_by_text(df: pd.DataFrame, col: str) -> pd.DataFrame:
    k = df[col].astype(str).str.strip().str.casefold()
    return df.loc[~k.duplicated(keep="first")].copy()

def build_base_df(order: List[str]) -> pd.DataFrame:
    if not PARQUET_BASE.exists():
        return pd.DataFrame(columns=["text", "label"])
    df = pd.read_parquet(PARQUET_BASE, engine="pyarrow")
    ren = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in {"text","description","desc","prompt","input"}: ren[c] = "text"
        if lc in {"label","labels","category","class","food","target"}: ren[c] = "label"
    if ren:
        df = df.rename(columns=ren)
    df["text"]  = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")]
    df = df[df["label"].isin(order)]
    df = dedup_by_text(df, "text")
    return df[["text", "label"]].reset_index(drop=True)

def build_logs_df(order: List[str]) -> pd.DataFrame:
    """
    Build logs dataframe from predict + ranked feedback (accepted_rank 0–9).
    Columns: text, label (may be None), accepted_rank, reward.
    """
    if (not PREDICT_PARQ.exists()) or (not FEEDBACK_RANKED_PARQ.exists()):
        return pd.DataFrame(columns=["text", "label", "accepted_rank", "reward"])

    pred = pd.read_parquet(PREDICT_PARQ)
    fb   = pd.read_parquet(FEEDBACK_RANKED_PARQ)

    if fb.empty:
        return pd.DataFrame(columns=["text", "label", "accepted_rank", "reward"])

    # one feedback per event_id (latest wins)
    fb = fb.sort_values("ts").drop_duplicates("event_id", keep="last")

    # inner join: only predictions that have ranked feedback
    df = pred.merge(fb, on="event_id", how="inner", suffixes=("", "_fb"))

    df["text"] = df["text"].astype(str)

    # supervision: correct_label if it is a known class; else None
    if "correct_label" in df.columns:
        df["label"] = df["correct_label"].where(df["correct_label"].isin(order), None)
    else:
        df["label"] = None

    # ensure numeric accepted_rank / reward
    if "accepted_rank" in df.columns:
        df["accepted_rank"] = pd.to_numeric(df["accepted_rank"], errors="coerce")
    else:
        df["accepted_rank"] = np.nan

    if "reward" in df.columns:
        df["reward"] = pd.to_numeric(df["reward"], errors="coerce")
    else:
        df["reward"] = np.nan

    df = dedup_by_text(df, "text")
    return df[["text", "label", "accepted_rank", "reward"]].reset_index(drop=True)

# ───────────────────────────────────────────────────────────────────────────────
# Build training targets and meta
# ───────────────────────────────────────────────────────────────────────────────
def build_targets(order: List[str], enc: TextEncoder) -> Tuple[np.ndarray, List[str], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Build target embeddings for PPO + per-example meta (accepted_rank, reward).
    - Base data: label → prototype (supervised)
    - Logs data: label → prototype if known, else self-embedding
    """
    base = build_base_df(order)      # ["text","label"]
    logs = build_logs_df(order)      # ["text","label","accepted_rank","reward"]

    # Upweight logs for training (not eval)
    logs_up = logs.copy()
    if LOG_UPWEIGHT > 1 and not logs_up.empty:
        logs_up = pd.concat([logs_up] * LOG_UPWEIGHT, ignore_index=True)

    # Align columns: base has no accepted_rank/reward
    if base.empty:
        base_for_concat = pd.DataFrame(columns=["text", "label", "accepted_rank", "reward"])
    else:
        base_for_concat = base.copy()
        base_for_concat["accepted_rank"] = np.nan
        base_for_concat["reward"]        = np.nan

    base_for_concat["src"] = "base"
    logs_up["src"]         = "logs"

    train_pool = pd.concat([base_for_concat, logs_up], ignore_index=True)
    train_pool = dedup_by_text(train_pool, "text")

    # Labels mapping
    cls2idx = {c: i for i, c in enumerate(order)}

    # Embeddings for all training texts
    texts = train_pool["text"].tolist()
    E = enc.encode_many(texts)            # [N,D], normalized
    E = normalize_rows(E)

    # Prototypes
    P = load_or_build_prototypes(order, enc)   # [C,D]

    # Targets = for labeled rows → corresponding prototype; else self-embedding
    t = E.copy()
    lab = train_pool["label"].tolist()
    for i, y in enumerate(lab):
        if isinstance(y, str) and y in cls2idx:
            t[i] = P[cls2idx[y]]
    T = normalize_rows(t)

    # Meta per example for PPO env
    meta: List[Dict[str, Any]] = []
    for _, row in train_pool.iterrows():
        meta.append({
            "src": row.get("src"),
            "accepted_rank": row.get("accepted_rank"),
            "user_reward": row.get("reward"),
        })

    info = {
        "N": len(texts),
        "D": E.shape[1],
        "texts": texts,
        "labels": lab,
        "cls2idx": cls2idx,
        "prototypes": P,
        "num_base": len(base),
        "num_logs": len(logs),
    }
    return T, texts, info, meta

# ───────────────────────────────────────────────────────────────────────────────
# PPO Env: obs=D, act=D, reward=cos→[0,1] shaped by logs
# ───────────────────────────────────────────────────────────────────────────────
class EmbeddingMatchEnv(gym.Env):
    """
    One-step env:
      - obs: target embedding (D,)
      - action: (D,)
      - reward: cosine(action, target) in [0,1],
                plus small shaping from accepted_rank and user_reward.
    """
    metadata = {}

    def __init__(self, targets: np.ndarray, meta: List[Dict[str, Any]]):
        super().__init__()
        assert isinstance(targets, np.ndarray) and targets.ndim == 2
        self.targets = targets.astype(np.float32)
        self.meta    = meta
        self.N, self.D = self.targets.shape
        assert len(self.meta) == self.N, "meta length must match number of targets"

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.D,), dtype=np.float32)
        self.action_space      = spaces.Box(low=-1.0,   high=1.0,
                                            shape=(self.D,), dtype=np.float32)
        self.i = 0
        self.weights = self._compute_weights()

    @staticmethod
    def _norm(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x) + 1e-9
        return (x / n).astype(np.float32)

    def _compute_weights(self) -> np.ndarray:
        """
        Compute sampling weights per example based on accepted_rank and user_reward.
        - Higher weight for low accepted_rank (top-1, top-2)
        - Higher weight for positive user_reward, lower for negative
        """
        w = np.ones(self.N, dtype=np.float32)
        for idx, m in enumerate(self.meta):
            ar = m.get("accepted_rank")
            ur = m.get("user_reward")

            # accepted_rank: 0 -> 1.0, 1 -> 0.5, 2 -> 0.25, else ~0.5 if unknown/-1
            if ar is None or (isinstance(ar, float) and math.isnan(ar)) or (ar < 0):
                w_rank = 0.5
            else:
                lam = math.log(2.0)
                w_rank = math.exp(-lam * float(ar))

            # user_reward: e.g. -1,0,1 -> scale between 0.5 and 1.5
            w_user = 1.0
            if isinstance(ur, (int, float)) and not (isinstance(ur, float) and math.isnan(ur)):
                w_user += 0.5 * float(ur)   # reward 1→1.5, 0→1.0, -1→0.5

            w[idx] = max(0.1, w_rank * w_user)

        s = float(w.sum())
        if s <= 0.0:
            return np.ones(self.N, dtype=np.float32) / float(self.N)
        return w / s

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # sample example index according to weights
        self.i = int(np.random.choice(self.N, p=self.weights))
        obs = self._norm(self.targets[self.i])
        return obs, {}

    def step(self, action):
        a = self._norm(np.asarray(action, dtype=np.float32).reshape(-1))
        t = self._norm(self.targets[self.i])

        cos = float(np.dot(a, t))           # [-1,1]
        base_r = (cos + 1.0) / 2.0          # [0,1] geometric reward

        m = self.meta[self.i]
        ar = m.get("accepted_rank")
        ur = m.get("user_reward")

        # Rank-based bonus: 0 -> +0.05, 1 -> ~+0.025, etc.
        if ar is None or (isinstance(ar, float) and math.isnan(ar)) or (ar < 0):
            r_rank = 0.5
        else:
            lam = math.log(2.0)
            r_rank = math.exp(-lam * float(ar))    # 1.0, 0.5, 0.25, ...
        bonus_rank = 0.1 * (r_rank - 0.5)          # in [-0.05, +0.05]

        # User reward: map {-1,0,1} to [-0.2, 0, 0.2]
        bonus_user = 0.0
        if isinstance(ur, (int, float)) and not (isinstance(ur, float) and math.isnan(ur)):
            bonus_user = 0.2 * float(ur)

        reward = base_r + bonus_rank + bonus_user
        reward = float(np.clip(reward, 0.0, 1.0))

        return t, reward, True, False, {"cos": cos}

def make_env(targets: np.ndarray, meta: List[Dict[str, Any]]):
    def _f():
        return Monitor(EmbeddingMatchEnv(targets, meta))
    return _f

# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────
def main():
    set_seed(SEED)
    print("=== Offline PPO retrain (text-conditioned) ===")
    order = load_class_order()
    bert_dir = _latest_bert_dir()
    enc = TextEncoder(bert_dir)

    targets, texts, info, meta = build_targets(order, enc)
    N, D = targets.shape
    print(f"[i] Targets: N={N}, D={D}")
    print(f"[i] Base examples: {info.get('num_base', 0)}, log examples: {info.get('num_logs', 0)}")

    env = DummyVecEnv([make_env(targets, meta)])

    ppo = PPO(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        clip_range=CLIP_RANGE,
        policy_kwargs=POLICY_KW,
        seed=SEED,
        verbose=1,
        gamma=0.0,
        gae_lambda=0.0,
        device=DEVICE,
    )

    total_ts = max(TOTAL_TIMESTEPS_MIN, 6 * N)   # scale with data size
    print(f"[i] Training PPO for {total_ts} steps …")
    ppo.learn(total_timesteps=total_ts)

    PPO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_zip = PPO_OUT_DIR / f"ppo_{ts}.zip"
    ppo.save(str(out_zip))
    print(f"[✓] Saved PPO → {out_zip}")

    # refresh best_model.zip
    best_zip = PPO_OUT_DIR / "best_model.zip"
    try:
        shutil.copy(str(out_zip), str(best_zip))
        print(f"[✓] Updated {best_zip.name}")
    except Exception as e:
        print("[WARN] Could not refresh best_model.zip:", e)

    print("[i] Done.")

if __name__ == "__main__":
    main()
