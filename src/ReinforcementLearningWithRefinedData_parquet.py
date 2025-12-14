# =========================
# Parquet-split pipeline (BERT + PPO + eval)
# =========================
# --- Imports & seeds ---
import os
import json
import glob
import random
import datetime
from pathlib import Path
from paths import get_project_root

import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange
from scipy.special import softmax

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

# =========================
# 0) GLOBALS
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================
# PROJECT ROOT & STANDARD PATHS
# =========================
PROJECT_ROOT = get_project_root()

# Canonical data locations
PARQUET_PATH = PROJECT_ROOT / "data" / "RefinedData.parquet"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
TRAIN_SPLIT = SPLIT_DIR / "train.parquet"
VAL_SPLIT   = SPLIT_DIR / "val.parquet"
TEST_SPLIT  = SPLIT_DIR / "test.parquet"
SPLIT_JSON  = SPLIT_DIR / "split_seed42.json"

# Optional class order file (for stable label ids across runs)
CLASS_ORDER_CANDIDATES = [
    str(PROJECT_ROOT / "config" / "class_order.json"),
    "config/class_order.json",
]

# Checkpoint directories (canonical)
BERT_CKPT_DIR = PROJECT_ROOT / "models" / "bert_checkpoints"
PPO_CKPT_DIR  = PROJECT_ROOT / "models" / "ppo_checkpoints"
os.makedirs(BERT_CKPT_DIR, exist_ok=True)
os.makedirs(PPO_CKPT_DIR, exist_ok=True)

# =========================
# 1) UTILITIES
# =========================
def harmonize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename to ['text','label'], strip whitespace, drop empties & duplicate texts.
    """
    orig_cols = list(df.columns)
    rename_map = {}
    for col in df.columns:
        lc = col.lower().strip()
        if lc in ("text", "description", "desc", "prompt", "input"):
            rename_map[col] = "text"
        elif lc in ("label", "labels", "category", "class", "food", "target", "label_id"):
            rename_map[col] = "label"
    if rename_map:
        df = df.rename(columns=rename_map)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError(
            f"[data] Expected columns ['text','label'] after harmonization. "
            f"Original: {orig_cols} | Current: {list(df.columns)}"
        )
    df["text"]  = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df

def load_class_order_or_infer(*dfs: pd.DataFrame) -> list[str]:
    """
    Priority:
      1) Load class_order.json if present
      2) Otherwise infer from union of labels across provided dataframes (sorted)
    """
    for cp in CLASS_ORDER_CANDIDATES:
        p = Path(cp)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                classes = json.load(f)
            print(f"[data] Loaded class order: {cp}")
            return classes
    # infer from union of labels in the splits
    labels_union = set()
    for d in dfs:
        labels_union.update(d["label"].unique().tolist())
    classes = sorted(labels_union)
    print("[data] No class_order.json found — inferred classes from split data.")
    return classes

def map_labels(df: pd.DataFrame, label2id: dict) -> pd.DataFrame:
    unknown = set(df["label"].unique()) - set(label2id.keys())
    if unknown:
        raise ValueError(f"[data] Unknown labels absent in class list: {unknown}")
    df = df.copy()
    df["label_id"] = df["label"].map(label2id).astype(int)
    return df

# =========================
# 2) LOAD SPLIT PARQUETS (fallback to JSON indices if needed)
# =========================
if TRAIN_SPLIT.exists() and VAL_SPLIT.exists() and TEST_SPLIT.exists():
    print(f"[data] Loading split Parquet files from: {SPLIT_DIR}")
    df_train = harmonize(pd.read_parquet(TRAIN_SPLIT, engine="pyarrow"))
    df_val   = harmonize(pd.read_parquet(VAL_SPLIT,   engine="pyarrow"))
    df_test  = harmonize(pd.read_parquet(TEST_SPLIT,  engine="pyarrow"))
else:
    print("[data] Split Parquet files not found. Attempting fallback via split_seed42.json…")
    if not (PARQUET_PATH.exists() and SPLIT_JSON.exists()):
        raise FileNotFoundError(
            f"Missing required files. Expected either split Parquets in {SPLIT_DIR}, "
            f"or both {PARQUET_PATH} and {SPLIT_JSON}."
        )
    df_full = harmonize(pd.read_parquet(PARQUET_PATH, engine="pyarrow"))
    with open(SPLIT_JSON, "r", encoding="utf-8") as f:
        split = json.load(f)
    train_idx = split["train_idx"]
    val_idx   = split["val_idx"]
    test_idx  = split["test_idx"]
    df_train = df_full.loc[train_idx].reset_index(drop=True)
    df_val   = df_full.loc[val_idx].reset_index(drop=True)
    df_test  = df_full.loc[test_idx].reset_index(drop=True)
    print("[data] Loaded splits via JSON indices.")

print(f"[data] Split sizes: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

# Build class mapping (stable across runs)
class_names = load_class_order_or_infer(df_train, df_val, df_test)
label2id = {c: i for i, c in enumerate(class_names)}
id2label = {i: c for c, i in label2id.items()}

# Map to numeric ids
df_train = map_labels(df_train, label2id)
df_val   = map_labels(df_val,   label2id)
df_test  = map_labels(df_test,  label2id)

# Canonical lists for downstream components
train_texts = df_train["text"].tolist()
val_texts   = df_val["text"].tolist()
test_texts  = df_test["text"].tolist()

train_labels = np.array(df_train["label_id"].tolist(), dtype=np.int64)
val_labels   = np.array(df_val["label_id"].tolist(),   dtype=np.int64)
test_labels  = np.array(df_test["label_id"].tolist(),  dtype=np.int64)

print(f"[data] Classes={len(class_names)} | "
      f"train={len(train_texts)} | val={len(val_texts)} | test={len(test_texts)}")

# =========================
# 3) TOKENIZATION
# =========================
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
max_length = 128

def tokenize_texts(text_list):
    filtered = [t if t is not None else "" for t in text_list]
    return tokenizer(
        filtered, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt"
    )

train_encodings = tokenize_texts(train_texts)
val_encodings   = tokenize_texts(val_texts)
test_encodings  = tokenize_texts(test_texts)

print(
    "[tok] Shapes:",
    train_encodings["input_ids"].shape,
    val_encodings["input_ids"].shape,
    test_encodings["input_ids"].shape,
)

# =========================
# 4) DATALOADERS
# =========================
train_dataset = TensorDataset(
    train_encodings["input_ids"], train_encodings["attention_mask"], torch.tensor(train_labels)
)
val_dataset = TensorDataset(
    val_encodings["input_ids"], val_encodings["attention_mask"], torch.tensor(val_labels)
)
test_dataset = TensorDataset(
    test_encodings["input_ids"], test_encodings["attention_mask"], torch.tensor(test_labels)
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

print("[data] Dataloaders ready.")

# =========================
# 5) FINE-TUNE BERT (sequence classification)
# =========================
num_labels = len(class_names)
clf_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels).to(device)

epochs = 8
lr = 2e-5
weight_decay = 0.01
warmup_proportion = 0.1

optimizer = AdamW(clf_model.parameters(), lr=lr, weight_decay=weight_decay)
total_steps = len(train_loader) * epochs
warmup_steps = int(total_steps * warmup_proportion)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

best_val_loss = float("inf")
patience = 2
patience_counter = 0

for epoch in range(epochs):
    clf_model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"[BERT] Train epoch {epoch+1}"):
        b_input_ids, b_attn, b_labels = [b.to(device) for b in batch]
        outputs = clf_model(input_ids=b_input_ids, attention_mask=b_attn, labels=b_labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(clf_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # validation
    clf_model.eval()
    val_loss = 0.0
    preds, trues = [], []
    with torch.no_grad():
        for batch in val_loader:
            b_input_ids, b_attn, b_labels = [b.to(device) for b in batch]
            outputs = clf_model(input_ids=b_input_ids, attention_mask=b_attn, labels=b_labels)
            loss = outputs.loss
            val_loss += loss.item()
            logits = outputs.logits.detach().cpu().numpy()
            preds.extend(np.argmax(logits, axis=1).tolist())
            trues.extend(b_labels.detach().cpu().numpy().tolist())

    avg_val_loss = val_loss / len(val_loader)
    val_acc = accuracy_score(trues, preds)
    val_f1  = f1_score(trues, preds, average="weighted")
    print(f"[BERT] Epoch {epoch+1}/{epochs} "
          f"Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f} | "
          f"Acc {val_acc:.4f} | F1 {val_f1:.4f}")

    if avg_val_loss < best_val_loss - 1e-4:
        best_val_loss = avg_val_loss
        patience_counter = 0
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        ckpt_dir = BERT_CKPT_DIR / f"bert_{ts}"
        clf_model.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))
        print(f"[BERT] Saved best checkpoint: {ckpt_dir}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("[BERT] Early stopping triggered.")
            break

# Load latest best checkpoint for embedding extraction
runs = sorted(glob.glob(str(BERT_CKPT_DIR / "bert_*")))
if not runs:
    raise FileNotFoundError(f"No checkpoint found in {BERT_CKPT_DIR}/.")
latest_ckpt = runs[-1]
print(f"[BERT] Loading best checkpoint for embeddings: {latest_ckpt}")
bert_for_emb = AutoModel.from_pretrained(latest_ckpt).to(device)
bert_tok     = AutoTokenizer.from_pretrained(latest_ckpt)

# =========================
# 6) EMBEDDINGS
# =========================
def compute_embeddings(model_base, tokenizer, texts_list, batch_size=64, max_len=128):
    model_base.eval()
    embs = []
    for i in tqdm(range(0, len(texts_list), batch_size), desc="[BERT] Embeddings"):
        batch_texts = texts_list[i:i+batch_size]
        batch_texts = [t if t is not None else "" for t in batch_texts]
        enc = tokenizer(
            batch_texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out = model_base(**enc, return_dict=True)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                emb = out.pooler_output
            else:
                emb = out.last_hidden_state.mean(dim=1)
            emb = F.normalize(emb, p=2, dim=1)
            embs.append(emb.cpu())
    return torch.cat(embs, dim=0) if embs else torch.zeros((0, model_base.config.hidden_size))

print("[BERT] Computing embeddings...")
train_embeddings = compute_embeddings(bert_for_emb, bert_tok, train_texts, max_len=max_length)
val_embeddings   = compute_embeddings(bert_for_emb, bert_tok, val_texts,   max_len=max_length)
test_embeddings  = compute_embeddings(bert_for_emb, bert_tok, test_texts,  max_len=max_length)
print("[BERT] Embedding shapes:", train_embeddings.shape, val_embeddings.shape, test_embeddings.shape)

# =========================
# 7) RL ENV (embedding match via cosine reward)
# =========================
class EmbeddingMatchEnv(gym.Env):
    """
    One-step task:
      - obs: target embedding (D,)
      - action: (D,)
      - reward: cosine(action, target) mapped to [0, 1]
    """
    metadata = {"render_modes": []}

    def __init__(self, target_embeddings: torch.Tensor, max_steps: int = 1):
        super().__init__()
        assert isinstance(target_embeddings, torch.Tensor), "target_embeddings must be a torch.Tensor"
        self.targets = target_embeddings.detach().cpu().float()  # shape: [N, D]
        self.N, self.D = self.targets.shape
        self.max_steps = max_steps
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.D,), dtype=np.float32)
        self.action_space      = spaces.Box(low=-1.0,     high=1.0,   shape=(self.D,), dtype=np.float32)
        self.idx = None
        self.steps = 0

    @staticmethod
    def _norm_vec(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        n = np.linalg.norm(x)
        return x if n == 0 else (x / (n + 1e-9))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = int(np.random.randint(0, self.N))
        self.steps = 0
        obs = self._norm_vec(self.targets[self.idx])  # (D,)
        return obs.astype(np.float32), {}

    def step(self, action):
        self.steps += 1
        terminated = True
        truncated = False
        a = self._norm_vec(action)                  # (D,)
        t = self._norm_vec(self.targets[self.idx])  # (D,)
        cos = float(np.dot(a, t))                   # [-1, 1]
        reward = (cos + 1.0) / 2.0                  # [0, 1]
        info = {"cos_sim": cos}
        next_obs = t.astype(np.float32)
        return next_obs, reward, terminated, truncated, info

def make_env_from_embeddings(embs: torch.Tensor):
    def _init():
        env = EmbeddingMatchEnv(embs, max_steps=1)
        return Monitor(env)
    return _init

# VecEnvs (training & validation)
env      = DummyVecEnv([make_env_from_embeddings(train_embeddings)])
eval_env = DummyVecEnv([make_env_from_embeddings(val_embeddings)])
print("[RL] Envs created.")

# =========================
# 8) PPO TRAIN
# =========================
policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])
learning_rate = 3e-4
n_steps       = 1024
batch_size    = 64
ent_coef      = 0.01
vf_coef       = 0.5
max_grad_norm = 0.5
clip_range    = 0.2
total_timesteps   = 200_000
patience_evals    = 6
eval_freq         = 10_000
n_eval_episodes   = 5

stop_cb = StopTrainingOnNoModelImprovement(
    max_no_improvement_evals=patience_evals, min_evals=1, verbose=1
)
eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=str(PPO_CKPT_DIR),            # best_model.zip goes here
    log_path=str(PPO_CKPT_DIR / "eval_logs"),          # optional eval logs
    eval_freq=eval_freq,
    n_eval_episodes=n_eval_episodes,
    deterministic=True,
    callback_on_new_best=stop_cb,
)

ppo = PPO(
    "MlpPolicy",
    env,
    learning_rate=learning_rate,
    n_steps=n_steps,
    batch_size=batch_size,
    ent_coef=ent_coef,
    vf_coef=vf_coef,
    max_grad_norm=max_grad_norm,
    clip_range=clip_range,
    policy_kwargs=policy_kwargs,
    verbose=1,
    seed=SEED,
    gamma=0.0,     # one-step bandit
    gae_lambda=0.0
)
ppo.learn(total_timesteps=total_timesteps, callback=eval_cb)

# versioned PPO save (timestamped)
ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
ppo_path = PPO_CKPT_DIR / f"ppo_{ts}.zip"
ppo.save(str(ppo_path))
print(f"[RL] PPO saved to: {ppo_path}")

# =========================
# 9) PPO EVALUATION (on test)
# =========================
train_env_for_load = DummyVecEnv([make_env_from_embeddings(train_embeddings)])
best_model_zip = PPO_CKPT_DIR / "best_model.zip"
if best_model_zip.exists():
    ppo_eval = PPO.load(str(best_model_zip), env=train_env_for_load)
    print(f"[RL] Loaded best model for evaluation: {best_model_zip}")
else:
    ckpts = sorted(glob.glob(str(PPO_CKPT_DIR / "ppo_*.zip")))
    if not ckpts:
        raise FileNotFoundError(f"No PPO checkpoints found in {PPO_CKPT_DIR}.")
    latest = ckpts[-1]
    ppo_eval = PPO.load(latest, env=train_env_for_load)
    print(f"[RL] Loaded latest PPO: {latest}")

eval_env_test = DummyVecEnv([make_env_from_embeddings(test_embeddings)])
n_episodes = 200
all_rewards, all_cos = [], []
for _ in trange(n_episodes, desc="[RL] Eval episodes"):
    obs = eval_env_test.reset()
    total_r = 0.0
    while True:
        action, _ = ppo_eval.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env_test.step(action)
        r = float(rewards[0]) if hasattr(rewards, "__len__") else float(rewards)
        done = bool(dones[0]) if hasattr(dones, "__len__") else bool(dones)
        info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
        total_r += r
        if done:
            all_cos.append(info0.get("cos_sim", None))
            break
    all_rewards.append(total_r)

print("[RL] Episodes:", len(all_rewards))
print("[RL] Mean reward:", float(np.mean(all_rewards)))
valid_cos = [c for c in all_cos if c is not None]
if valid_cos:
    print("[RL] Mean cosine similarity (unscaled):", float(np.mean(valid_cos)))

# =========================
# 10) BERT EVALUATION (on test split)
# =========================
runs = sorted(glob.glob(str(BERT_CKPT_DIR / "bert_*")))
if not runs:
    raise FileNotFoundError(f"No checkpoint found in {BERT_CKPT_DIR}/.")
latest_ckpt = runs[-1]
print(f"[BERT] Evaluating checkpoint: {latest_ckpt}")
bert_cls = AutoModelForSequenceClassification.from_pretrained(latest_ckpt).eval()
bert_tok = AutoTokenizer.from_pretrained(latest_ckpt)

texts_eval  = df_test["text"].tolist()
labels_eval = np.array(df_test["label_id"].tolist(), dtype=np.int64)

batch_size_eval = 32
all_preds, all_probs = [], []
for i in range(0, len(texts_eval), batch_size_eval):
    batch = texts_eval[i:i+batch_size_eval]
    enc = bert_tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    with torch.no_grad():
        out = bert_cls(**enc)
        logits = out.logits.detach().cpu().numpy()
        probs  = softmax(logits, axis=1)
        preds  = np.argmax(probs, axis=1)
        all_preds.extend(preds)
        all_probs.extend(probs)

top1 = accuracy_score(labels_eval, all_preds)
top3 = np.mean([labels_eval[i] in np.argsort(all_probs[i])[-3:] for i in range(len(labels_eval))])
f1_macro = f1_score(labels_eval, all_preds, average="macro")

def compute_ece(probs, labels, n_bins=10):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    bin_bounds = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = (confidences > bin_bounds[b]) & (confidences <= bin_bounds[b + 1])
        if not np.any(in_bin):
            continue
        prop = np.mean(in_bin)
        acc_bin = np.mean(labels[in_bin] == predictions[in_bin])
        avg_conf = np.mean(confidences[in_bin])
        ece += abs(acc_bin - avg_conf) * prop
    return ece

ece = compute_ece(np.array(all_probs), labels_eval)
print(f"[BERT] Test Top-1 Accuracy: {top1*100:.2f}%")
print(f"[BERT] Test Top-3 Accuracy: {top3*100:.2f}%")
print(f"[BERT] Test Macro F1 Score: {f1_macro:.2f}")
print(f"[BERT] Test Expected Calibration Error: {ece:.4f}")
print("\nAll done ✅")
