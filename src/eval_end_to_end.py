from __future__ import annotations
from paths import get_project_root

import argparse
import json
import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score
from scipy.special import softmax

# === Canonical Project Paths ===
PROJECT_ROOT = get_project_root()

# Canonical full dataset (for prototypes, fallback, etc.)
DATA_PARQUET = PROJECT_ROOT / "data" / "RefinedData.parquet"

# Canonical split locations
SPLIT_DIR   = PROJECT_ROOT / "data" / "splits"
TEST_SPLIT  = SPLIT_DIR / "test.parquet"
SPLIT_JSON  = SPLIT_DIR / "split_seed42.json"

# Models / config / reports
PPO_DIR      = PROJECT_ROOT / "models" / "ppo_checkpoints"
CONFIG_DIR   = PROJECT_ROOT / "config"
REPORT_DIR   = PROJECT_ROOT / "reports"

CLASS_ORDER_PATH  = CONFIG_DIR / "class_order.json"
PROTOS_PATH       = CONFIG_DIR / "class_prototypes.npz"
FUZZY_PATH        = CONFIG_DIR / "fuzzy_terms.json"
SYMBOLIC_PATH     = CONFIG_DIR / "symbolic_rules.json"

# === Your Model Components (must exist in src/) ===
# user_model should expose:
#   predict_logits(texts: List[str]) -> (np.ndarray [N,C], List[str])
#   encode_texts(texts: List[str]) -> np.ndarray [N,D]  (L2-normalized)
#   build_or_load_prototypes() -> (np.ndarray [C,D], List[str])
from user_model import predict_logits, encode_texts, build_or_load_prototypes
from ppo_infer import PPOClassifier

# --- KBS wrapper (compatible with your kbs.py) ---
try:
    from kbs import KBSExpert
except Exception:
    KBSExpert = None  # no KBS available

# Cache for different "modes" (both / fuzzy / symbolic)
_KBS_CACHE: Dict[str, object] = {}


def _get_kbs(mode: str):
    """
    mode ∈ {"both", "fuzzy", "symbolic"}
    Simulated using scale knobs:
        both     → fuzzy_scale=1.0, rule_scale=1.0
        fuzzy    → fuzzy_scale=1.0, rule_scale=0.0
        symbolic → fuzzy_scale=0.0, rule_scale=1.0
    """
    if KBSExpert is None:
        return None

    mode = mode.lower()
    if mode in _KBS_CACHE:
        return _KBS_CACHE[mode]

    # determine scale knobs
    fuzzy_scale = 1.0 if mode in ("both", "fuzzy") else 0.0
    rule_scale  = 1.0 if mode in ("both", "symbolic") else 0.0

    # construct expert
    kbs = KBSExpert(
        class_order_path=str(CLASS_ORDER_PATH),
        fuzzy_terms_path=str(FUZZY_PATH),
        symbolic_rules_path=str(SYMBOLIC_PATH),
        fuzzy_boost=0.75,
        allow_hard_masks=False,
        conf_gate=0.70,
        bias_cap=0.80,
        rule_scale=rule_scale,
        fuzzy_scale=fuzzy_scale,
        generic_dampen=0.60,
    )

    _KBS_CACHE[mode] = kbs
    return kbs


def apply_kbs_adjustments(logits: np.ndarray, texts: list[str], mode: str = "both") -> np.ndarray:
    """
    Apply KBS to logits.
    NOTE: KBSExpert.adjust_logits(text, logits) has *no* 'mode' parameter.
    We simulate "mode" using per-instance scale knobs in `_get_kbs`.
    """
    kbs = _get_kbs(mode)
    if kbs is None:
        return logits

    outs = []
    for t, lg in zip(texts, logits):
        adj, _meta = kbs.adjust_logits(t, np.asarray(lg, dtype=float))
        outs.append(adj)

    return np.stack(outs, axis=0)


# -----------------------------
# Data / class utilities
# -----------------------------
def load_parquet_and_labels(parquet_path: Path, class_order_path: Path) -> Tuple[pd.DataFrame, List[str], dict]:
    """
    Load a parquet file, harmonize to ['text','label'], clean empties/dupes,
    and attach numeric label ids according to class_order.json (if present)
    or inferred sorted labels otherwise.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")
    df = pd.read_parquet(parquet_path, engine="pyarrow")

    # Harmonize columns
    rename = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in {"text", "description", "desc", "prompt", "input"}:
            rename[c] = "text"
        if lc in {"label", "labels", "category", "class", "food", "target"}:
            rename[c] = "label"
    if rename:
        df = df.rename(columns=rename)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Expected columns 'text' and 'label'; got {list(df.columns)}")

    df["text"]  = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Class order
    if class_order_path.exists():
        classes = json.loads(class_order_path.read_text(encoding="utf-8"))
    else:
        classes = sorted(df["label"].unique().tolist())

    label2id = {c: i for i, c in enumerate(classes)}
    df["label_id"] = df["label"].map(label2id).astype(int)

    return df, classes, label2id


def ensure_prototypes(classes: List[str]) -> None:
    """
    Ensure PROTOS_PATH exists with keys 'prototypes' and 'class_names' and
    that both are pickle-free. If missing/bad, rebuild by averaging per-class
    text embeddings using user_model.encode_texts() from the *full* dataset.
    """
    def _ok_npz(p: Path) -> bool:
        try:
            with np.load(p, allow_pickle=False) as z:
                if "prototypes" not in z or "class_names" not in z:
                    return False
                P = z["prototypes"]
                names = z["class_names"]
                # same class count
                return P.ndim == 2 and len(names) == len(classes)
        except Exception:
            return False

    if _ok_npz(PROTOS_PATH):
        return

    print("[eval] Prototypes NPZ missing or incompatible; rebuilding…")
    # Build by averaging per class from canonical full parquet
    df, classes2, _ = load_parquet_and_labels(DATA_PARQUET, CLASS_ORDER_PATH)
    # Restrict to desired classes if needed
    if set(classes2) != set(classes):
        df = df[df["label"].isin(classes)].copy()

    texts_by_cls = {c: df[df["label"] == c]["text"].tolist() for c in classes}
    embs = []
    for c in classes:
        texts = texts_by_cls.get(c, [])
        if not texts:
            embs.append(None)
        else:
            E = encode_texts(texts)  # [n,D], L2-normalized
            embs.append(E.mean(axis=0))

    D = next((e.shape[0] for e in embs if e is not None), 768)
    P = []
    for e in embs:
        if e is None:
            e = np.zeros((D,), dtype=np.float32)
        e = e / (np.linalg.norm(e) + 1e-9)
        P.append(e.astype(np.float32))
    P = np.stack(P, axis=0)

    # Save pickle-free (class_names as fixed-width unicode, not object)
    np.savez(
        PROTOS_PATH,
        prototypes=P,
        class_names=np.array(classes, dtype="<U256"),
    )
    print(f"[eval] Rebuilt prototypes → {PROTOS_PATH} shape={P.shape}")


def load_test_split() -> tuple[pd.DataFrame, list[str], dict]:
    """
    Load the canonical test split for evaluation.

    Priority:
    1) data/splits/test.parquet  (preferred, created by splitter.py)
    2) RefinedData.parquet + split_seed42.json (fallback)
    """
    if TEST_SPLIT.exists():
        # Use the saved test.parquet produced by splitter.py
        df, classes, label2id = load_parquet_and_labels(TEST_SPLIT, CLASS_ORDER_PATH)
        return df, classes, label2id

    # Fallback: use indices from split_seed42.json on the full RefinedData.parquet
    if not (DATA_PARQUET.exists() and SPLIT_JSON.exists()):
        raise FileNotFoundError(
            f"Neither {TEST_SPLIT} nor both {DATA_PARQUET} and {SPLIT_JSON} exist. "
            "Run splitter.py first to create a reproducible split."
        )
    df_full, classes, label2id = load_parquet_and_labels(DATA_PARQUET, CLASS_ORDER_PATH)
    split = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    test_idx = split["test_idx"]
    df_test = df_full.loc[test_idx].reset_index(drop=True)
    return df_test, classes, label2id


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(probs: np.ndarray, y_true: np.ndarray) -> dict:
    y_pred = np.argmax(probs, axis=1)
    top1 = accuracy_score(y_true, y_pred)
    top3 = float(np.mean([y_true[i] in np.argsort(probs[i])[-3:] for i in range(len(y_true))]))
    f1_macro = f1_score(y_true, y_pred, average="macro")

    # ECE
    confidences = np.max(probs, axis=1)
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        mask = (confidences > bins[b]) & (confidences <= bins[b + 1])
        if not np.any(mask):
            continue
        prop = float(np.mean(mask))
        acc_bin = float(np.mean(y_true[mask] == y_pred[mask]))
        avg_conf = float(np.mean(confidences[mask]))
        ece += abs(acc_bin - avg_conf) * prop

    return {"top1": top1, "top3": top3, "f1_macro": f1_macro, "ece": ece}


# -----------------------------
# Evaluation Variants
# -----------------------------
def _auto_latest_ppo() -> Optional[str]:
    best = PPO_DIR / "best_model.zip"
    if best.exists():
        return str(best)
    zips = sorted(PPO_DIR.glob("ppo_*.zip"))
    return str(zips[-1]) if zips else None


def eval_variant(texts: List[str], y: np.ndarray, variant: str, ppo_ckpt: Optional[str]) -> dict:
    variant = variant.lower()

    if variant == "bert_proto":
        logits, model_classes = predict_logits(texts)  # [N,C]
        probs = softmax(logits, axis=1)

    elif variant == "ppo":
        ckpt = ppo_ckpt or _auto_latest_ppo()
        if ckpt is None:
            raise FileNotFoundError(f"No PPO checkpoint in {PPO_DIR}")
        ppo = PPOClassifier(ppo_path=ckpt)
        probs, model_classes = ppo.predict_proba(texts)

    elif variant == "ppo_kbs":
        ckpt = ppo_ckpt or _auto_latest_ppo()
        if ckpt is None:
            raise FileNotFoundError(f"No PPO checkpoint in {PPO_DIR}")
        ppo = PPOClassifier(ppo_path=ckpt)
        probs_base, model_classes = ppo.predict_proba(texts)
        logits_base = np.log(np.clip(probs_base, 1e-9, 1.0))
        logits_adj = apply_kbs_adjustments(logits_base, texts, mode="both")
        probs = softmax(logits_adj, axis=1)

    elif variant == "kbs_fuzzy_only":
        logits, model_classes = predict_logits(texts)
        logits_adj = apply_kbs_adjustments(logits, texts, mode="fuzzy")
        probs = softmax(logits_adj, axis=1)

    elif variant == "kbs_sym_only":
        logits, model_classes = predict_logits(texts)
        logits_adj = apply_kbs_adjustments(logits, texts, mode="symbolic")
        probs = softmax(logits_adj, axis=1)

    else:
        raise ValueError(f"Unknown variant: {variant}")

    metrics = compute_metrics(probs, y)
    return {
        "variant": variant,
        "metrics": metrics,
        "num_samples": int(len(texts)),
        "num_classes": int(probs.shape[1]),
    }


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant",
        type=str,
        default="ppo_kbs",
        choices=["bert_proto", "ppo", "ppo_kbs", "kbs_fuzzy_only", "kbs_sym_only"],
    )
    ap.add_argument(
        "--ppo_ckpt",
        type=str,
        default=None,
        help="Optional explicit PPO ckpt path; otherwise best_model.zip or latest ppo_*.zip",
    )
    ap.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Use first N rows of the test split (quick test)",
    )
    args = ap.parse_args()

    # Always evaluate on the canonical test split
    df, classes, label2id = load_test_split()

    if args.subset:
        df = df.iloc[:args.subset].copy()

    # ensure prototypes exist (pickle-free) for user_model, built from full DATA_PARQUET
    ensure_prototypes(classes)

    texts = df["text"].tolist()
    y = df["label_id"].to_numpy(dtype=np.int64)

    print(f"[eval] variant={args.variant}  samples={len(df)}  classes={len(classes)}")
    out = eval_variant(texts, y, args.variant, args.ppo_ckpt)

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"eval_{args.variant}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("[eval] Results:", out["metrics"])
    print(f"[eval] Saved report → {out_path}")


if __name__ == "__main__":
    main()

# cd "<repo_root>/src"

#python eval_end_to_end.py --variant bert_proto
#python eval_end_to_end.py --variant ppo
#python eval_end_to_end.py --variant ppo_kbs
#python eval_end_to_end.py --variant kbs_fuzzy_only
#python eval_end_to_end.py --variant kbs_sym_only
