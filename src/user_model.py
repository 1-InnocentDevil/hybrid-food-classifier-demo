# src/user_model.py
from __future__ import annotations
import json, glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# ============== Canonical paths ==============
PROJECT_ROOT   = Path(r"C:\Users\sup028\OneDrive - University of Salford\Hybrid Knowledge-Based Expert System")
BERT_CKPT_DIR  = PROJECT_ROOT / "models" / "bert_checkpoints"
DATA_PARQUET   = PROJECT_ROOT / "data" / "RefinedData.parquet"
CLASS_ORDER    = PROJECT_ROOT / "config" / "class_order.json"
PROTOTYPE_NPZ  = PROJECT_ROOT / "config" / "class_prototypes.npz"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ BERT helpers ------------------
def _latest_bert_dir() -> Path:
    runs = sorted(glob.glob(str(BERT_CKPT_DIR / "bert_*")))
    if not runs:
        raise FileNotFoundError(f"No BERT checkpoints under {BERT_CKPT_DIR}")
    return Path(runs[-1])

def load_tokenizer_and_models() -> Tuple[AutoTokenizer, AutoModel, AutoModelForSequenceClassification]:
    ckpt = _latest_bert_dir()
    tok = AutoTokenizer.from_pretrained(str(ckpt))
    enc = AutoModel.from_pretrained(str(ckpt)).to(device).eval()
    clf = AutoModelForSequenceClassification.from_pretrained(str(ckpt)).to(device).eval()
    return tok, enc, clf

@torch.no_grad()
def encode_texts(texts: List[str], tokenizer: AutoTokenizer = None, encoder: AutoModel = None, max_len: int = 128) -> np.ndarray:
    """
    Returns L2-normalized embeddings [N, D].
    """
    if tokenizer is None or encoder is None:
        tokenizer, encoder, _ = load_tokenizer_and_models()

    texts = [t if t is not None else "" for t in texts]
    embs: List[np.ndarray] = []

    for i in range(0, len(texts), 64):
        batch = texts[i:i+64]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        out = encoder(**inputs, return_dict=True)
        # pooler if available else mean-pool
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            e = out.pooler_output
        else:
            e = out.last_hidden_state.mean(dim=1)
        e = F.normalize(e, p=2, dim=1)
        embs.append(e.detach().cpu().numpy())

    D = encoder.config.hidden_size
    return np.vstack(embs) if embs else np.zeros((0, D), dtype=np.float32)

# ------------------ Class order ------------------
def _load_class_order() -> List[str]:
    if CLASS_ORDER.exists():
        return json.loads(CLASS_ORDER.read_text(encoding="utf-8"))
    # derive from parquet
    import pandas as pd
    df = pd.read_parquet(DATA_PARQUET, engine="pyarrow")
    # harmonize column names
    rename = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in {"text","description","desc","prompt","input"}:  rename[c] = "text"
        if lc in {"label","labels","category","class","food","target"}: rename[c] = "label"
    if rename:
        df = df.rename(columns=rename)
    if "label" not in df.columns:
        raise ValueError("Could not infer classes: 'label' column missing in parquet.")
    labels = df["label"].astype(str).str.strip()
    return sorted(list(labels.unique()))

# ------------------ Prototypes ------------------
def _load_prototypes_if_valid(expected_classes: List[str]) -> Tuple[np.ndarray, List[str]] | None:
    """
    Try to load a clean NPZ with keys 'prototypes' (float2d) and 'class_names' (string array).
    Must not require pickle.
    """
    if not PROTOTYPE_NPZ.exists():
        return None
    try:
        with np.load(PROTOTYPE_NPZ, allow_pickle=False) as z:
            if "prototypes" not in z or "class_names" not in z:
                return None
            P = np.asarray(z["prototypes"], dtype=np.float32)
            names = [str(x) for x in z["class_names"].astype(str).tolist()]
        if P.ndim != 2 or len(names) != len(expected_classes):
            return None
        if names != expected_classes:
            return None
        # ensure rows are normalized
        P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
        return P, names
    except Exception:
        return None

def build_or_load_prototypes(tokenizer: AutoTokenizer = None, encoder: AutoModel = None) -> Tuple[np.ndarray, List[str]]:
    """
    Returns (prototypes [C,D], class_names [C]), saving a pickle-free NPZ:
      - 'prototypes': float32 [C,D]
      - 'class_names': unicode string array [C]
    """
    classes = _load_class_order()

    loaded = _load_prototypes_if_valid(classes)
    if loaded is not None:
        return loaded

    # Rebuild from parquet
    import pandas as pd
    df = pd.read_parquet(DATA_PARQUET, engine="pyarrow")
    # harmonize
    rename = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in {"text","description","desc","prompt","input"}:  rename[c] = "text"
        if lc in {"label","labels","category","class","food","target"}: rename[c] = "label"
    if rename:
        df = df.rename(columns=rename)
    if not {"text","label"}.issubset(df.columns):
        raise ValueError("Parquet must contain 'text' and 'label' columns.")

    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")]
    df = df[df["label"].isin(classes)]

    # load encoder if needed
    if tokenizer is None or encoder is None:
        tokenizer, encoder, _ = load_tokenizer_and_models()

    D = encoder.config.hidden_size
    protos: List[np.ndarray] = []
    for c in classes:
        texts = df.loc[df["label"] == c, "text"].tolist()
        if not texts:
            proto = np.zeros((D,), dtype=np.float32)
        else:
            E = encode_texts(texts, tokenizer, encoder)   # [n,D], normalized rows
            proto = E.mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-9)
        protos.append(proto.astype(np.float32))

    P = np.stack(protos, axis=0)  # [C,D]
    # Save WITHOUT pickle (store class names as Unicode strings)
    PROTOTYPE_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(PROTOTYPE_NPZ, prototypes=P, class_names=np.array(classes, dtype="<U128"))
    return P, classes

# ------------------ Inference API ------------------
def cosine_logits(embeddings: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """
    embeddings: [N,D] L2-normalized
    prototypes: [C,D] L2-normalized
    returns: cosine similarities [N,C]
    """
    return embeddings @ prototypes.T

def predict_logits(texts: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    BERT encoder → cosine-to-prototypes logits.
    Returns (logits [N,C], class_names [C])
    """
    tok, enc, _ = load_tokenizer_and_models()
    X = encode_texts(texts, tok, enc)            # [N,D], normalized
    P, names = build_or_load_prototypes(tok, enc)  # [C,D]
    logits = cosine_logits(X, P)                  # [N,C]
    return logits, names

# ------------------ Manual test ------------------
if __name__ == "__main__":
    demo = ["I love spicy crispy fries with aioli.", "Fresh salad with feta and olives."]
    l, names = predict_logits(demo)
    print("Classes:", names[:5], "… (total:", len(names), ")")
    print("Logits shape:", l.shape)
