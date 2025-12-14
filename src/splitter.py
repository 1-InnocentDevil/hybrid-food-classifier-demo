"""
splitter.py

Creates a reproducible train/val/test split from RefinedData.parquet
and saves:
    - JSON with indices
    - train.parquet, val.parquet, test.parquet in splits directory
"""

from __future__ import annotations
from pathlib import Path
import json
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

# Canonical project paths
PROJECT_ROOT = Path(r"C:\Users\sup028\OneDrive - University of Salford\Hybrid Knowledge-Based Expert System")
DEFAULT_PARQUET = PROJECT_ROOT / "data" / "RefinedData.parquet"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
DEFAULT_JSON = SPLIT_DIR / "split_seed42.json"

def harmonize(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to text/label, strip, and drop empties."""
    rename = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in {"text", "description", "desc", "prompt", "input"}:
            rename[c] = "text"
        if lc in {"label", "labels", "category", "class", "food", "target", "label_id"}:
            rename[c] = "label"
    if rename:
        df = df.rename(columns=rename)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Expected columns ['text','label'] after harmonization, got {list(df.columns)}")

    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df

def make_split_indices(df: pd.DataFrame, seed: int) -> dict:
    """Fixed counts for 75 / 12.5 / 12.5 split, stratified on label."""
    y = df["label"].astype("category").cat.codes
    idx = df.index.to_list()
    n = len(df)

    n_train = int(n * 0.75)
    n_val = int(n * 0.125)
    n_test = n - n_train - n_val

    idx_trainval, idx_test = train_test_split(idx, test_size=n_test, stratify=y, random_state=seed)
    y_trainval = y.loc[idx_trainval]
    idx_train, idx_val = train_test_split(idx_trainval, test_size=n_val, stratify=y_trainval, random_state=seed)

    return {
        "train_idx": list(map(int, idx_train)),
        "val_idx": list(map(int, idx_val)),
        "test_idx": list(map(int, idx_test)),
    }

def main(parquet_path: Path, json_path: Path, seed: int):
    print(f"[INFO] Reading dataset: {parquet_path}")
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    df = harmonize(df)

    print("[INFO] Creating fixed-count 75/12.5/12.5 stratified split…")
    split = make_split_indices(df, seed)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    # Save JSON
    payload = {
        **split,
        "seed": seed,
        "meta": {
            "n_total": len(df),
            "n_train": len(split["train_idx"]),
            "n_val": len(split["val_idx"]),
            "n_test": len(split["test_idx"]),
            "parquet": str(parquet_path),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"[OK] Split saved → {json_path}")
    print(f"[OK] Counts: train={payload['meta']['n_train']}, val={payload['meta']['n_val']}, test={payload['meta']['n_test']}")

    # Save Parquet files
    train_df = df.loc[split["train_idx"]]
    val_df = df.loc[split["val_idx"]]
    test_df = df.loc[split["test_idx"]]

    train_df.to_parquet(SPLIT_DIR / "train.parquet", engine="pyarrow")
    val_df.to_parquet(SPLIT_DIR / "val.parquet", engine="pyarrow")
    test_df.to_parquet(SPLIT_DIR / "test.parquet", engine="pyarrow")
    print(f"[OK] Parquet files saved in {SPLIT_DIR}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(DEFAULT_PARQUET), help="Path to RefinedData.parquet")
    ap.add_argument("--json", default=str(DEFAULT_JSON), help="Where to save split JSON")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    main(Path(args.parquet), Path(args.json), args.seed)