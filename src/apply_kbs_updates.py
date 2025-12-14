"""
apply_kbs_updates.py

Applies *approved* updates to the KBS:
- config/fuzzy_terms.json     (class -> [[term, weight], ...])
- config/symbolic_rules.json  ({"soft_bias": [...], "hard_mask": [...]})
- config/version.json         (bumps kbs_version)

Input:
  --review path\to\review.csv

Expected CSV columns (case-insensitive):
  type,key,class,weight,approved,notes
  - type ∈ {fuzzy, symbolic}
  - approved ∈ {true/false/1/0/yes/no}
  - key: token/phrase (for fuzzy) OR condition token (for symbolic soft-bias)
  - class: target class (required for fuzzy and typical symbolic soft-bias)
  - weight: float (fuzzy weight or symbolic bias)

Any extra columns are ignored.
"""

from __future__ import annotations
import json
import shutil
from pathlib import Path
from datetime import datetime

import pandas as pd

# ───────────────────────────────────────────────────────────────────────────────
# Canonical paths (you can still pass a different CSV anywhere on disk)
# ───────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(r"C:\Users\sup028\OneDrive - University of Salford\Hybrid Knowledge-Based Expert System")
CONFIG_DIR   = PROJECT_ROOT / "config"
FUZZY_PATH   = CONFIG_DIR / "fuzzy_terms.json"
SYMB_PATH    = CONFIG_DIR / "symbolic_rules.json"
VER_PATH     = CONFIG_DIR / "version.json"
BACKUPS_DIR  = CONFIG_DIR / "_backups"

# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────
def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def _boolish(x) -> bool:
    s = str(x).strip().lower()
    return s in {"1","true","yes","y","approved"}

def _backup_all(ts: str):
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    out = BACKUPS_DIR / f"backup_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    for p in [FUZZY_PATH, SYMB_PATH, VER_PATH]:
        if p.exists():
            shutil.copy(p, out / p.name)
    return out

# ───────────────────────────────────────────────────────────────────────────────
# Apply logic
# ───────────────────────────────────────────────────────────────────────────────
def apply_updates(review_csv: str):
    review_csv = Path(review_csv)
    if not review_csv.exists():
        raise FileNotFoundError(review_csv)

    # Load current configs (class-centric fuzzy; soft_bias list in symbolic)
    fuzzy = _read_json(FUZZY_PATH, default={})                     # {class: [[term, weight], ...]}
    sym   = _read_json(SYMB_PATH,  default={"soft_bias": [], "hard_mask": []})

    if "soft_bias" not in sym:   sym["soft_bias"] = []
    if "hard_mask" not in sym:   sym["hard_mask"] = []

    ver   = _read_json(VER_PATH,  default={"kbs_version": 0})

    # Backup
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    bdir = _backup_all(ts)
    print(f"[INFO] Backups saved -> {bdir}")

    # Load review
    df = pd.read_csv(review_csv)
    # normalise headers
    df.columns = [c.strip().lower() for c in df.columns]

    # basic presence
    need = {"type","key","approved"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"review.csv missing required columns {need}; has {set(df.columns)}")

    # counters
    n_fuzzy, n_sym = 0, 0

    # Make fuzzy dict class-centric in memory for quick dedupe
    def _ensure_class_bucket(cls: str):
        if cls not in fuzzy or not isinstance(fuzzy[cls], list):
            fuzzy[cls] = []

    # For fast lookup to avoid duplicates
    def _has_fuzzy(cls: str, term: str) -> bool:
        return any((isinstance(x, (list, tuple)) and len(x) == 2 and str(x[0]) == term) for x in fuzzy.get(cls, []))

    # Iterate rows
    for _, r in df.iterrows():
        if not _boolish(r.get("approved", False)):
            continue

        typ   = str(r.get("type","")).strip().lower()
        key   = str(r.get("key","")).strip()
        cls   = None if pd.isna(r.get("class")) else str(r.get("class")).strip()
        notes = None if pd.isna(r.get("notes")) else str(r.get("notes")).strip()

        # parse weight
        w_raw = r.get("weight", None)
        try:
            weight = float(w_raw) if w_raw is not None and str(w_raw).strip() != "" else None
        except Exception:
            weight = None

        if typ == "fuzzy":
            if not cls or weight is None or not key:
                print(f"[WARN] Skip fuzzy row (need class+key+weight): {r.to_dict()}")
                continue
            _ensure_class_bucket(cls)
            if not _has_fuzzy(cls, key):
                fuzzy[cls].append([key, float(weight)])
                n_fuzzy += 1

        elif typ == "symbolic":
            # We use soft_bias entries by default: {"if_any":[key], "then_class": cls, "bias": weight, "notes": ...}
            if not key:
                print(f"[WARN] Skip symbolic row (missing key): {r.to_dict()}")
                continue
            entry = {"if_any": [key]}
            if cls:       entry["then_class"] = cls
            if weight is not None: entry["bias"] = float(weight)
            if notes:     entry["notes"] = notes
            sym["soft_bias"].append(entry)
            n_sym += 1

        else:
            print(f"[WARN] Unknown type={typ}; skipping.")

    # Sort fuzzy terms within each class by weight desc then term asc; cap to 200 if desired
    for cls, lst in fuzzy.items():
        try:
            lst = [[str(t), float(w)] for t, w in lst]
            lst.sort(key=lambda x: (-x[1], x[0]))
            fuzzy[cls] = lst[:200]
        except Exception:
            pass

    # Bump version
    ver["kbs_version"] = int(ver.get("kbs_version", 0)) + 1
    ver["last_updated"] = ts

    # Save
    _write_json(FUZZY_PATH, fuzzy)
    _write_json(SYMB_PATH,  sym)
    _write_json(VER_PATH,   ver)

    print(f"[OK] Applied updates: +{n_fuzzy} fuzzy, +{n_sym} symbolic. KBS version -> {ver['kbs_version']}")

# ───────────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", required=True, help="Path to curated review.csv")
    args = ap.parse_args()
    apply_updates(args.review)
