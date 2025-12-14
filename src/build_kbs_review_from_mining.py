from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\sup028\OneDrive - University of Salford\Hybrid Knowledge-Based Expert System")
REPORTS_DIR = PROJECT_ROOT / "reports"


def find_latest_mining_dir() -> Path:
    """Find latest reports/kbs_mining_* directory if none is specified."""
    if not REPORTS_DIR.exists():
        raise FileNotFoundError(f"{REPORTS_DIR} does not exist")
    candidates = sorted(REPORTS_DIR.glob("kbs_mining_*"))
    if not candidates:
        raise FileNotFoundError(f"No kbs_mining_* folders found under {REPORTS_DIR}")
    return candidates[-1]


def build_review_from_mining(
    mining_dir: Path,
    min_fuzzy_score: float = 0.7,
    max_fuzzy_per_class: int | None = 20,
) -> Path:
    """
    Read fuzzy_term_suggestions.json (and optionally symbolic_soft_bias_suggestions.json)
    from mining_dir and create a review CSV with columns:
      type,key,class,weight,approved,notes
    suitable for apply_kbs_updates.py.
    """
    if not mining_dir.exists():
        raise FileNotFoundError(mining_dir)

    print(f"[INFO] Using mining_dir = {mining_dir}")

    fuzzy_path = mining_dir / "fuzzy_term_suggestions.json"
    symb_path = mining_dir / "symbolic_soft_bias_suggestions.json"

    rows: List[Dict[str, Any]] = []

    # ─────────────────────────────────────────────────────────
    # Fuzzy suggestions → type=fuzzy
    # ─────────────────────────────────────────────────────────
    if fuzzy_path.exists():
        with fuzzy_path.open("r", encoding="utf-8") as f:
            fuzzy_data = json.load(f)

        # fuzzy_data: {class_name: [[term, score], ...], ...}
        for cls, term_list in fuzzy_data.items():
            if not isinstance(term_list, list):
                continue

            cleaned = []
            for item in term_list:
                if not (isinstance(item, (list, tuple)) and len(item) == 2):
                    continue
                term, score = item
                try:
                    score = float(score)
                except Exception:
                    continue
                cleaned.append((str(term), score))

            # sort by score desc
            cleaned.sort(key=lambda x: -x[1])

            if max_fuzzy_per_class is not None:
                cleaned = cleaned[:max_fuzzy_per_class]

            for term, score in cleaned:
                approved = score >= min_fuzzy_score
                rows.append(
                    {
                        "type": "fuzzy",
                        "key": term,
                        "class": cls,
                        "weight": score,
                        "approved": "yes" if approved else "no",
                        "notes": f"auto from fuzzy_term_suggestions (score={score:.3f})",
                    }
                )
        print(f"[OK] Collected fuzzy suggestions from {fuzzy_path}")
    else:
        print(f"[WARN] No fuzzy_term_suggestions.json found in {mining_dir}")

    # ─────────────────────────────────────────────────────────
    # Symbolic suggestions → type=symbolic  (future use)
    # ─────────────────────────────────────────────────────────
    if symb_path.exists():
        with symb_path.open("r", encoding="utf-8") as f:
            try:
                sym_data = json.load(f)
            except Exception:
                sym_data = []

        if isinstance(sym_data, list) and sym_data:
            print(f"[INFO] Found {len(sym_data)} symbolic suggestions in {symb_path}")
            for item in sym_data:
                if not isinstance(item, dict):
                    continue

                key = item.get("key") or item.get("pattern") or item.get("token")
                cls = item.get("class") or item.get("target_class")
                weight = item.get("weight") or item.get("bias") or item.get("score")
                notes = item.get("notes") or "auto from symbolic_soft_bias_suggestions"

                if not key:
                    continue

                try:
                    weight_val = float(weight) if weight is not None else 0.5
                except Exception:
                    weight_val = 0.5

                rows.append(
                    {
                        "type": "symbolic",
                        "key": str(key),
                        "class": str(cls) if cls is not None else "",
                        "weight": weight_val,
                        "approved": "yes",
                        "notes": notes,
                    }
                )
        else:
            print(f"[INFO] No symbolic suggestions to process in {symb_path}")
    else:
        print(f"[WARN] No symbolic_soft_bias_suggestions.json found in {mining_dir}")

    if not rows:
        raise RuntimeError("No suggestions found to build review.csv")

    df = pd.DataFrame(rows, columns=["type", "key", "class", "weight", "approved", "notes"])
    out_path = mining_dir / "review_auto.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Wrote review file → {out_path}  (rows={len(df)})")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mining_dir",
        type=str,
        default=None,
        help="Path to kbs_mining_* directory (default: latest under reports/)",
    )
    ap.add_argument(
        "--min_fuzzy_score",
        type=float,
        default=0.7,
        help="Minimum fuzzy score to auto-approve (others kept but marked approved=no)",
    )
    ap.add_argument(
        "--max_fuzzy_per_class",
        type=int,
        default=20,
        help="Max fuzzy suggestions per class to include (None for all)",
    )
    args = ap.parse_args()

    if args.mining_dir is None:
        mining_dir = find_latest_mining_dir()
    else:
        mining_dir = Path(args.mining_dir)

    build_review_from_mining(
        mining_dir=mining_dir,
        min_fuzzy_score=args.min_fuzzy_score,
        max_fuzzy_per_class=args.max_fuzzy_per_class,
    )


if __name__ == "__main__":
    main()
