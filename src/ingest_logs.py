"""
ingest_logs.py

Validates and merges:
- <PROJECT_ROOT>/logs/predict.jsonl
- <PROJECT_ROOT>/logs/feedback.jsonl

Outputs:
- <same dir>/predict.parquet
- <same dir>/feedback.parquet
- <same dir>/anomalies.csv (schema errors)
- <same dir>/unmatched_event_ids.csv (missing event_id matches)

This version maps fields written by src/server.py into the schema expected by schema_validate.py.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any

import pandas as pd

from schema_validate import (
    validate_predict_record,
    validate_feedback_record
)

# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[Dict]:
    """Load a JSONL file into a list of dicts (best-effort)."""
    records: List[Dict[str, Any]] = []
    if not path.exists():
        print(f"[WARN] {path} does not exist; treating as empty.")
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                records.append(json.loads(s))
            except json.JSONDecodeError as e:
                print(f"[ERROR] Bad JSON in {path}: {e}")
    return records

def _iso_to_ts(iso_str: str) -> float:
    """Convert ISO8601 string (UTC) to epoch seconds (float)."""
    # Python 3.11: datetime.fromisoformat supports offsets but not 'Z'; handle both.
    from datetime import datetime
    if not iso_str:
        return 0.0
    s = iso_str.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        return dt.timestamp()
    except Exception:
        # last resort: try pandas
        try:
            return float(pd.Timestamp(s).timestamp())
        except Exception:
            return 0.0

# ───────────────────────────────────────────────────────────────────────────────
# Mapping from server.py logs → validator schema
# ───────────────────────────────────────────────────────────────────────────────

def map_predict_record(r: Dict[str, Any]) -> Dict[str, Any]:
    """
    Server predicts log row (as written by src/server.py):

      {
        "id": "...",                     # predict ID
        "ts": "2025-11-08T12:34:56+00:00",
        "backend": "ppo" | "user_model",
        "use_kbs": true/false,
        "text": "...",
        "classes": [...class names...],  # full order used for probs
        "probs": [...float per class...],
        "top_k": 5,
        "kbs_meta": {...},
        "ranked_labels": [...sorted desc by prob...],
        "ranked_indices": [...]
      }

    Validator expects (typical):
      - event_id: str
      - ts: float
      - text: str
      - label_pred: str
      - topk: List[str]
      - probs: List[float]
      - ... (any extras may be kept in 'extra')
    """
    classes = r.get("classes") or []
    probs = r.get("probs") or []

    # derive ranking if not provided
    if "ranked_indices" in r and isinstance(r["ranked_indices"], list) and probs:
        order = r["ranked_indices"]
    else:
        order = list(sorted(range(len(probs)), key=lambda i: probs[i], reverse=True))

    # top-k names (use ranked_labels if present, else by order over classes)
    if "ranked_labels" in r and isinstance(r["ranked_labels"], list) and r["ranked_labels"]:
        topk_names = r["ranked_labels"][: int(r.get("top_k", 5) or 5)]
        label_pred = topk_names[0] if topk_names else (classes[order[0]] if order else "")
    else:
        k = int(r.get("top_k", 5) or 5)
        topk_names = [classes[i] for i in order[:k]] if classes and order else []
        label_pred = topk_names[0] if topk_names else (classes[order[0]] if order else "")

    mapped = {
        "event_id": str(r.get("id", "")),
        "ts": _iso_to_ts(str(r.get("ts", ""))),
        "text": str(r.get("text", "")),
        "label_pred": str(label_pred),
        "topk": [str(x) for x in topk_names],
        "probs": [float(x) for x in probs] if isinstance(probs, list) else [],
        # keep useful extras for analysis
        "extra": {
            "backend": r.get("backend"),
            "use_kbs": r.get("use_kbs"),
            "kbs_meta": r.get("kbs_meta"),
            "classes": classes,
            "ranked_indices": r.get("ranked_indices"),
        },
    }
    return mapped

def map_feedback_record(r: Dict[str, Any]) -> Dict[str, Any]:
    """
    Server feedback log row (current server.py):

      {
        "ts": "2025-11-08T12:35:10+00:00",
        "predict_id": "...",
        "text": "...",
        "predicted_label": "...",
        "correct_label": "...",
        "accepted_rank": 0 | 1 | ... | -1,

        "chosen_label": "...",
        "correct": true/false,
        "reward": -1/0/+1,
        "comment": "..."
      }

    Validator expects FeedbackEvent:
      - event_id: str
      - ts: float
      - user_action: str ("confirm" | "correct" | "flag")
      - correct_label: Optional[str]
      - reward: Optional[float]
      - predicted_label: Optional[str]
      - accepted_rank: Optional[int]
      - notes: Optional[str]
    """
    # --- existing logic to infer user_action, chosen, reward, correct_flag ---
    chosen = r.get("chosen_label")
    correct_flag = r.get("correct")
    reward = r.get("reward")

    # Derive a user_action string
    if correct_flag is True:
        user_action = "confirm"
    elif correct_flag is False:
        user_action = "correct"
    else:
        user_action = "flag"

    # Prefer new correct_label if server wrote it; otherwise fall back to chosen_label
    new_correct = r.get("correct_label")
    if new_correct is not None:
        correct_label = str(new_correct)
    elif chosen is not None:
        correct_label = str(chosen)
    else:
        correct_label = None

    # Coerce accepted_rank into int or None
    ar = r.get("accepted_rank")
    try:
        accepted_rank = int(ar) if ar is not None else None
    except (TypeError, ValueError):
        accepted_rank = None

    mapped = {
        "event_id": str(r.get("predict_id", "")),
        "ts": _iso_to_ts(str(r.get("ts", ""))),
        "user_action": user_action,
        "correct_label": correct_label,
        "reward": float(reward) if isinstance(reward, (int, float)) else None,

        # NEW: pass these directly into FeedbackEvent
        "predicted_label": r.get("predicted_label"),
        "accepted_rank": accepted_rank,

        # Map comment -> notes so it survives validation
        "notes": r.get("comment"),
    }
    return mapped
# ───────────────────────────────────────────────────────────────────────────────
# Validation
# ───────────────────────────────────────────────────────────────────────────────

def safe_validate(records: List[Dict[str, Any]], validator_fn, label: str) -> Tuple[List[Dict], List[Dict]]:
    """Validate list of dicts using a validator. Return (validated, errors)."""
    val_ok: List[Dict] = []
    val_bad: List[Dict] = []

    for r in records:
        try:
            v = validator_fn(r)
            # pydantic models may have .model_dump() in v2; keep .dict() for v1 compatibility
            val_ok.append(v.dict() if hasattr(v, "dict") else dict(v))
        except Exception as e:
            val_bad.append({"record": r, "error": str(e), "type": label})

    return val_ok, val_bad

# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────

def main(predict_path: str, feedback_path: str):
    predict_path = Path(predict_path)
    feedback_path = Path(feedback_path)

    out_dir = predict_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    anomalies_path = out_dir / "anomalies.csv"

    print(f"[INFO] Loading logs from:\n  {predict_path}\n  {feedback_path}")

    # 1) Load raw JSONL
    raw_predict = load_jsonl(predict_path)
    raw_feedback = load_jsonl(feedback_path)

    # 2) Map fields to validator schema
    mapped_predict = [map_predict_record(r) for r in raw_predict]
    mapped_feedback = [map_feedback_record(r) for r in raw_feedback]

    # 3) Validate
    ok_pred, bad_pred = safe_validate(mapped_predict, validate_predict_record, "predict")
    ok_feed, bad_feed = safe_validate(mapped_feedback, validate_feedback_record, "feedback")

    # 4) Save cleaned parquet files
    df_pred = pd.DataFrame(ok_pred)
    df_feed = pd.DataFrame(ok_feed)

    pred_parquet = out_dir / "predict.parquet"
    feed_parquet = out_dir / "feedback.parquet"

    if len(df_pred) > 0:
        df_pred.to_parquet(pred_parquet, index=False)
    else:
        print("[WARN] No valid predict records after validation.")


    if len(df_feed) > 0:
        df_feed.to_parquet(feed_parquet, index=False)
    else:
        print("[WARN] No valid feedback records after validation.")

    # NEW: split feedback by accepted_rank
    if "accepted_rank" in df_feed.columns and len(df_feed) > 0:
        # Rows where model suggestion list was used (rank 0–9)
        df_feed_ranked = df_feed[
            df_feed["accepted_rank"].notna()
            & (df_feed["accepted_rank"] >= 0)
            & (df_feed["accepted_rank"] <= 9)
        ]

        # Rows where user typed something outside suggestions (accepted_rank = -1)
        df_feed_manual = df_feed[
            df_feed["accepted_rank"] == -1
        ]

        ranked_parquet = out_dir / "feedback_ranked.parquet"
        manual_parquet = out_dir / "feedback_manual.parquet"

        if len(df_feed_ranked) > 0:
            df_feed_ranked.to_parquet(ranked_parquet, index=False)
            print(f"[OK] Saved ranked feedback (accepted_rank 0–9) -> {ranked_parquet}")
        else:
            print("[INFO] No feedback rows with accepted_rank in [0,9].")

        if len(df_feed_manual) > 0:
            df_feed_manual.to_parquet(manual_parquet, index=False)
            print(f"[OK] Saved manual/novel feedback (accepted_rank = -1) -> {manual_parquet}")
        else:
            print("[INFO] No feedback rows with accepted_rank = -1.")

    print(f"[OK] Saved cleaned parquet to:\n  {pred_parquet if len(df_pred)>0 else '(none)'}\n  {feed_parquet if len(df_feed)>0 else '(none)'}")

    # 5) Report anomalies (schema errors)
    df_bad = pd.DataFrame(bad_pred + bad_feed)
    if len(df_bad) > 0:
        df_bad.to_csv(anomalies_path, index=False)
        print(f"[WARN] Found {len(df_bad)} invalid records -> {anomalies_path}")
    else:
        print("[OK] No schema issues found.")

    # 6) Join check: missing event_id pairs
    unmatched = []
    pred_ids = set(df_pred["event_id"].tolist()) if "event_id" in df_pred.columns else set()
    feed_ids = set(df_feed["event_id"].tolist()) if "event_id" in df_feed.columns else set()

    only_pred = pred_ids - feed_ids
    only_feed = feed_ids - pred_ids

    for eid in sorted(only_pred):
        unmatched.append({"event_id": eid, "side": "predict_only"})

    for eid in sorted(only_feed):
        unmatched.append({"event_id": eid, "side": "feedback_only"})

    if unmatched:
        df_unmatched = pd.DataFrame(unmatched)
        unmatched_path = out_dir / "unmatched_event_ids.csv"
        df_unmatched.to_csv(unmatched_path, index=False)
        print(f"[WARN] Missing matches for {len(unmatched)} event IDs -> {unmatched_path}")
    else:
        print("[OK] Every predict event has a matching feedback event (or no events present).")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", required=True, help="Path to logs/predict.jsonl")
    ap.add_argument("--feedback", required=True, help="Path to logs/feedback.jsonl")
    args = ap.parse_args()
    main(args.predict, args.feedback)
