# src/mine_kbs_candidates.py
# Robust fuzzy-term & symbolic suggestions from logs (parquet/jsonl).
# Works with the ingested parquet schema written by ingest_logs.py.
#
# Usage (defaults to canonical paths):
#   python src/mine_kbs_candidates.py
#   python src/mine_kbs_candidates.py --apply-fuzzy --apply-symbolic
#   python src/mine_kbs_candidates.py --logs "<ROOT>/logs" --config "<ROOT>/config"
#   python src/mine_kbs_candidates.py --fallback-top1   # treat top1 as truth without reward
#
from __future__ import annotations
from pathlib import Path
import re, json, time, argparse, collections, math
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ───────────────────────────────────────────────────────────────────────────────
# Canonical project paths (overridable via CLI)
# ───────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(r"C:\Users\sup028\OneDrive - University of Salford\Hybrid Knowledge-Based Expert System")

DEFAULT_LOGS_DIR   = PROJECT_ROOT / "logs"
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_REPORTS    = PROJECT_ROOT / "reports"

# File names we expect (from ingest_logs.py output)
PREDICT_PARQ  = "predict.parquet"
FEEDBACK_PARQ = "feedback.parquet"
PREDICT_JSONL = "predict.jsonl"
FEEDBACK_JSONL= "feedback.jsonl"

CLASS_ORDER_FILE    = "class_order.json"
FUZZY_TERMS_FILE    = "fuzzy_terms.json"
SYMBOLIC_RULES_FILE = "symbolic_rules.json"
BACKUPS_DIRNAME     = "_backups"

# ───────────────────────────────────────────────────────────────────────────────
# Mining knobs
# ───────────────────────────────────────────────────────────────────────────────
MIN_COUNT = 1                  # min occurrences per class to keep a term (liberal for small logs)
TOP_PER_CLASS = 30
STOPWORDS = {
    "please","want","like","get","really","very","more","some","extra",
    "with","and","or","for","the","a","an","of","to","me","my","your",
    "is","it","that","this","on","in","at","from","be","can","need"
}
MAX_EXAMPLES_PER_TOKEN = 3
MIN_PMI = 2.0
MAX_NGRAM_LEN = 3
SOFT_BIAS_WEIGHT = 0.3
SOFT_BIAS_MIN_COUNT = 3
ENABLE_HARD_MASK_SUGGESTIONS = False  # placeholder if you later implement hard masks

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:'[a-z]+)?")

# ───────────────────────────────────────────────────────────────────────────────
# Utilities
# ───────────────────────────────────────────────────────────────────────────────

def timestamp_dir(parent: Path, prefix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = parent / f"{prefix}_{ts}"
    out.mkdir(parents=True, exist_ok=False)
    return out

def read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                pass
    return pd.DataFrame(rows)

def _iso_to_ts(iso_str: str) -> float:
    from datetime import datetime
    if not iso_str:
        return 0.0
    s = iso_str.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        try:
            return float(pd.Timestamp(s).timestamp())
        except Exception:
            return 0.0

def load_classes(config_dir: Path) -> List[str]:
    p = config_dir / CLASS_ORDER_FILE
    if not p.exists():
        raise FileNotFoundError(f"Missing class order: {p}")
    return json.loads(p.read_text(encoding="utf-8"))

def load_logs_any(logs_dir: Path) -> pd.DataFrame:
    """
    Load joined predict/feedback logs, preferring parquet (from ingest_logs.py).
    Falls back to raw JSONL if parquet is missing.
    Output columns (normalized):
      - event_id, ts (float), text (str), correct_label (opt), reward (opt),
        label_pred (opt), probs (opt list[float]), ranked_labels/indices (opt), extra (opt dict)
    """
    pred_parq   = logs_dir / PREDICT_PARQ
    feed_parq   = logs_dir / FEEDBACK_PARQ
    pred_jsonl  = logs_dir / PREDICT_JSONL
    feed_jsonl  = logs_dir / FEEDBACK_JSONL

    # -------- Predict side --------
    if pred_parq.exists():
        pred = pd.read_parquet(pred_parq)
    else:
        raw = read_jsonl(pred_jsonl)
        if raw.empty:
            return pd.DataFrame()
        def map_pred_row(r: Dict[str, Any]) -> Dict[str, Any]:
            probs = r.get("probs")
            classes = r.get("classes", [])
            # derive top-1 label
            if isinstance(r.get("ranked_labels"), list) and r["ranked_labels"]:
                label_pred = r["ranked_labels"][0]
            elif isinstance(probs, list) and classes:
                i = int(np.argmax(np.asarray(probs, dtype=float)))
                label_pred = classes[i]
            else:
                label_pred = None
            return {
                # normalize to event_id
                "event_id": r.get("id") or r.get("event_id"),
                "ts": _iso_to_ts(r.get("ts")),
                "text": r.get("text", ""),
                "label_pred": label_pred,
                "probs": probs if isinstance(probs, list) else None,
                "ranked_labels": r.get("ranked_labels"),
                "ranked_indices": r.get("ranked_indices"),
                "extra": {
                    "backend": r.get("backend"),
                    "use_kbs": r.get("use_kbs"),
                    "kbs_meta": r.get("kbs_meta"),
                    "classes": classes,
                },
            }
        pred = pd.DataFrame([map_pred_row(x) for x in raw.to_dict(orient="records")])

    # If parquet/older schema: rename id -> event_id if needed
    if "event_id" not in pred.columns:
        if "id" in pred.columns:
            pred = pred.rename(columns={"id": "event_id"})
        elif "predict_id" in pred.columns:
            pred = pred.rename(columns={"predict_id": "event_id"})
        # else: leave as is; merge will handle below

    # -------- Feedback side --------
    if feed_parq.exists():
        fb = pd.read_parquet(feed_parq)
    else:
        rawf = read_jsonl(feed_jsonl)
        if rawf.empty:
            fb = pd.DataFrame()
        else:
            def map_fb_row(r: Dict[str, Any]) -> Dict[str, Any]:
                user_action = "flag"
                reward = r.get("reward", None)
                chosen = r.get("chosen_label", None)
                correct_flag = r.get("correct", None)
                if reward == 1 and (correct_flag is True or chosen is None):
                    user_action = "confirm"
                elif reward == -1 and chosen:
                    user_action = "correct"
                elif correct_flag is False and chosen:
                    user_action = "correct"
                return {
                    # normalize to event_id
                    "event_id": r.get("predict_id") or r.get("event_id") or r.get("id"),
                    "ts": _iso_to_ts(r.get("ts")),
                    "user_action": user_action,
                    "correct_label": chosen,
                    "reward": reward if isinstance(reward, (int, float)) else None,
                    "extra": {"correct_flag": correct_flag, "comment": r.get("comment")},
                }
            fb = pd.DataFrame([map_fb_row(x) for x in rawf.to_dict(orient="records")])

    if pred.empty:
        return pd.DataFrame()

    # standardize feedback id column name if needed
    if not fb.empty and "event_id" not in fb.columns:
        if "predict_id" in fb.columns:
            fb = fb.rename(columns={"predict_id": "event_id"})
        elif "id" in fb.columns:
            fb = fb.rename(columns={"id": "event_id"})

    # dedupe feedback by latest ts per event_id
    if not fb.empty and "event_id" in fb.columns:
        fb = fb.sort_values("ts").drop_duplicates("event_id", keep="last")

    # robust merge: choose available key names
    def _key(df):
        return "event_id" if "event_id" in df.columns else ("id" if "id" in df.columns else None)

    pred_key = _key(pred)
    fb_key   = _key(fb) if not fb.empty else None

    if fb_key is None:
        df = pred.copy()
    elif pred_key == fb_key:
        df = pred.merge(fb, on=pred_key, how="left", suffixes=("", "_fb"))
    else:
        df = pred.merge(fb, left_on=pred_key, right_on=fb_key, how="left", suffixes=("", "_fb"))

    # guarantee a canonical event_id column exists in the output
    if "event_id" not in df.columns:
        if pred_key and pred_key in df.columns:
            df = df.rename(columns={pred_key: "event_id"})
        elif fb_key and fb_key in df.columns:
            df = df.rename(columns={fb_key: "event_id"})
        else:
            df["event_id"] = None  # last resort (shouldn't happen)

    # ensure text exists
    if "text" not in df.columns:
        df["text"] = ""
    df["text"] = df["text"].astype(str)

    return df

def safe_list(x): return x if isinstance(x, list) else []

def recover_top1(row: pd.Series, classes: List[str]) -> str | None:
    """
    Prefer ingested fields first:
      1) label_pred (string)
      2) ranked_labels[0]
      3) argmax over probs + (extra.classes if available, else global classes)
    """
    if isinstance(row.get("label_pred"), str) and row["label_pred"]:
        return row["label_pred"]

    ranked = row.get("ranked_labels")
    if isinstance(ranked, list) and ranked:
        return str(ranked[0])

    probs = row.get("probs")
    if isinstance(probs, list) and probs:
        local_classes = None
        extra = row.get("extra")
        if isinstance(extra, dict) and isinstance(extra.get("classes"), list) and extra["classes"]:
            local_classes = extra["classes"]
        arr = np.asarray(probs, dtype=float)
        if local_classes and len(local_classes) == len(arr):
            return str(local_classes[int(np.argmax(arr))])
        # fall back to global classes if same length
        if len(classes) == len(arr):
            return str(classes[int(np.argmax(arr))])

    return None

def tok_unigrams(s: str) -> list[str]:
    return [w for w in TOKEN_RE.findall((s or "").lower()) if len(w) >= 3]

def gen_ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def compute_ngram_stats(texts: list[str], max_n=3):
    uni_ctr = collections.Counter()
    bi_ctr  = collections.Counter()
    tri_ctr = collections.Counter()
    doc_freq = collections.Counter()
    for s in texts:
        toks = tok_unigrams(s)
        seen = set()
        if max_n >= 1:
            uni_ctr.update(toks); seen.update(toks)
        if max_n >= 2:
            bigrams = gen_ngrams(toks, 2); bi_ctr.update(bigrams); seen.update(bigrams)
        if max_n >= 3:
            trigrams = gen_ngrams(toks, 3); tri_ctr.update(trigrams); seen.update(trigrams)
        for t in seen:
            doc_freq[t] += 1
    return uni_ctr, bi_ctr, tri_ctr, doc_freq

def pmi_like(bigram: str, uni_ctr: collections.Counter, bi_ctr: collections.Counter, total_unis: int):
    if " " not in bigram: return 0.0
    x, y = bigram.split(" ", 1)
    c_xy = bi_ctr[bigram]; c_x = uni_ctr[x]; c_y = uni_ctr[y]
    if c_xy < 1 or c_x < 1 or c_y < 1: return 0.0
    return math.log((c_xy * total_unis) / (c_x * c_y + 1e-9) + 1e-9, 2)

def tfidf_weight(cnt: int, max_cnt: int, df_term: int, total_docs: int) -> float:
    tf = cnt / max(max_cnt, 1)
    idf = math.log(1.0 + total_docs / max(1, df_term))
    return tf * idf

# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs",   default=str(DEFAULT_LOGS_DIR),   help="Logs directory (contains predict.parquet/feedback.parquet or jsonl)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_DIR), help="Config directory (class_order.json, fuzzy_terms.json, symbolic_rules.json)")
    ap.add_argument("--reports",default=str(DEFAULT_REPORTS),    help="Where to write mining reports")
    ap.add_argument("--apply-fuzzy", action="store_true", help="Merge into config/fuzzy_terms.json (with backup)")
    ap.add_argument("--apply-symbolic", action="store_true", help="Append soft_bias into config/symbolic_rules.json (with backup)")
    ap.add_argument("--fallback-top1", action="store_true", help="Treat top-1 as truth even without reward/correct_label")
    args = ap.parse_args()

    LOGS_DIR = Path(args.logs)
    CONFIG_DIR = Path(args.config)
    REPORTS_DIR = Path(args.reports)

    print("=== Mine KBS Candidates ===")
    classes: List[str] = load_classes(CONFIG_DIR)

    df = load_logs_any(LOGS_DIR)
    if df.empty:
        print("[x] No logs found. Run server, generate predictions/feedback, then ingest.")
        return

    # Recover top-1 per row (post-KBS) and truth label
    df["pred_class"] = df.apply(lambda r: recover_top1(r, classes), axis=1)

    def truth(row):
        corr = row.get("correct_label")
        if isinstance(corr, str) and corr in classes:
            return corr
        r = row.get("reward", None)
        if r is not None:
            try:
                if float(r) > 0:  # positive reward
                    return row.get("pred_class")
            except Exception:
                pass
        if args.fallback_top1:
            return row.get("pred_class")
        return None

    df["truth_cls"] = df.apply(truth, axis=1)
    total_rows = len(df)
    usable = df[~df["truth_cls"].isna()].copy()

    pos_rewards = int((df["reward"] > 0).sum()) if "reward" in df.columns else 0
    with_corr   = int(df["correct_label"].notna().sum()) if "correct_label" in df.columns else 0
    print(f"[i] Rows: total={total_rows}, with_correct_label={with_corr}, reward_pos={pos_rewards}, usable={len(usable)}")

    if usable.empty:
        print("[x] No usable rows (need correct_label or reward>0, or pass --fallback-top1).")
        return

    # Per-class texts
    texts_by_cls: Dict[str, List[str]] = {c: [] for c in classes}
    for t, c in zip(usable["text"].astype(str), usable["truth_cls"]):
        if isinstance(c, str) and c in texts_by_cls:
            texts_by_cls[c].append(t)

    # Existing fuzzy terms to avoid duplicates
    fuzzy_path = CONFIG_DIR / FUZZY_TERMS_FILE
    try:
        fuzzy_cfg = json.loads(fuzzy_path.read_text(encoding="utf-8"))
    except Exception:
        fuzzy_cfg = {}
    already = {c: set(t for t, _ in fuzzy_cfg.get(c, [])) for c in classes}

    # Global stats
    global_texts = usable["text"].astype(str).tolist()
    uni_all, bi_all, tri_all, df_all = compute_ngram_stats(global_texts, max_n=MAX_NGRAM_LEN)
    total_docs = len(global_texts)
    total_unis = sum(uni_all.values())

    out_rows: List[Dict[str, Any]] = []
    suggestions: Dict[str, List[List[Any]]] = {c: [] for c in classes}
    examples: Dict[str, Dict[str, List[str]]] = {c: {} for c in classes}

    for c in classes:
        texts = texts_by_cls[c]
        if not texts:
            continue
        uni_c, bi_c, tri_c, df_c = compute_ngram_stats(texts, max_n=MAX_NGRAM_LEN)
        max_cnt_uni = max(uni_c.values() or [1])
        max_cnt_bi  = max(bi_c.values()  or [1])
        max_cnt_tri = max(tri_c.values() or [1])

        local_candidates: list[tuple[str, int, float]] = []

        # Unigrams
        for term, cnt in uni_c.most_common(TOP_PER_CLASS * 5):
            if cnt < MIN_COUNT:
                break
            if term in STOPWORDS or term in already[c]:
                continue
            tfidf = tfidf_weight(cnt, max_cnt_uni, df_all[term], total_docs)
            lift = (cnt / max_cnt_uni) * math.log1p(cnt) / max(1.0, math.log(df_all[term] + 1))
            score = tfidf + 0.25 * lift
            local_candidates.append((term, cnt, score))

        # Bigrams
        if MAX_NGRAM_LEN >= 2:
            for term, cnt in bi_c.most_common(TOP_PER_CLASS * 5):
                if cnt < MIN_COUNT:
                    break
                if term in already[c]:
                    continue
                pmi = pmi_like(term, uni_all, bi_all, total_unis)
                if pmi < MIN_PMI:
                    continue
                tfidf = tfidf_weight(cnt, max_cnt_bi, df_all[term], total_docs)
                score = tfidf + 0.5 * pmi
                local_candidates.append((term, cnt, score))

        # Trigrams
        if MAX_NGRAM_LEN >= 3:
            for term, cnt in tri_c.most_common(TOP_PER_CLASS * 5):
                if cnt < MIN_COUNT:
                    break
                if term in already[c]:
                    continue
                parts = term.split(" ")
                if len(parts) != 3:
                    continue
                pmi12 = pmi_like(" ".join(parts[:2]), uni_all, bi_all, total_unis)
                pmi23 = pmi_like(" ".join(parts[1:]), uni_all, bi_all, total_unis)
                pmi = 0.5 * (pmi12 + pmi23)
                if pmi < MIN_PMI:
                    continue
                tfidf = tfidf_weight(cnt, max_cnt_tri, df_all[term], total_docs)
                score = tfidf + 0.5 * pmi
                local_candidates.append((term, cnt, score))

        local_candidates.sort(key=lambda x: x[2], reverse=True)
        keep = local_candidates[:TOP_PER_CLASS]
        if keep:
            scores = np.array([s for _, _, s in keep], dtype=float)
            if scores.max() > scores.min():
                scaled = 0.2 + 0.6 * (scores - scores.min()) / (scores.max() - scores.min())
            else:
                scaled = np.full_like(scores, 0.5)
            for (term, cnt, _), w in zip(keep, scaled):
                w = float(round(float(w), 3))
                suggestions[c].append([term, w])
                out_rows.append({"class": c, "term": term, "weight": w, "count": cnt})
                # collect few example snippets
                ex_list = []
                for s in texts:
                    if term in s.lower():
                        ex_list.append(s)
                        if len(ex_list) >= MAX_EXAMPLES_PER_TOKEN:
                            break
                examples[c][term] = ex_list

    # Simple symbolic soft_bias proposals
    symbolic_soft_bias = []
    ctr_by_cls = {c: collections.Counter() for c in classes}
    for r in out_rows:
        ctr_by_cls[r["class"]][r["term"]] += r["count"]
    for c in classes:
        for term, cnt in ctr_by_cls[c].most_common(10):
            if cnt < SOFT_BIAS_MIN_COUNT:
                break
            symbolic_soft_bias.append({"if_any": [term], "then_class": c, "bias": float(SOFT_BIAS_WEIGHT)})

    # Write reports
    out_dir = timestamp_dir(REPORTS_DIR, "kbs_mining")
    (out_dir / "examples").mkdir(parents=True, exist_ok=True)

    tok_csv  = out_dir / "kbs_tokens_by_class.csv"
    sugg_json= out_dir / "fuzzy_term_suggestions.json"
    sym_json = out_dir / "symbolic_soft_bias_suggestions.json"

    pd.DataFrame(out_rows).to_csv(tok_csv, index=False)
    sugg_json.write_text(json.dumps(suggestions, indent=2, ensure_ascii=False), encoding="utf-8")
    for c in classes:
        (out_dir / "examples" / f"{c}.json").write_text(json.dumps(examples[c], indent=2, ensure_ascii=False), encoding="utf-8")
    sym_json.write_text(json.dumps(symbolic_soft_bias, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[✓] Fuzzy suggestions →", sugg_json.resolve())
    print("[✓] Token report     →", tok_csv.resolve())
    print("[✓] Example snippets →", (out_dir / "examples").resolve())
    print("[✓] Symbolic soft_bias suggestions →", sym_json.resolve())

    # Optional: apply to config with backup
    def backup_and_write(path: Path, content: str):
        backups = (path.parent / BACKUPS_DIRNAME)
        backups.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        if path.exists():
            path.replace(backups / f"{path.stem}_{ts}.bak")
        path.write_text(content, encoding="utf-8")

    fuzzy_terms_path = CONFIG_DIR / FUZZY_TERMS_FILE
    symbolic_rules_path = CONFIG_DIR / SYMBOLIC_RULES_FILE

    if args.apply_fuzzy:
        try:
            existing = json.loads(fuzzy_terms_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        merged = {}
        for c in classes:
            exist = {t: w for t, w in existing.get(c, [])}
            for t, w in suggestions.get(c, []):
                if t not in exist:
                    exist[t] = w
            merged[c] = sorted([[t, float(w)] for t, w in exist.items()], key=lambda x: (-x[1], x[0]))[:200]
        backup_and_write(fuzzy_terms_path, json.dumps(merged, indent=2, ensure_ascii=False))
        print("[✓] Updated config/fuzzy_terms.json (backup in config/_backups)")

    if args.apply_symbolic:
        try:
            old = json.loads(symbolic_rules_path.read_text(encoding="utf-8"))
        except Exception:
            old = {"soft_bias": [], "hard_mask": []}
        soft_bias = old.get("soft_bias", [])
        soft_bias.extend(symbolic_soft_bias)
        new_rules = {"soft_bias": soft_bias, "hard_mask": old.get("hard_mask", [])}
        backup_and_write(symbolic_rules_path, json.dumps(new_rules, indent=2, ensure_ascii=False))
        print("[✓] Updated config/symbolic_rules.json (backup in config/_backups)")

if __name__ == "__main__":
    main()
