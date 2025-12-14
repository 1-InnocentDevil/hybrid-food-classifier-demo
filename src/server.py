# src/server.py
import os
import json
import time
import math
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from paths import get_project_root

import numpy as np
from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ───────────────────────────────────────────────────────────────────────────────
# Canonical project paths
# ───────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = get_project_root()

CONFIG_DIR = PROJECT_ROOT / "config"
LOG_DIR    = PROJECT_ROOT / "logs"
PPO_DIR    = PROJECT_ROOT / "models" / "ppo_checkpoints"  # <— canonical PPO dir

CONFIG_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
PPO_DIR.mkdir(parents=True, exist_ok=True)

PREDICT_LOG = LOG_DIR / "predict.jsonl"
FEEDBACK_LOG = LOG_DIR / "feedback.jsonl"

# ───────────────────────────────────────────────────────────────────────────────
# Utils
# ───────────────────────────────────────────────────────────────────────────────

def _json_safe(o):
    import numpy as _np
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, (_np.ndarray,)):
        return o.tolist()
    # fallback
    return str(o)

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def read_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=_json_safe) + "\n")

def find_predict_event(predict_id: str) -> Optional[Dict[str, Any]]:
    """
    Find the predict.jsonl row matching `predict_id`.
    Returns None if not found or if the log file does not exist.
    """
    if not PREDICT_LOG.exists():
        return None

    # Simple linear scan; fine for modest log sizes. You can optimise later.
    with PREDICT_LOG.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except Exception:
                continue
            if evt.get("id") == predict_id:
                return evt
    return None

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

def to_logits_from_probs(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return np.log(p)

def get_latest_file_with_suffix(folder: Path, suffix: str) -> Optional[Path]:
    if not folder.exists():
        return None
    candidates = sorted(
        folder.glob(f"*{suffix}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    return candidates[0] if candidates else None

# ───────────────────────────────────────────────────────────────────────────────
# KBS Loader
# ───────────────────────────────────────────────────────────────────────────────

class NoOpKBS:
    def __init__(self, classes: List[str]):
        self.classes = classes
        self.version = {"kbs_version": 0}
    def adjust_logits(self, text: str, logits: np.ndarray):
        return logits, {"intervened": False, "reason": "noop"}

def load_kbs(classes: List[str]):
    """
    Your kbs.KBSExpert signature:
      KBSExpert(class_order_path, fuzzy_terms_path, symbolic_rules_path, *, ...)
    We pass file PATHS from CONFIG_DIR.
    """
    try:
        # relative import (package: src)
        from kbs import KBSExpert  # type: ignore
    except Exception as e:
        print("[WARN] KBS import failed:", e)
        return NoOpKBS(classes)

    class_order_path    = CONFIG_DIR / "class_order.json"
    fuzzy_terms_path    = CONFIG_DIR / "fuzzy_terms.json"
    symbolic_rules_path = CONFIG_DIR / "symbolic_rules.json"
    version_path        = CONFIG_DIR / "version.json"

    missing = [p for p in [class_order_path, fuzzy_terms_path, symbolic_rules_path] if not p.exists()]
    if missing:
        print("[WARN] KBS config missing:", ", ".join(str(m) for m in missing))
        return NoOpKBS(classes)

    try:
        kbs = KBSExpert(
            class_order_path=str(class_order_path),
            fuzzy_terms_path=str(fuzzy_terms_path),
            symbolic_rules_path=str(symbolic_rules_path),
            # runtime knobs (tune as needed)
            fuzzy_boost=0.75,
            allow_hard_masks=False,
            conf_gate=0.70,
            bias_cap=0.80,
            rule_scale=1.0,
            fuzzy_scale=1.0,
            generic_dampen=0.60,
        )
        try:
            kbs.version = json.loads(version_path.read_text(encoding="utf-8")) if version_path.exists() else {"kbs_version": 0}
        except Exception:
            kbs.version = {"kbs_version": 0}

        # sanity: must return same-shape logits
        test_logits = np.zeros(len(classes), dtype=float)
        adj, _ = kbs.adjust_logits("sanity", test_logits)
        if not (isinstance(adj, np.ndarray) and adj.shape == test_logits.shape):
            print("[WARN] KBS.adjust_logits returned unexpected shape; using NoOpKBS.")
            return NoOpKBS(classes)

        print("[INFO] KBS loaded:", {
            "classes": len(classes),
            "fuzzy_path": str(fuzzy_terms_path),
            "rules_path": str(symbolic_rules_path),
            "version": getattr(kbs, "version", None),
        })
        return kbs

    except Exception as e:
        print("[WARN] KBS init failed; using NoOpKBS:", e)
        return NoOpKBS(classes)

# ───────────────────────────────────────────────────────────────────────────────
# Classifier (PPO-first, graceful fallback to user_model)
# ───────────────────────────────────────────────────────────────────────────────

class HybridOnlineClassifier:
    """
    Uses PPO for online prediction (text-conditioned; obs = BERT embedding).
    Falls back to prototype/BERT scorer if PPO zip not found.
    """
    def __init__(self, classes: List[str]):
        self.classes = classes
        self.num_classes = len(classes)
        self._backend = None
        self._mode = "none"   # "ppo" | "user_model" | "none"
        self._init_backend()

    def _init_backend(self):
        # Try PPO first
        try:
            from ppo_infer import PPOClassifier  # type: ignore
            latest_zip = get_latest_file_with_suffix(PPO_DIR, ".zip")
            # Our PPOClassifier signature is PPOClassifier(ppo_path: Optional[str])
            self._backend = PPOClassifier(ppo_path=str(latest_zip) if latest_zip else None)

            # Sanity: ensure class order consistency (names returned match CONFIG)
            probs, names = self._backend.predict_proba(["sanity"])
            if list(names) != list(self.classes):
                print("[WARN] PPO class order != CONFIG class order; will reorder at inference.")
                # store mapping for reorder at predict
                self._ppo_names = list(names)
            else:
                self._ppo_names = list(self.classes)

            self._mode = "ppo"
            print(f"[INFO] Online backend: PPO ({latest_zip.name if latest_zip else 'auto'})")
            return
        except Exception as e:
            print("[WARN] PPO not available, falling back to user_model:", e)

        # Fall back to user_model logits -> softmax
        try:
            import user_model  # type: ignore
            # sanity
            logits, names = user_model.predict_logits(["sanity"])
            if list(names) != list(self.classes):
                print("[WARN] user_model class order != CONFIG class order; logits will be aligned by index order.")
            self._backend = user_model
            self._mode = "user_model"
            print("[INFO] Online backend: user_model.predict_logits")
            return
        except Exception as e:
            print("[ERROR] No prediction backend available:", e)
            self._mode = "none"

    def reload(self):
        """Re-initialize backend (used when new PPO checkpoint appears or KBS updated)."""
        self._init_backend()

    def predict_logits(self, texts: List[str]) -> np.ndarray:
        """
        Return base (pre-KBS) logits shape [N, C] aligned to CONFIG class order.
        """
        if self._mode == "ppo":
            probs, names = self._backend.predict_proba(texts)  # [N,C], names list
            # Reorder columns if the PPO class order differs from CONFIG class order
            if list(names) != list(self.classes):
                # build index mapping names->position for PPO
                name2idx = {n: i for i, n in enumerate(names)}
                idxs = [name2idx.get(c, None) for c in self.classes]
                # any missing classes become uniform tiny prob
                missing = [i for i, j in enumerate(idxs) if j is None]
                if missing:
                    # grow probs with tiny columns for missing classes
                    probs = np.hstack([probs, np.full((probs.shape[0], len(missing)), 1e-9, dtype=float)])
                    for k, miss_pos in enumerate(missing):
                        idxs[miss_pos] = probs.shape[1] - len(missing) + k
                probs = probs[:, idxs]
            return to_logits_from_probs(probs)

        elif self._mode == "user_model":
            logits, names = self._backend.predict_logits(texts)  # [N,C], names
            if list(names) != list(self.classes):
                # assume indices already aligned by data prep; if not, realign here if you store mapping
                print("[WARN] user_model names != CONFIG; returning logits as-is.")
            return np.asarray(logits, dtype=float)
        else:
            raise RuntimeError("No prediction backend available. Train PPO or implement user_model.predict_logits.")

    def mode(self) -> str:
        return self._mode

# ───────────────────────────────────────────────────────────────────────────────
# FastAPI Schemas
# ───────────────────────────────────────────────────────────────────────────────

class PredictIn(BaseModel):
    text: str = Field(..., description="User input text")
    top_k: int = Field(5, ge=1, le=50, description="Top-K classes to return")
    use_kbs: bool = Field(True, description="Whether to apply KBS adjustments")

class TopClass(BaseModel):
    label: str
    prob: float

class PredictOut(BaseModel):
    id: str
    ts: str
    backend: str
    classes: List[TopClass]
    all_probs: List[float]
    kbs_meta: Dict[str, Any]
    # extra UI/debug
    ranked_labels: List[str]
    ranked_indices: List[int]

class FeedbackIn(BaseModel):
    predict_id: str
    chosen_label: Optional[str] = None
    correct: Optional[bool] = None
    reward: Optional[float] = Field(None, description="Optional scalar reward for RL usage")
    comment: Optional[str] = None

class ReloadOut(BaseModel):
    kbs_version: Any
    backend_mode: str

# ───────────────────────────────────────────────────────────────────────────────
# App init
# ───────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Hybrid KBS + PPO Server", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load classes
CLASS_ORDER_PATH = CONFIG_DIR / "class_order.json"
_classes = read_json(CLASS_ORDER_PATH, default=None)
if not _classes or not isinstance(_classes, list):
    _classes = [f"class_{i}" for i in range(10)]
    print(f"[WARN] Missing {CLASS_ORDER_PATH}. Using placeholder classes: {_classes}")

# Instantiate KBS + Classifier
KBS = load_kbs(_classes)
CLF = HybridOnlineClassifier(_classes)

# ───────────────────────────────────────────────────────────────────────────────
# Minimal HTML UI
# ───────────────────────────────────────────────────────────────────────────────

_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Hybrid KBS + PPO — Interactive Suggestion</title>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>

  <style>
    :root {
      --bg: #f4f3ef;          /* warm soft grey */
      --card: #ffffffdd;      /* soft blurred white */
      --text: #2f3a3f;        /* dark slate */
      --muted: #6c757d;       /* muted slate */
      --accent: #4fa6a6;      /* gentle teal */
      --accent-dark: #3d8d8d;
      --border: #d2d1cd;
    }

    body {
      background: var(--bg);
      color: var(--text);
      margin: 24px;
      font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
      line-height: 1.4;
    }

    textarea {
      width: 100%;
      height: 160px;
      padding: 12px;
      border-radius: 12px;
      border: 1px solid var(--border);
      font-size: 1rem;
      background: var(--card);
      color: var(--text);
    }

    .btn {
      padding: 8px 16px;
      border-radius: 12px;
      border: 1px solid var(--accent-dark);
      background: var(--accent);
      color: white;
      cursor: pointer;
      font-size: 0.95rem;
      transition: background 0.15s ease, transform 0.1s ease;
    }

    .btn:hover {
      background: var(--accent-dark);
    }

    .btn:active {
      transform: scale(0.97);
    }

    .row {
      margin: 14px 0;
    }

    code {
      background: #e8e6e1;
      padding: 2px 8px;
      border-radius: 6px;
      font-size: 0.95rem;
      color: var(--text);
    }

    .muted { color: var(--muted); }

    .bar {
      height: 8px;
      background: #d9e7e7;
      border-radius: 50px;
      overflow: hidden;
    }
    .bar > span {
      display: block;
      height: 100%;
      background: var(--accent);
      width: 0%;
      transition: width .4s ease;
    }

    #prompt { font-size: 1.1rem; }

    #fallback input {
      border-radius: 10px;
      border: 1px solid var(--border);
      padding: 8px;
      background: var(--card);
      width: 60%;
    }

    details summary {
      cursor: pointer;
      color: var(--muted);
      font-size: 0.9rem;
    }

    #result {
      font-size: 1.05rem;
      color: var(--accent-dark);
    }
  </style>
</head>

<body>
  <h2 style="color: var(--accent-dark); font-weight: 600;">Hybrid Knowledge-Based Expert System — Interactive Suggestion</h2>

  <div class="row">
    <em class="muted">Backend:</em> 
    <code id="mode">…</code>
    &nbsp; | &nbsp;
    <em class="muted">KBS version:</em> 
    <code id="kver">…</code>
  </div>

  <div class="row">
    <textarea id="text" placeholder="Tell me what you're craving… e.g., 'crispy spicy fries with aioli'"></textarea>
  </div>

  <div class="row">
    <label class="muted">
      <input id="use_kbs" type="checkbox" checked> Apply KBS
    </label>
    &nbsp;
    <button class="btn" onclick="startFlow()">Suggest</button>
    <button class="btn" onclick="reload()">Reload KBS/Backend</button>
  </div>

  <div id="outTop" class="row muted"></div>

  <div class="row bar">
    <span id="confBar"></span>
  </div>

  <div id="step" class="row" style="display:none;">
    <div id="prompt" style="margin-bottom:10px;"></div>
    <button class="btn" onclick="accept()">Yes</button>
    <button class="btn" onclick="reject()">No</button>
  </div>

  <div id="fallback" class="row" style="display:none;">
    <div style="margin-bottom:8px;">What did you actually want?</div>
    <input id="correct_label" placeholder="Type correct food/class name…" />
    <button class="btn" onclick="submitFallback()">Submit</button>
  </div>

  <div id="result" class="row"></div>

  <div id="debug" class="row">
    <details>
      <summary>Debug</summary>
      <pre id="dbgpre" style="background:#eee; padding:10px; border-radius:12px;"></pre>
    </details>
  </div>

<script>
    // ------- State -------
    let state = {
      predict: null,    // last /predict JSON
      suggIdx: 0,       // 0 = top-1, 1 = top-2
      offered: []       // labels offered in order
    };

    // ------- Helpers -------
    function setText(id, html) { document.getElementById(id).innerHTML = html; }
    function show(id, on=true) { document.getElementById(id).style.display = on ? "" : "none"; }
    function val(id) { return document.getElementById(id).value; }
    function checked(id) { return document.getElementById(id).checked; }
    function setConfBar(p) {
      const pct = Math.max(0, Math.min(1, p)) * 100;
      document.getElementById("confBar").style.width = pct.toFixed(1) + "%";
    }

    async function info() {
      const r = await fetch('/admin/reload', {method:'POST'});
      const j = await r.json();
      setText('mode', j.backend_mode);
      setText('kver', JSON.stringify(j.kbs_version));
      return j;
    }

    function showSuggestion() {
      const labels = state.predict?.ranked_labels || [];
      if (state.suggIdx >= labels.length) { showFallback(); return; }
      const label = labels[state.suggIdx];
      if (!state.offered.includes(label)) state.offered.push(label);

      // quick confidence display (top-1 prob if available)
      const pAll = state.predict?.all_probs || [];
      const p1 = pAll.length ? pAll[state.predict.ranked_indices[0]] : null;
      if (p1 != null) {
        setConfBar(p1);
        setText("outTop", `Top-1 confidence: ${(p1*100).toFixed(1)}%`);
      }

      setText('prompt', `Do you want <b>${label}</b>?`);
      show('step', true);
      show('fallback', false);
    }

    function showFallback() {
      show('step', false);
      show('fallback', true);
    }

    // ------- Predict flow -------
    async function startFlow() {
      const text = val('text').trim();
      if (!text) { alert("Please paste a description first."); return; }
      const use_kbs = checked('use_kbs');

      // Always request full ranking (top_k=50) so we have top-2 available
      const r = await fetch('/predict', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({text, top_k: 50, use_kbs})
      });
      const j = await r.json();
      state.predict = j;
      state.suggIdx = 0;
      state.offered = [];
      setText('dbgpre', JSON.stringify(j, null, 2));
      setText('result', '');
      showSuggestion();
    }

    // ------- Feedback flow -------
    async function accept() {
      const labels = state.predict?.ranked_labels || [];
      const label = labels[state.suggIdx];
      if (!label) return;

      // reward mapping: top-1 accept => 1.0, top-2 accept => 0.5
      const reward = (state.suggIdx === 0) ? 1.0 : 0.5;
      await sendFeedback(label, true, reward, `accepted_rank_${state.suggIdx+1}`);
      setText('result', `✅ Thanks! Logged your confirmation for <b>${label}</b> (reward=${reward}).`);
      show('step', false);
      show('fallback', false);
    }

    async function reject() {
      if (state.suggIdx === 0) {
        // move to top-2 suggestion
        state.suggIdx = 1;
        showSuggestion();
      } else {
        // after top-2 rejection, ask for correct label (no reward)
        showFallback();
      }
    }

    async function submitFallback() {
      const manual = val('correct_label').trim();
      if (!manual) { alert("Please type the correct item."); return; }
      // wrong on both top-1 and top-2 → reward 0.0
      await sendFeedback(manual, false, 0.0, `manual_correction_after_top2`, /*isManual=*/true);
      setText('result', `✍️ Logged your correction as <b>${manual}</b> (reward=0.0).`);
      show('step', false);
      show('fallback', false);
      document.getElementById('correct_label').value = '';
    }

    async function sendFeedback(chosen_label, correct, reward, comment, isManual=false) {
      const body = {
        predict_id: state.predict?.id,
        chosen_label,
        correct,
        reward,
        comment: comment + (isManual ? " | offered=" + JSON.stringify(state.offered) : "")
      };
      try {
        await fetch('/feedback', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify(body)
        });
      } catch (e) {
        console.error(e);
      }
    }

    async function doReload() {
      await info();
      alert('Reloaded KBS/Backend');
    }

    // ------- Wire up buttons & initial info -------
    document.getElementById('btnSuggest').addEventListener('click', startFlow);
    document.getElementById('btnReload').addEventListener('click', doReload);
    document.getElementById('btnYes').addEventListener('click', accept);
    document.getElementById('btnNo').addEventListener('click', reject);
    document.getElementById('btnSubmitManual').addEventListener('click', submitFallback);

    info(); // populate header on load
  </script>
</body>
</html>
"""


# ───────────────────────────────────────────────────────────────────────────────
# Routes
# ───────────────────────────────────────────────────────────────────────────────

class PredictIn(BaseModel):
    text: str = Field(..., description="User input text")
    top_k: int = Field(5, ge=1, le=50, description="Top-K classes to return")
    use_kbs: bool = Field(True, description="Whether to apply KBS adjustments")

class TopClass(BaseModel):
    label: str
    prob: float

class PredictOut(BaseModel):
    id: str
    ts: str
    backend: str
    classes: List[TopClass]
    all_probs: List[float]
    kbs_meta: Dict[str, Any]
    ranked_labels: List[str]
    ranked_indices: List[int]

@app.get("/", response_class=None)
def root():
    from fastapi.responses import HTMLResponse
    return HTMLResponse(_HTML)

@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn = Body(...)):
    if not inp.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    try:
        base_logits = CLF.predict_logits([inp.text])  # [1, C] aligned to CONFIG
        base_logits = base_logits.reshape(1, -1)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction backend failed: {e}")

    logits = base_logits[0]
    kbs_meta: Dict[str, Any] = {"intervened": False}
    if inp.use_kbs:
        try:
            logits, kbs_meta = KBS.adjust_logits(inp.text, logits)
        except Exception as e:
            print("[WARN] KBS.adjust_logits failed; returning base scores.", e)
            traceback.print_exc()

    probs = softmax(logits[None, :])[0]  # [C]

    ranked_indices = list(np.argsort(probs)[::-1])
    ranked_labels = [_classes[i] for i in ranked_indices]

    order = ranked_indices[: inp.top_k]
    classes_out = [
        TopClass(label=_classes[i], prob=float(probs[i]))
        for i in order
    ]

    predict_id_str = f"{int(time.time()*1000)}-{os.getpid()}"
    log_row = {
        "id": predict_id_str,
        "ts": utc_now_iso(),
        "backend": CLF.mode(),
        "use_kbs": inp.use_kbs,
        "text": inp.text,
        "classes": _classes,
        "probs": probs.tolist(),
        "top_k": inp.top_k,
        "kbs_meta": kbs_meta,
        "ranked_labels": ranked_labels,
        "ranked_indices": ranked_indices,
    }
    append_jsonl(PREDICT_LOG, log_row)

    return PredictOut(
        id=predict_id_str,
        ts=utc_now_iso(),
        backend=CLF.mode(),
        classes=classes_out,
        all_probs=[float(x) for x in probs],
        kbs_meta=kbs_meta,
        ranked_labels=ranked_labels,
        ranked_indices=[int(i) for i in ranked_indices],
    )

class FeedbackIn(BaseModel):
    predict_id: str
    chosen_label: Optional[str] = None
    correct: Optional[bool] = None
    reward: Optional[float] = Field(None, description="Optional scalar reward for RL usage")
    comment: Optional[str] = None

@app.post("/feedback")
def feedback(inp: FeedbackIn = Body(...)):
    if not inp.predict_id:
        raise HTTPException(status_code=400, detail="predict_id is required")

    # Look up the original prediction to enrich feedback
    predict_evt = find_predict_event(inp.predict_id)

    text = None
    predicted_label = None
    accepted_rank = None

    if predict_evt is not None:
        text = predict_evt.get("text")

        ranked_labels = predict_evt.get("ranked_labels") or []
        if ranked_labels:
            predicted_label = ranked_labels[0]  # model top-1

        if inp.chosen_label:
            if inp.chosen_label in ranked_labels:
                accepted_rank = int(ranked_labels.index(inp.chosen_label))
            else:
                # user provided a label not in the suggestion list
                accepted_rank = -1

    # For clarity, treat chosen_label as the correct/final label the user gave
    correct_label = inp.chosen_label

    row = {
        "ts": utc_now_iso(),
        "predict_id": inp.predict_id,

        # new, richer fields:
        "text": text,                         # user description (from /predict)
        "predicted_label": predicted_label,   # model's top-1 suggestion
        "correct_label": correct_label,       # final accepted / supplied label
        "accepted_rank": accepted_rank,       # 0=top-1, 1=top-2, -1=manual/outside-list

        # keep old fields for compatibility:
        "chosen_label": inp.chosen_label,
        "correct": inp.correct,
        "reward": inp.reward,
        "comment": inp.comment,
    }

    append_jsonl(FEEDBACK_LOG, row)
    return {"ok": True}

class ReloadOut(BaseModel):
    kbs_version: Any
    backend_mode: str

@app.post("/admin/reload", response_model=ReloadOut)
def admin_reload():
    global KBS, CLF
    KBS = load_kbs(_classes)
    CLF.reload()
    kver = getattr(KBS, "version", {"kbs_version": 0})
    return ReloadOut(kbs_version=kver, backend_mode=CLF.mode())

@app.get("/healthz")
def healthz():
    return {"ok": True, "backend": CLF.mode()}

# ───────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("server:app", host=host, port=port, reload=True)

# Run:
# cd "<repo_root>/src"
# python server.py
# http://localhost:8000/

