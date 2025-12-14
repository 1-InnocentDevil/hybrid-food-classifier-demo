# kbs.py
# Hybrid Knowledge-Based Expert layer (Fuzzy + Symbolic) with safety knobs and
# compatibility with your existing fuzzy_terms.json / symbolic_rules.json.

from __future__ import annotations
import json, re
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np


def _lower_tokens(text: str) -> List[str]:
    # very light tokenization; simple and robust
    return re.findall(r"[a-zA-Z0-9]+(?:'[a-z]+)?", text.lower())


def _term_in_text(term: str, text_lc: str, tokens: List[str]) -> bool:
    t = term.strip().lower()
    if " " in t:
        # phrase match
        return t in text_lc
    else:
        # token match
        return t in tokens


class KBSExpert:
    def __init__(
        self,
        class_order_path: str | Path,
        fuzzy_terms_path: str | Path,
        symbolic_rules_path: str | Path,
        *,
        fuzzy_boost: float = 0.75,
        allow_hard_masks: bool = False,
        # ---- runtime knobs ----
        conf_gate: float = 0.70,        # skip KBS if base max prob >= this
        bias_cap: float | None = 0.80,  # clamp L1|bias| per sample (None disables)
        rule_scale: float = 1.00,       # multiply all symbolic soft-biases
        fuzzy_scale: float = 1.00,      # multiply all fuzzy term contributions
        generic_dampen: float = 0.60    # multiplier when generic words present
    ):
        self.fuzzy_boost = float(fuzzy_boost)
        self.allow_hard_masks = bool(allow_hard_masks)
        self.conf_gate = float(conf_gate)
        self.bias_cap = float(bias_cap) if bias_cap is not None else None
        self.rule_scale = float(rule_scale)
        self.fuzzy_scale = float(fuzzy_scale)
        self.generic_dampen = float(generic_dampen)

        with open(class_order_path, "r", encoding="utf-8") as f:
            self.classes: List[str] = json.load(f)
        with open(fuzzy_terms_path, "r", encoding="utf-8") as f:
            self.fuzzy: Dict[str, List[List[Any]]] = json.load(f)
        with open(symbolic_rules_path, "r", encoding="utf-8") as f:
            self.rules: Dict[str, List[Dict[str, Any]]] = json.load(f)

        self.cls2idx = {c: i for i, c in enumerate(self.classes)}

        # very generic adjectives/adverbs that we should dampen if present
        self._generic_words = {
            "crispy","crunchy","fresh","tasty","yummy","delicious","nice","good","great",
            "hot","spicy","sweet","savoury","salty","sauce","dip","dipping","meal","snack",
            "portion","extra","regular","classic","quick","fast","please","get","want","like",
            "order","small","large","more","less","cheap","budget"
        }

    # ---------- core API ----------
    def adjust_logits(self, text: str, logits: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        text: raw user utterance
        logits: base model scores (shape [C])
        returns: (adjusted_logits, meta)
        """
        # (0) confidence gate: if base is confident, skip intervention
        probs = np.exp(logits - logits.max()); probs /= probs.sum()
        base_conf = float(probs.max())
        if base_conf >= self.conf_gate:
            return logits, {"intervened": False, "bias_norm": 0.0, "reason": "confident"}

        text_lc = text.lower()
        tokens = _lower_tokens(text)

        total_bias = np.zeros_like(logits, dtype=float)
        hard_ban = np.zeros_like(logits, dtype=bool)

        # (1) fuzzy contributions
        fuzz_bias = self._fuzzy_bias(text_lc, tokens)
        total_bias += fuzz_bias

        # (2) symbolic soft-bias
        rule_bias = self._symbolic_soft_bias(text_lc, tokens)
        total_bias += rule_bias

        # (3) optional hard masks
        if self.allow_hard_masks:
            hard_ban |= self._symbolic_hard_mask(text_lc, tokens)

        # (4) clamp overall strength so rules cannot overpower base model
        if self.bias_cap is not None:
            l1 = float(np.linalg.norm(total_bias, ord=1))
            if l1 > self.bias_cap and l1 > 0:
                total_bias *= (self.bias_cap / (l1 + 1e-9))

        logits_adj = logits + total_bias

        # apply hard bans at the very end
        if self.allow_hard_masks and hard_ban.any():
            logits_adj = logits_adj.copy()
            logits_adj[hard_ban] = -1e9

        return logits_adj, {"intervened": True, "bias_norm": float(np.linalg.norm(total_bias, ord=1))}

    # ---------- internals ----------
    def _fuzzy_bias(self, text_lc: str, tokens: List[str]) -> np.ndarray:
        bias = np.zeros(len(self.classes), dtype=float)
        generic_present = any(g in text_lc for g in self._generic_words)
        generic_factor = self.generic_dampen if generic_present else 1.0

        for cls, pairs in self.fuzzy.items():
            if cls not in self.cls2idx:
                continue
            j = self.cls2idx[cls]
            acc = 0.0
            for term, w in pairs:
                if not term:
                    continue
                if _term_in_text(term, text_lc, tokens):
                    acc += float(w)
            if acc != 0.0:
                bias[j] += acc * self.fuzzy_boost * self.fuzzy_scale * generic_factor
        return bias

    def _symbolic_soft_bias(self, text_lc: str, tokens: List[str]) -> np.ndarray:
        bias = np.zeros(len(self.classes), dtype=float)
        for rule in self.rules.get("soft_bias", []):
            if self._rule_triggers(rule, text_lc, tokens):
                b = float(rule.get("bias", 0.0)) * self.rule_scale
                if "then_class" in rule:
                    cls = rule["then_class"]
                    if cls in self.cls2idx:
                        bias[self.cls2idx[cls]] += b
                elif "then_any_of" in rule:
                    for cls in rule["then_any_of"]:
                        if cls in self.cls2idx:
                            bias[self.cls2idx[cls]] += b
        return bias

    def _symbolic_hard_mask(self, text_lc: str, tokens: List[str]) -> np.ndarray:
        mask = np.zeros(len(self.classes), dtype=bool)
        for rule in self.rules.get("hard_mask", []):
            if self._rule_triggers(rule, text_lc, tokens):
                if "ban_any_of" in rule:
                    for cls in rule["ban_any_of"]:
                        if cls in self.cls2idx:
                            mask[self.cls2idx[cls]] = True
        return mask

    def _rule_triggers(self, rule: Dict[str, Any], text_lc: str, tokens: List[str]) -> bool:
        ok_any = True
        ok_all = True
        if "if_any" in rule:
            ok_any = any(_term_in_text(t, text_lc, tokens) for t in rule["if_any"])
        if "if_all" in rule:
            ok_all = all(_term_in_text(t, text_lc, tokens) for t in rule["if_all"])
        return ok_any and ok_all
