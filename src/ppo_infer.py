from __future__ import annotations
import glob, os
from pathlib import Path
from typing import List, Tuple
import numpy as np

import torch
from stable_baselines3 import PPO

# Reuse encoder & prototypes from user_model.py
from user_model import (
    PROJECT_ROOT, encode_texts, build_or_load_prototypes, cosine_logits
)

# Canonical PPO path
PPO_CKPT_DIR = PROJECT_ROOT / "models" / "ppo_checkpoints"

class PPOClassifier:
    """
    Text-conditioned PPO inference:
      1) Encode text -> embedding (L2-normalized)
      2) PPO policy maps obs=embedding -> action vector (D,)
      3) Cosine(action, class_prototype) -> logits -> softmax
    Requires PPO trained in the same Box-observation env (D-dim obs/action).
    """

    def __init__(self, ppo_path: str | None = None):
        self.ppo_path = self._resolve_ckpt(ppo_path)
        self.model = PPO.load(str(self.ppo_path))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load prototypes once
        self.prototypes, self.class_names = build_or_load_prototypes()
        self.D = self.prototypes.shape[1]

        # Sanity: policy action dim must equal embedding dim
        act_dim = int(self.model.policy.action_space.shape[0])
        if act_dim != self.D:
            raise ValueError(f"Action dim {act_dim} != prototype/embedding dim {self.D}")

    def _resolve_ckpt(self, explicit: str | None) -> Path:
        if explicit:
            p = Path(explicit)
            if not p.exists():
                raise FileNotFoundError(f"PPO checkpoint not found: {explicit}")
            return p
        best = PPO_CKPT_DIR / "best_model.zip"
        if best.exists():
            return best
        cand = sorted(glob.glob(str(PPO_CKPT_DIR / "ppo_*.zip")))
        if not cand:
            raise FileNotFoundError(f"No PPO checkpoints under {PPO_CKPT_DIR}")
        return Path(cand[-1])

    def _obs_from_texts(self, texts: List[str]) -> np.ndarray:
        # Encoder from user_model; returns [N,D] normalized embeddings
        return encode_texts(texts)

    def _policy_action_embeddings(self, obs: np.ndarray) -> np.ndarray:
        # obs: [N,D]; SB3 predict works one sample at a time for numpy inputs
        outs = []
        for i in range(obs.shape[0]):
            a, _ = self.model.predict(obs[i], deterministic=True)
            # normalize action to unit length to keep cosine scale meaningful
            a = np.asarray(a, dtype=np.float32).reshape(-1)
            n = np.linalg.norm(a)
            outs.append(a if n == 0 else a / (n + 1e-9))
        return np.vstack(outs)  # [N,D]

    def predict_proba(self, texts: List[str]) -> Tuple[np.ndarray, List[str]]:
        obs = self._obs_from_texts(texts)               # [N,D]
        acts = self._policy_action_embeddings(obs)      # [N,D]
        logits = cosine_logits(acts, self.prototypes)   # [N,C]
        probs = self._softmax(logits, axis=1)
        return probs, self.class_names

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x = x - np.max(x, axis=axis, keepdims=True)
        e = np.exp(x)
        return e / (np.sum(e, axis=axis, keepdims=True) + 1e-9)

if __name__ == "__main__":
    clf = PPOClassifier()  # auto-picks best_model.zip or latest ppo_*.zip
    demo = ["I love spicy crispy fries with aioli.", "Fresh salad with feta and olives."]
    probs, names = clf.predict_proba(demo)
    print("Classes:", names[:5], "… (total:", len(names), ")")
    print("Probs shape:", probs.shape)
    print("Top-1:", [names[int(np.argmax(p))] for p in probs])
