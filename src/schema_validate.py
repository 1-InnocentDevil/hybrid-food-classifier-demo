"""
schema_validate.py

Defines pydantic models for validating online logs:
- Prediction events (predict.jsonl)
- Feedback events (feedback.jsonl)

These schemas ensure that:
- Each event has the required fields
- Data types are correct
- Missing/extra fields are detected
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any


# ----------------------------------------------------
# Prediction Event Schema
# ----------------------------------------------------

class PredictEvent(BaseModel):
    event_id: str = Field(..., description="Unique event ID to join prediction ↔ feedback")
    ts: float = Field(..., description="Unix timestamp (float)")
    text: str = Field(..., description="User input text")
    
    label_pred: str = Field(..., description="Predicted top-1 label")
    topk: List[str] = Field(..., description="Top-k predicted labels")
    probs: List[float] = Field(..., description="Corresponding probabilities")
    
    model_version: Optional[str] = Field(
    None,
    description="Model version (optional for older logs)",
    )

    kbs_version: Optional[int] = Field(None, description="KBS config version")

    extra: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional free-form payload for audit/debug"
    )

    @validator("probs")
    def probs_sum_to_one(cls, v):
        if not v:
            return v
        s = sum(v)
        if abs(s - 1.0) > 0.02:
            # Don’t error — log ingestion will warn instead
            print(f"[WARN] Probabilities sum to {s:.3f} != 1")
        return v


# ----------------------------------------------------
# Feedback Event Schema
# ----------------------------------------------------

class FeedbackEvent(BaseModel):
    event_id: str = Field(..., description="Must match PredictEvent.event_id")
    ts: float = Field(..., description="Unix timestamp")
    
    user_action: str = Field(..., description="confirm | correct | flag")
    correct_label: Optional[str] = Field(None, description="Provided when correction")
    reward: Optional[float] = Field(None, description="RL reward signal (optional)")
    
    # NEW: carry richer feedback info
    predicted_label: Optional[str] = Field(
        None,
        description="Model's top-1 suggestion at predict time",
    )
    accepted_rank: Optional[int] = Field(
        None,
        description="0=top-1, 1=top-2, ... 9=top-10, -1=typed/unknown/outside suggestions",
    )
    
    notes: Optional[str] = Field(None, description="Optional short explanation")

# ----------------------------------------------------
# Convenience API
# ----------------------------------------------------

def validate_predict_record(obj: dict) -> PredictEvent:
    """Validate a prediction JSONL record."""
    return PredictEvent(**obj)


def validate_feedback_record(obj: dict) -> FeedbackEvent:
    """Validate a feedback JSONL record."""
    return FeedbackEvent(**obj)
