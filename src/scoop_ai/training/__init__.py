"""Dataset, split, and event-evaluation tooling for scoop AI."""

from .dataset_validation import (
    REQUIRED_CLASSES,
    DatasetValidationError,
    DatasetValidationOptions,
    DatasetValidationReport,
    validate_dataset,
)
from .event_evaluation import EvaluationEvent, EvaluationResult, evaluate_events
from .handover_evaluation import (
    HandoverEvaluationError,
    build_truth_template,
    evaluate_handovers,
    format_summary,
    load_replay_session,
    load_truth,
)
from .session_split import deterministic_session_split
from .motion_baseline import (
    MotionBaselineTrainingResult,
    SessionMotionSummary,
    train_motion_baseline,
    train_motion_baseline_from_videos,
)
from .order_profile import ServedOrderProfileResult, create_served_order_profile
from .replay import run_replay
from .handover_replay import run_handover_replay

__all__ = [
    "DatasetValidationError",
    "DatasetValidationOptions",
    "DatasetValidationReport",
    "EvaluationEvent",
    "EvaluationResult",
    "HandoverEvaluationError",
    "REQUIRED_CLASSES",
    "build_truth_template",
    "evaluate_handovers",
    "format_summary",
    "load_replay_session",
    "load_truth",
    "deterministic_session_split",
    "MotionBaselineTrainingResult",
    "SessionMotionSummary",
    "train_motion_baseline",
    "train_motion_baseline_from_videos",
    "ServedOrderProfileResult",
    "create_served_order_profile",
    "evaluate_events",
    "run_replay",
    "run_handover_replay",
    "validate_dataset",
]
