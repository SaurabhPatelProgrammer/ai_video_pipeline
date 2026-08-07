"""Application orchestration for the Scoop AI edge service."""

from .quality import FrameQualityAssessment, FrameQualityGate

__all__ = ["FrameQualityAssessment", "FrameQualityGate"]
