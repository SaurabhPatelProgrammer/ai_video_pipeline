"""Stable domain contracts for scoop-deposit inference."""

from .deposit_fsm import DepositFSMConfig, DepositState, ReleaseAwareDepositFSM
from .models import BoundingBox, DepositEvent, ObjectClass, Observation

__all__ = [
    "BoundingBox",
    "DepositEvent",
    "DepositFSMConfig",
    "DepositState",
    "ObjectClass",
    "Observation",
    "ReleaseAwareDepositFSM",
]
