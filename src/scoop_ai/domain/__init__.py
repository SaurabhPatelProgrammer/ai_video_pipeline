"""Stable domain contracts for scoop-deposit inference."""

from .deposit_fsm import DepositFSMConfig, DepositState, ReleaseAwareDepositFSM
from .handover_fsm import HandoverFSMConfig, HandoverState, IceCreamHandoverFSM
from .models import BoundingBox, DepositEvent, HandoverEvent, ObjectClass, Observation

__all__ = [
    "BoundingBox",
    "DepositEvent",
    "DepositFSMConfig",
    "DepositState",
    "HandoverEvent",
    "HandoverFSMConfig",
    "HandoverState",
    "IceCreamHandoverFSM",
    "ObjectClass",
    "Observation",
    "ReleaseAwareDepositFSM",
]
