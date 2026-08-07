"""Deterministic, leakage-resistant assignment of whole sessions to splits."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from math import floor, isfinite


DEFAULT_RATIOS = {"train": 0.70, "valid": 0.15, "test": 0.15}


def deterministic_session_split(
    session_ids: Iterable[str],
    *,
    ratios: Mapping[str, float] | None = None,
    seed: int = 42,
) -> dict[str, str]:
    """Return ``session_id -> split`` without ever splitting one session.

    Session ordering is derived from SHA-256 instead of process-randomized hash
    values. Quotas use largest-remainder apportionment and guarantee at least one
    session in every requested split.
    """

    normalized_sessions = []
    seen: set[str] = set()
    for raw_session in session_ids:
        session = str(raw_session).strip()
        if not session:
            raise ValueError("session IDs cannot be empty")
        if session not in seen:
            normalized_sessions.append(session)
            seen.add(session)

    split_ratios = dict(ratios or DEFAULT_RATIOS)
    if not split_ratios:
        raise ValueError("at least one split ratio is required")
    if any(
        not isinstance(name, str)
        or not name.strip()
        or not isfinite(value)
        or value <= 0
        for name, value in split_ratios.items()
    ):
        raise ValueError("split names must be non-empty and ratios must be positive")
    if len(normalized_sessions) < len(split_ratios):
        raise ValueError(
            f"need at least {len(split_ratios)} independent sessions for "
            f"{len(split_ratios)} non-empty splits"
        )

    total_ratio = sum(split_ratios.values())
    exact = {
        name: len(normalized_sessions) * ratio / total_ratio
        for name, ratio in split_ratios.items()
    }
    quotas = {name: max(1, floor(value)) for name, value in exact.items()}

    while sum(quotas.values()) > len(normalized_sessions):
        donors = [name for name, count in quotas.items() if count > 1]
        if not donors:
            raise ValueError("not enough sessions to keep every split non-empty")
        donor = min(donors, key=lambda name: (exact[name] - quotas[name], name))
        quotas[donor] -= 1
    while sum(quotas.values()) < len(normalized_sessions):
        receiver = max(
            quotas,
            key=lambda name: (exact[name] - quotas[name], split_ratios[name], name),
        )
        quotas[receiver] += 1

    ordered_sessions = sorted(
        normalized_sessions,
        key=lambda session: (
            hashlib.sha256(f"{seed}:{session}".encode("utf-8")).digest(),
            session,
        ),
    )
    # Stable ordering also makes the quota allocation inspectable in manifests.
    ordered_splits = sorted(split_ratios, key=lambda name: (-split_ratios[name], name))
    assignments: dict[str, str] = {}
    cursor = 0
    for split in ordered_splits:
        for session in ordered_sessions[cursor : cursor + quotas[split]]:
            assignments[session] = split
        cursor += quotas[split]
    return assignments
