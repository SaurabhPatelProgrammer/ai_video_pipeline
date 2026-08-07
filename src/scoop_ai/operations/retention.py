"""Plan-first evidence retention with an explicit dry-run/delete boundary."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path


DEFAULT_EVIDENCE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".mp4", ".webm"})


@dataclass(frozen=True)
class RetentionPolicy:
    max_age: timedelta | None = None
    max_total_bytes: int | None = None
    allowed_suffixes: frozenset[str] = DEFAULT_EVIDENCE_SUFFIXES
    protected_relative_paths: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.max_age is None and self.max_total_bytes is None:
            raise ValueError("at least one retention limit is required")
        if self.max_age is not None and self.max_age.total_seconds() < 0:
            raise ValueError("max_age cannot be negative")
        if self.max_total_bytes is not None and self.max_total_bytes < 0:
            raise ValueError("max_total_bytes cannot be negative")
        object.__setattr__(
            self,
            "allowed_suffixes",
            frozenset(suffix.lower() for suffix in self.allowed_suffixes),
        )


@dataclass(frozen=True)
class RetentionCandidate:
    relative_path: str
    size_bytes: int
    modified_at: str
    modified_ns: int
    reason: str


@dataclass(frozen=True)
class RetentionPlan:
    root: str
    generated_at: str
    candidates: tuple[RetentionCandidate, ...]
    total_scanned_bytes: int
    reclaimable_bytes: int


@dataclass(frozen=True)
class RetentionResult:
    dry_run: bool
    planned_files: int
    deleted_files: int
    reclaimed_bytes: int
    skipped: tuple[str, ...]
    errors: tuple[str, ...]


class RetentionManager:
    """Delete only files that were captured in a plan from the configured root."""

    def __init__(self, root: Path | str, policy: RetentionPolicy) -> None:
        self.root = Path(root).resolve()
        self.policy = policy

    def _is_protected(self, relative: Path) -> bool:
        normalized = Path(os.path.normcase(relative.as_posix()))
        for configured in self.policy.protected_relative_paths:
            protected = Path(configured)
            if protected.is_absolute() or ".." in protected.parts:
                raise ValueError("protected paths must stay under the retention root")
            protected = Path(os.path.normcase(protected.as_posix()))
            if normalized == protected or normalized.is_relative_to(protected):
                return True
        return False

    def _eligible_files(self) -> list[tuple[Path, os.stat_result]]:
        if not self.root.exists():
            return []
        files: list[tuple[Path, os.stat_result]] = []
        for path in self.root.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(self.root):
                continue
            relative = resolved.relative_to(self.root).as_posix()
            if (
                self._is_protected(Path(relative))
                or path.suffix.lower() not in self.policy.allowed_suffixes
            ):
                continue
            files.append((resolved, resolved.stat()))
        return files

    def plan(self, *, now: datetime | None = None) -> RetentionPlan:
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError("now must include a timezone")
        files = self._eligible_files()
        total_bytes = sum(stat.st_size for _, stat in files)
        reasons: dict[Path, set[str]] = {}

        if self.policy.max_age is not None:
            cutoff = now.timestamp() - self.policy.max_age.total_seconds()
            for path, stat in files:
                if stat.st_mtime <= cutoff:
                    reasons.setdefault(path, set()).add("age")

        if self.policy.max_total_bytes is not None:
            remaining = total_bytes
            for path, stat in sorted(files, key=lambda pair: (pair[1].st_mtime_ns, str(pair[0]))):
                if remaining <= self.policy.max_total_bytes:
                    break
                reasons.setdefault(path, set()).add("quota")
                remaining -= stat.st_size

        candidates: list[RetentionCandidate] = []
        stat_by_path = dict(files)
        for path in sorted(reasons, key=lambda item: (stat_by_path[item].st_mtime_ns, str(item))):
            stat = stat_by_path[path]
            candidates.append(
                RetentionCandidate(
                    relative_path=path.relative_to(self.root).as_posix(),
                    size_bytes=stat.st_size,
                    modified_at=datetime.fromtimestamp(
                        stat.st_mtime, timezone.utc
                    ).isoformat(timespec="milliseconds"),
                    modified_ns=stat.st_mtime_ns,
                    reason="+".join(sorted(reasons[path])),
                )
            )
        return RetentionPlan(
            root=str(self.root),
            generated_at=now.astimezone(timezone.utc).isoformat(timespec="milliseconds"),
            candidates=tuple(candidates),
            total_scanned_bytes=total_bytes,
            reclaimable_bytes=sum(item.size_bytes for item in candidates),
        )

    def execute(self, plan: RetentionPlan, *, dry_run: bool = True) -> RetentionResult:
        if Path(plan.root).resolve() != self.root:
            raise ValueError("retention plan belongs to another root")
        if dry_run:
            return RetentionResult(
                dry_run=True,
                planned_files=len(plan.candidates),
                deleted_files=0,
                reclaimed_bytes=0,
                skipped=(),
                errors=(),
            )

        deleted = 0
        reclaimed = 0
        skipped: list[str] = []
        errors: list[str] = []
        for candidate in plan.candidates:
            relative = Path(candidate.relative_path)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.suffix.lower() not in self.policy.allowed_suffixes
                or self._is_protected(relative)
            ):
                skipped.append(candidate.relative_path)
                continue
            lexical_path = self.root / relative
            current = self.root
            has_symlink = False
            for part in relative.parts:
                current = current / part
                if current.is_symlink():
                    has_symlink = True
                    break
            path = lexical_path.resolve()
            if has_symlink or not path.is_relative_to(self.root):
                skipped.append(candidate.relative_path)
                continue
            try:
                stat = path.stat()
            except FileNotFoundError:
                skipped.append(candidate.relative_path)
                continue
            if stat.st_size != candidate.size_bytes or stat.st_mtime_ns != candidate.modified_ns:
                skipped.append(candidate.relative_path)
                continue
            try:
                path.unlink()
            except OSError as exc:
                errors.append(f"{candidate.relative_path}: {exc.__class__.__name__}")
                continue
            deleted += 1
            reclaimed += candidate.size_bytes
        return RetentionResult(
            dry_run=False,
            planned_files=len(plan.candidates),
            deleted_files=deleted,
            reclaimed_bytes=reclaimed,
            skipped=tuple(skipped),
            errors=tuple(errors),
        )
