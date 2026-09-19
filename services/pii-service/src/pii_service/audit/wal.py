"""Write-ahead spill for audit records the database would not take.

Fail open (brief §8). If Postgres is unreachable, the guardrail keeps masking
and keeps serving, and the records it could not write land here as JSON lines
to be replayed later. A guardrail that fails requests because its audit log is
down is a self-inflicted outage -- it converts a logging problem into a
gateway outage for 100+ engineers.

Two properties matter and both are easy to get wrong:

*Bounded.* An unbounded spill file fills the disk, and a full disk takes the
service down anyway -- the outage arrives later and is harder to diagnose. So
segments roll at a size limit and the oldest are dropped, with a loud metric.
Losing the oldest audit records is bad; losing the gateway is worse, and
dropping silently is worst of all.

*Non-PII.* A spill file is a file on a disk that nobody is watching, so it must
be exactly as safe as the table. It holds serialized ``AuditRecord``s, which
have no value field, so it is.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import structlog

from pii_service.audit.records import AuditRecord

__all__ = ["AuditWal"]

logger: Final = structlog.get_logger(__name__)

_SEGMENT_SUFFIX: Final = ".wal"
_DEFAULT_MAX_SEGMENT_BYTES: Final = 16 * 1024 * 1024
_DEFAULT_MAX_SEGMENTS: Final = 8


class AuditWal:
    """An append-only, size-bounded JSON-lines spill directory."""

    def __init__(
        self,
        directory: Path,
        *,
        max_segment_bytes: int = _DEFAULT_MAX_SEGMENT_BYTES,
        max_segments: int = _DEFAULT_MAX_SEGMENTS,
    ) -> None:
        self._directory: Final = Path(directory)
        self._max_segment_bytes: Final = max_segment_bytes
        self._max_segments: Final = max_segments
        self._dropped_segments = 0

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def dropped_segments(self) -> int:
        """Segments discarded to stay within the size bound. Emit as a metric."""
        return self._dropped_segments

    def ensure_directory(self) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)

    def append(self, records: list[AuditRecord]) -> int:
        """Append records to the current segment. Returns how many were written."""
        if not records:
            return 0

        self.ensure_directory()
        segment = self._current_segment()
        payload = "".join(f"{record.to_json()}\n" for record in records)

        with segment.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        self._enforce_bound()
        return len(records)

    def segments(self) -> list[Path]:
        """Spill segments, oldest first."""
        if not self._directory.is_dir():
            return []
        return sorted(self._directory.glob(f"*{_SEGMENT_SUFFIX}"))

    def pending_count(self) -> int:
        total = 0
        for segment in self.segments():
            with segment.open("r", encoding="utf-8") as handle:
                total += sum(1 for line in handle if line.strip())
        return total

    def drain(self) -> Iterator[tuple[Path, list[AuditRecord]]]:
        """Yield ``(segment, records)`` oldest first, for replay.

        The segment is *not* deleted -- the caller deletes it with
        ``discard`` once the database has accepted the batch. Deleting first
        would lose exactly the records a flaky database is most likely to
        reject.
        """
        for segment in self.segments():
            records: list[AuditRecord] = []
            with segment.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        records.append(AuditRecord.from_json(line))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        # One torn line -- almost certainly a partial write at
                        # the tail after a kill -- must not strand the whole
                        # segment and every record behind it.
                        logger.warning(
                            "audit.wal.corrupt_line",
                            segment=segment.name,
                            line_number=line_number,
                        )
            yield segment, records

    def discard(self, segment: Path) -> None:
        """Delete a replayed segment."""
        segment.unlink(missing_ok=True)

    # -- internals ---------------------------------------------------------

    def _current_segment(self) -> Path:
        existing = self.segments()
        if existing:
            newest = existing[-1]
            if newest.stat().st_size < self._max_segment_bytes:
                return newest
        return self._new_segment()

    def _new_segment(self) -> Path:
        # A monotonic counter in the name keeps `sorted()` chronological
        # without relying on mtime, which replay and backup tools rewrite.
        index = len(self.segments())
        while True:
            candidate = self._directory / f"audit-{index:08d}{_SEGMENT_SUFFIX}"
            if not candidate.exists():
                return candidate
            index += 1

    def _enforce_bound(self) -> None:
        segments = self.segments()
        while len(segments) > self._max_segments:
            oldest = segments.pop(0)
            oldest.unlink(missing_ok=True)
            self._dropped_segments += 1
            logger.error(
                "audit.wal.segment_dropped",
                segment=oldest.name,
                dropped_total=self._dropped_segments,
                reason="wal size bound exceeded; oldest audit records discarded",
            )

    def write_atomic_marker(self, name: str, payload: dict[str, object]) -> None:
        """Write a small status file (used by health checks) without tearing it."""
        self.ensure_directory()
        target = self._directory / name
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self._directory, delete=False
        ) as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(target)
