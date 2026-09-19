"""The audit write path.

Three rules from brief §8, in order of how badly each one hurts when broken:

1. **The write must not sit in the request path.** ``submit`` puts records on
   an in-memory queue and returns; a background task batches and flushes them.
   A synchronous insert would add database latency to every masked request and
   couple gateway availability to database availability.

2. **Fail open.** If the database is unreachable, spill to the WAL and keep
   serving, loudly. Never raise into the caller.

3. **Batch.** One request produces many spans, and one insert should carry all
   of them.

The queue is bounded. When it is full the sink spills directly to the WAL
rather than growing without limit or blocking the caller -- backpressure onto
the request path would violate rule 1 to satisfy rule 2.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import Final

import structlog
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from pii_service.audit.records import AuditRecord
from pii_service.audit.wal import AuditWal
from pii_service.db.models import PiiEvent
from pii_service.settings import Settings

__all__ = ["AuditSink", "AuditStats"]

logger: Final = structlog.get_logger(__name__)


def _pool_options(database_url: str) -> dict[str, object]:
    """Connection-pool settings appropriate to the backend.

    SQLite gets none: SQLAlchemy selects ``StaticPool`` for an in-memory
    database and ``SingletonThreadPool`` for a file, and neither accepts
    ``pool_size`` -- passing them raises at engine construction rather than
    being ignored. Postgres is the deployment target; SQLite only ever appears
    in development and tests.
    """
    if database_url.startswith("sqlite"):
        return {}
    return {"pool_pre_ping": True, "pool_size": 5, "max_overflow": 5}


@dataclass
class AuditStats:
    """Counters worth exporting. `spilled` and `dropped` are the alarm signals."""

    submitted: int = 0
    written: int = 0
    spilled: int = 0
    dropped: int = 0
    replayed: int = 0
    flush_failures: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "submitted": self.submitted,
            "written": self.written,
            "spilled": self.spilled,
            "dropped": self.dropped,
            "replayed": self.replayed,
            "flush_failures": self.flush_failures,
        }


class AuditSink:
    """Queues audit records and flushes them to Postgres in batches."""

    def __init__(
        self,
        settings: Settings,
        *,
        engine: AsyncEngine | None = None,
        wal: AuditWal | None = None,
    ) -> None:
        self._settings: Final = settings
        self._engine: Final = engine or create_async_engine(
            settings.database_url,
            **_pool_options(settings.database_url),
        )
        self._session_factory: Final = async_sessionmaker(self._engine, expire_on_commit=False)
        self._wal: Final = wal or AuditWal(settings.audit_wal_path)
        self._queue: asyncio.Queue[AuditRecord] = asyncio.Queue(
            maxsize=settings.audit_queue_max
        )
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        self.stats: Final = AuditStats()

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        if self._task is not None:
            return
        if self._settings.audit_wal_enabled:
            self._wal.ensure_directory()
        self._stopping.clear()
        self._task = asyncio.create_task(self._run(), name="audit-sink-flush")

    async def stop(self, *, drain_timeout: float = 5.0) -> None:
        """Stop the flusher, making a best effort to drain what is queued.

        Anything still queued after the timeout goes to the WAL rather than
        being dropped: shutdown is exactly when records are most likely to be
        lost and least likely to be noticed.
        """
        if self._task is None:
            return

        self._stopping.set()
        with contextlib.suppress(TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(self._task, timeout=drain_timeout)
        if not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None

        remaining = self._drain_queue_nowait()
        if remaining:
            self._spill(remaining, reason="shutdown")

        await self._engine.dispose()

    # -- submission --------------------------------------------------------

    def submit(self, records: list[AuditRecord]) -> None:
        """Enqueue records. Never blocks, never raises into the request path."""
        if not records:
            return

        self.stats.submitted += len(records)
        overflow: list[AuditRecord] = []

        for record in records:
            try:
                self._queue.put_nowait(record)
            except asyncio.QueueFull:
                overflow.append(record)

        if overflow:
            # Spilling beats blocking: backpressure here would push database
            # latency back onto the gateway, which is the thing rule 1 forbids.
            logger.error(
                "audit.queue_full",
                overflow=len(overflow),
                queue_max=self._settings.audit_queue_max,
            )
            self._spill(overflow, reason="queue_full")

    # -- background flusher ------------------------------------------------

    async def _run(self) -> None:
        interval = self._settings.audit_flush_interval_seconds
        await self._replay_wal()

        while not self._stopping.is_set():
            batch = await self._collect_batch(interval)
            if batch:
                await self._flush(batch)

        # Final drain after the stop signal.
        while (batch := self._drain_queue_nowait(self._settings.audit_batch_size)):
            await self._flush(batch)

    async def _collect_batch(self, interval: float) -> list[AuditRecord]:
        """Wait for one record, then take everything else already queued."""
        batch: list[AuditRecord] = []
        try:
            first = await asyncio.wait_for(self._queue.get(), timeout=interval)
        except (TimeoutError, asyncio.CancelledError):
            return batch

        batch.append(first)
        batch.extend(self._drain_queue_nowait(self._settings.audit_batch_size - 1))
        return batch

    def _drain_queue_nowait(self, limit: int | None = None) -> list[AuditRecord]:
        drained: list[AuditRecord] = []
        while limit is None or len(drained) < limit:
            try:
                drained.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return drained

    async def _flush(self, batch: list[AuditRecord]) -> None:
        try:
            await self._insert(batch)
        except Exception as exc:  # noqa: BLE001 -- fail open, whatever went wrong
            self.stats.flush_failures += 1
            logger.error(
                "audit.flush_failed",
                error=type(exc).__name__,
                message=str(exc),
                batch_size=len(batch),
            )
            self._spill(batch, reason="flush_failed")
            return

        self.stats.written += len(batch)

    async def _insert(self, batch: list[AuditRecord]) -> None:
        rows = [record.to_row() for record in batch]
        async with self._session_factory() as session, session.begin():
            await session.execute(insert(PiiEvent), rows)

    # -- WAL ---------------------------------------------------------------

    def _spill(self, records: list[AuditRecord], *, reason: str) -> None:
        if not self._settings.audit_wal_enabled:
            self.stats.dropped += len(records)
            logger.error("audit.dropped", count=len(records), reason=reason)
            return

        try:
            self._wal.append(records)
            self.stats.spilled += len(records)
        except OSError as exc:
            self.stats.dropped += len(records)
            logger.error(
                "audit.spill_failed",
                count=len(records),
                reason=reason,
                error=type(exc).__name__,
                message=str(exc),
            )

    async def _replay_wal(self) -> None:
        """Replay spilled segments on startup, oldest first.

        A segment is deleted only after its batch is committed. Replaying a
        segment twice is survivable (duplicate rows an auditor can spot);
        deleting one whose insert then fails is not.
        """
        if not self._settings.audit_wal_enabled:
            return

        for segment, records in self._wal.drain():
            if not records:
                self._wal.discard(segment)
                continue
            try:
                await self._insert(records)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "audit.wal.replay_deferred",
                    segment=segment.name,
                    error=type(exc).__name__,
                    message=str(exc),
                )
                return  # Database still unhappy; try again next startup.

            self.stats.replayed += len(records)
            self._wal.discard(segment)
            logger.info("audit.wal.replayed", segment=segment.name, count=len(records))

    # -- health ------------------------------------------------------------

    async def database_reachable(self) -> bool:
        from sqlalchemy import text

        try:
            async with self._engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        except Exception:  # noqa: BLE001
            return False
        return True

    def health(self) -> dict[str, object]:
        return {
            **self.stats.as_dict(),
            "queue_depth": self._queue.qsize(),
            "wal_segments": len(self._wal.segments()),
            "wal_dropped_segments": self._wal.dropped_segments,
        }
