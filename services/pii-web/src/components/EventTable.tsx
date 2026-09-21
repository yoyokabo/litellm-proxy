/**
 * The event table under the timeline.
 *
 * Virtualized by hand: only the rows in view are mounted, plus a small
 * overscan. A guardrail on a busy gateway produces hundreds of thousands of
 * spans, and mounting that many <tr> elements locks the tab.
 */
import { useMemo, useState } from "react";
import type { EventDetail } from "../types";
import { formatDateTime } from "../lib/format";
import { ActionBadge, CategoryDot, Empty } from "./ui/primitives";

const ROW_HEIGHT = 36;
const OVERSCAN = 8;

export function EventTable({
  events,
  total,
  selectedId,
  onSelect,
  onPivot,
}: {
  events: EventDetail[];
  total: number;
  selectedId: number | null;
  onSelect: (event: EventDetail) => void;
  onPivot: (fp: string) => void;
}) {
  const [scrollTop, setScrollTop] = useState(0);
  const [height, setHeight] = useState(480);

  const { start, end } = useMemo(() => {
    const first = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN);
    const visible = Math.ceil(height / ROW_HEIGHT) + OVERSCAN * 2;
    return { start: first, end: Math.min(events.length, first + visible) };
  }, [scrollTop, height, events.length]);

  if (events.length === 0) {
    return <Empty>No events match these filters.</Empty>;
  }

  const slice = events.slice(start, end);

  return (
    <div className="flex h-full flex-col">
      <div className="grid grid-cols-[150px_110px_170px_1fr_120px_90px] gap-3 border-b border-border px-4 py-2 text-[11px] uppercase tracking-wide text-text-3">
        <span>Time</span>
        <span>Action</span>
        <span>Entity</span>
        <span>User</span>
        <span>Fingerprint</span>
        <span>Preview</span>
      </div>

      <div
        ref={(node) => {
          // Measured from the mounted node rather than assumed: the viewport
          // height decides how many rows exist at all, and guessing it wrong
          // either mounts too many or leaves a gap at the bottom.
          if (node && node.clientHeight && node.clientHeight !== height) {
            setHeight(node.clientHeight);
          }
        }}
        onScroll={(event) => setScrollTop(event.currentTarget.scrollTop)}
        className="flex-1 overflow-y-auto"
      >
        <div style={{ height: events.length * ROW_HEIGHT, position: "relative" }}>
          <div style={{ transform: `translateY(${start * ROW_HEIGHT}px)` }}>
            {slice.map((event) => (
              <button
                key={event.id}
                onClick={() => onSelect(event)}
                style={{ height: ROW_HEIGHT }}
                className={`grid w-full grid-cols-[150px_110px_170px_1fr_120px_90px] items-center gap-3 border-b border-border/40 px-4 text-left text-[12px] transition-colors hover:bg-bg-2 ${
                  selectedId === event.id ? "bg-bg-2" : ""
                }`}
              >
                <span className="font-mono text-text-2">{formatDateTime(event.ts)}</span>
                <span>
                  <ActionBadge action={event.action} />
                </span>
                <span className="flex items-center gap-2 truncate">
                  <CategoryDot category={event.category} />
                  <span className="truncate">{event.entity_type}</span>
                  {event.flag === "false_positive" && (
                    <span title="Flagged as a false positive" className="text-blocked">
                      ⚑
                    </span>
                  )}
                </span>
                <span className="truncate font-mono text-text-2">{event.user_id ?? "—"}</span>
                <span
                  onClick={(clicked) => {
                    if (!event.value_fp) return;
                    clicked.stopPropagation();
                    onPivot(event.value_fp);
                  }}
                  title="Pivot to every occurrence of this value"
                  className="cursor-pointer truncate font-mono text-accent hover:text-accent-hover hover:underline"
                >
                  {event.value_fp ?? "—"}
                </span>
                <span dir="auto" className="bidi-auto truncate font-mono text-text-3">
                  {event.preview ?? "—"}
                </span>
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="border-t border-border px-4 py-2 text-[11px] text-text-3">
        Showing {events.length.toLocaleString()} of {total.toLocaleString()} events
      </div>
    </div>
  );
}
