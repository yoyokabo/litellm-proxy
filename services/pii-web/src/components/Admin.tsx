/**
 * Admin view: timeline on top, event table below, drill-down on the right.
 *
 * Timeline and drill-down only. No leaderboard -- brief §10 excludes it, and
 * "who leaked the most PII" is a scoreboard that changes behaviour without
 * improving it.
 */
import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import type { EventDetail, EventPage, TimelineResponse } from "../types";
import { Card, Empty, Button, Input, Stat } from "./ui/primitives";
import { Timeline } from "./Timeline";
import { EventTable } from "./EventTable";
import { DrillDown } from "./DrillDown";

const PAGE = 500;

export function Admin() {
  const [timeline, setTimeline] = useState<TimelineResponse | null>(null);
  const [page, setPage] = useState<EventPage | null>(null);
  const [selected, setSelected] = useState<EventDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // Filters. `pivot` is the fingerprint the view is currently focused on --
  // the one-click investigative move -- and is shown as a dismissible banner
  // rather than buried in a filter dropdown, because it changes what the whole
  // screen means.
  const [range, setRange] = useState<{ since?: string; until?: string }>({});
  const [search, setSearch] = useState("");
  const [pivot, setPivot] = useState<string | null>(null);
  const [bucket, setBucket] = useState<"minute" | "hour" | "day">("hour");

  const params = useCallback(() => {
    const query = new URLSearchParams();
    if (range.since) query.set("since", range.since);
    if (range.until) query.set("until", range.until);
    if (search.trim()) query.set("search", search.trim());
    if (pivot) query.set("value_fp", pivot);
    return query;
  }, [range, search, pivot]);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const timelineQuery = params();
      timelineQuery.set("bucket", bucket);
      const eventQuery = params();
      eventQuery.set("limit", String(PAGE));

      const [timelineData, eventData] = await Promise.all([
        api.timeline(timelineQuery),
        api.events(eventQuery),
      ]);
      setTimeline(timelineData);
      setPage(eventData);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "failed to load");
    } finally {
      setLoading(false);
    }
  }, [params, bucket]);

  useEffect(() => {
    void load();
  }, [load]);

  function pivotTo(fp: string) {
    setPivot(fp);
    // Clear the brush: an occurrence outside the current window is exactly
    // what the pivot exists to surface, and keeping the range would hide it.
    setRange({});
    setSelected(null);
  }

  return (
    <div className="flex h-full">
      <div className="flex flex-1 flex-col overflow-hidden">
        <div className="border-b border-border px-6 py-4">
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex-1 min-w-[16rem]">
              <Input
                value={search}
                onChange={setSearch}
                placeholder="Filter by user, key alias or request id…"
              />
            </div>
            <div className="flex gap-1">
              {(["minute", "hour", "day"] as const).map((option) => (
                <button
                  key={option}
                  onClick={() => setBucket(option)}
                  className={`rounded-md px-3 py-2 text-[12px] transition-colors ${
                    bucket === option
                      ? "bg-accent text-bg-0"
                      : "bg-bg-2 text-text-2 hover:text-text-1"
                  }`}
                >
                  {option}
                </button>
              ))}
            </div>
            {(range.since || pivot || search) && (
              <Button
                variant="ghost"
                onClick={() => {
                  setRange({});
                  setPivot(null);
                  setSearch("");
                }}
              >
                Reset
              </Button>
            )}
          </div>

          {pivot && (
            <div className="mt-3 flex items-center justify-between rounded-md border border-accent-dim bg-accent/10 px-3 py-2">
              <span className="text-[13px]">
                Showing every occurrence of fingerprint{" "}
                <span className="font-mono text-accent">{pivot}</span>
                {timeline && (
                  <span className="text-text-2">
                    {" "}
                    — {timeline.summary.events} events from {timeline.summary.users}{" "}
                    {timeline.summary.users === 1 ? "person" : "people"}
                  </span>
                )}
              </span>
              <button
                onClick={() => setPivot(null)}
                className="text-[12px] text-text-3 hover:text-text-1"
              >
                Clear
              </button>
            </div>
          )}
        </div>

        <div className="overflow-y-auto px-6 py-4">
          {error && (
            <Card className="mb-4 border-blocked px-4 py-3 text-[13px] text-blocked">{error}</Card>
          )}

          {timeline && (
            <>
              <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-5">
                <Stat label="Events" value={timeline.summary.events.toLocaleString()} />
                <Stat label="People" value={timeline.summary.users.toLocaleString()} />
                <Stat
                  label="Distinct values"
                  value={timeline.summary.distinct_values.toLocaleString()}
                />
                <Stat
                  label="Masked"
                  value={timeline.summary.masked.toLocaleString()}
                  tone="var(--masked)"
                />
                <Stat
                  label="Blocked"
                  value={timeline.summary.blocked.toLocaleString()}
                  tone="var(--blocked)"
                />
              </div>

              <Card className="p-4">
                <Timeline
                  data={timeline}
                  onBrush={(since, until) => setRange({ since, until })}
                />
              </Card>
            </>
          )}

          {loading && !timeline && <Empty>Loading…</Empty>}
        </div>

        <div className="min-h-0 flex-1 border-t border-border">
          {page && (
            <EventTable
              events={page.events}
              total={page.total}
              selectedId={selected?.id ?? null}
              onSelect={setSelected}
              onPivot={pivotTo}
            />
          )}
        </div>
      </div>

      {selected && (
        <DrillDown
          event={selected}
          onClose={() => setSelected(null)}
          onPivot={pivotTo}
          onFlagged={() => void load()}
        />
      )}
    </div>
  );
}
