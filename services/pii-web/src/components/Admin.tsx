/**
 * Admin view: the event log.
 *
 * Brief §10 specifies a stacked timeline above this table with brush-to-zoom.
 * That was built and then removed on request -- the log is the thing people
 * actually read, and a chart above it mostly competed for vertical space with
 * the rows. The backend still serves /api/admin/timeline, so putting it back
 * is a component, not a migration.
 *
 * What survives from that design is the part that was never about the chart:
 * drill-down as a side panel rather than a route change, so the list stays as
 * context, and the one-click fingerprint pivot.
 */
import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import type { EventDetail, EventPage, SummaryStats } from "../types";
import { Button, Card, Empty, Input, Stat } from "./ui/primitives";
import { EventTable } from "./EventTable";
import { DrillDown } from "./DrillDown";

const PAGE = 500;

export function Admin() {
  const [summary, setSummary] = useState<SummaryStats | null>(null);
  const [page, setPage] = useState<EventPage | null>(null);
  const [selected, setSelected] = useState<EventDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const [search, setSearch] = useState("");
  // The fingerprint the view is focused on -- the one-click investigative
  // move. Shown as a dismissible banner rather than hidden in a filter,
  // because it changes what the whole screen means.
  const [pivot, setPivot] = useState<string | null>(null);

  const params = useCallback(() => {
    const query = new URLSearchParams();
    if (search.trim()) query.set("search", search.trim());
    if (pivot) query.set("value_fp", pivot);
    return query;
  }, [search, pivot]);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const eventQuery = params();
      eventQuery.set("limit", String(PAGE));

      const [summaryData, eventData] = await Promise.all([
        api.summary(params()),
        api.events(eventQuery),
      ]);
      setSummary(summaryData);
      setPage(eventData);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "failed to load");
    } finally {
      setLoading(false);
    }
  }, [params]);

  useEffect(() => {
    void load();
  }, [load]);

  function pivotTo(fp: string) {
    setPivot(fp);
    setSelected(null);
  }

  return (
    <div className="flex h-full">
      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <div className="shrink-0 border-b border-border px-6 py-4">
          <div className="flex flex-wrap items-center gap-3">
            <div className="min-w-[16rem] flex-1">
              <Input
                value={search}
                onChange={setSearch}
                placeholder="Filter by user, key alias or request id…"
              />
            </div>
            <Button variant="ghost" onClick={() => void load()}>
              Refresh
            </Button>
            {(pivot || search) && (
              <Button
                variant="ghost"
                onClick={() => {
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
                {summary && (
                  <span className="text-text-2">
                    {" "}
                    — {summary.events} events from {summary.users}{" "}
                    {summary.users === 1 ? "person" : "people"}
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

          {error && (
            <Card className="mt-3 border-blocked px-4 py-3 text-[13px] text-blocked">{error}</Card>
          )}

          {summary && (
            <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-5">
              <Stat label="Events" value={summary.events.toLocaleString()} />
              <Stat label="People" value={summary.users.toLocaleString()} />
              <Stat label="Distinct values" value={summary.distinct_values.toLocaleString()} />
              <Stat label="Masked" value={summary.masked.toLocaleString()} tone="var(--masked)" />
              <Stat
                label="Blocked"
                value={summary.blocked.toLocaleString()}
                tone="var(--blocked)"
              />
            </div>
          )}
        </div>

        <div className="min-h-0 flex-1">
          {page ? (
            <EventTable
              events={page.events}
              total={page.total}
              selectedId={selected?.id ?? null}
              onSelect={setSelected}
              onPivot={pivotTo}
            />
          ) : (
            <Empty>{loading ? "Loading…" : "No events."}</Empty>
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
