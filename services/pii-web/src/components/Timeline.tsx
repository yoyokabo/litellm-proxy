/**
 * The stacked timeline (brief §10).
 *
 * Two decisions the brief is specific about:
 *
 * 1. **Stacked by entity category, not by entity type.** Fifteen types is a
 *    legend nobody reads; six categories is a shape you can see.
 * 2. **Blocked events get their own thin lane** below the stack. Folded into
 *    the stack, a handful of blocks beside thousands of masks is a band one
 *    pixel high -- invisible exactly when it matters most.
 *
 * Brushing the chart filters the table below it.
 */
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  Brush,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { TimelineResponse } from "../types";
import { CATEGORY_COLOR, CATEGORY_LABEL, CATEGORY_ORDER, formatTime } from "../lib/format";

interface Row {
  bucket: string;
  label: string;
  blocked: number;
  [category: string]: string | number;
}

function toRows(data: TimelineResponse): Row[] {
  const byBucket = new Map<string, Row>();

  for (const point of data.points) {
    let row = byBucket.get(point.bucket);
    if (!row) {
      row = { bucket: point.bucket, label: formatTime(point.bucket, data.bucket), blocked: 0 };
      for (const category of CATEGORY_ORDER) row[category] = 0;
      byBucket.set(point.bucket, row);
    }
    row[point.category] = ((row[point.category] as number) ?? 0) + point.total;
    row.blocked += point.blocked;
  }

  return [...byBucket.values()].sort((a, b) => a.bucket.localeCompare(b.bucket));
}

function TooltipBox({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const rows = payload.filter((entry: any) => entry.value > 0);
  if (!rows.length) return null;

  return (
    <div className="rounded-md border border-border bg-bg-1 px-3 py-2 text-[12px] shadow-lg">
      <div className="mb-1 font-mono text-text-3">{label}</div>
      {rows.map((entry: any) => (
        <div key={entry.dataKey} className="flex items-center justify-between gap-4">
          <span className="flex items-center gap-2">
            <span
              className="inline-block h-2 w-2 rounded-full"
              style={{ background: entry.color }}
            />
            {CATEGORY_LABEL[entry.dataKey as keyof typeof CATEGORY_LABEL] ?? entry.dataKey}
          </span>
          <span className="font-mono">{entry.value}</span>
        </div>
      ))}
    </div>
  );
}

export function Timeline({
  data,
  onBrush,
}: {
  data: TimelineResponse;
  onBrush: (since: string, until: string) => void;
}) {
  const rows = toRows(data);
  const anyBlocked = rows.some((row) => row.blocked > 0);

  if (rows.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center text-[13px] text-text-3">
        No events in this window.
      </div>
    );
  }

  function handleBrush(range: { startIndex?: number; endIndex?: number }) {
    const start = rows[range.startIndex ?? 0];
    const end = rows[range.endIndex ?? rows.length - 1];
    if (start && end) onBrush(start.bucket, end.bucket);
  }

  return (
    <div className="select-none">
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={rows} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="2 4" stroke="var(--border)" vertical={false} />
          <XAxis
            dataKey="label"
            tick={{ fill: "var(--text-3)", fontSize: 11 }}
            stroke="var(--border)"
          />
          <YAxis tick={{ fill: "var(--text-3)", fontSize: 11 }} stroke="var(--border)" />
          <Tooltip content={<TooltipBox />} cursor={{ stroke: "var(--text-3)" }} />
          {CATEGORY_ORDER.map((category) => (
            <Area
              key={category}
              type="monotone"
              dataKey={category}
              stackId="events"
              stroke={CATEGORY_COLOR[category]}
              fill={CATEGORY_COLOR[category]}
              fillOpacity={0.35}
              strokeWidth={1.5}
              isAnimationActive={false}
            />
          ))}
          <Brush
            dataKey="label"
            height={22}
            travellerWidth={8}
            stroke="var(--accent-dim)"
            fill="var(--bg-2)"
            onChange={handleBrush}
          />
        </AreaChart>
      </ResponsiveContainer>

      {/* The separate blocked lane. Always rendered, even at zero, so its
          absence reads as "none" rather than as a missing feature. */}
      <div className="mt-2">
        <div className="mb-1 flex items-center gap-2 text-[11px] uppercase tracking-wide text-text-3">
          <span className="inline-block h-2 w-2 rounded-sm border border-blocked" />
          Blocked {anyBlocked ? "" : "— none in this window"}
        </div>
        <ResponsiveContainer width="100%" height={48}>
          <BarChart data={rows} margin={{ top: 0, right: 8, left: -20, bottom: 0 }}>
            <XAxis dataKey="label" hide />
            <YAxis hide />
            <Tooltip content={<TooltipBox />} cursor={{ fill: "var(--bg-2)" }} />
            <Bar dataKey="blocked" fill="var(--blocked)" isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-3 flex flex-wrap gap-4">
        {CATEGORY_ORDER.map((category) => (
          <span key={category} className="flex items-center gap-2 text-[12px] text-text-2">
            <span
              className="inline-block h-2 w-2 rounded-full"
              style={{ background: CATEGORY_COLOR[category] }}
            />
            {CATEGORY_LABEL[category]}
          </span>
        ))}
      </div>
    </div>
  );
}
