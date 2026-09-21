/**
 * The right-side drill-down panel (brief §10).
 *
 * A panel, not a route change, so the timeline stays on screen as context
 * while you read one span.
 *
 * The fingerprint block is the core investigative move: one click pivots the
 * whole view to every other occurrence of that value. What it can show is
 * counts, people and times; what it cannot show -- because the pepper lives
 * outside this database -- is the value itself. That is the property the
 * whole audit design exists to have.
 */
import { useEffect, useState } from "react";
import { api } from "../api";
import type { EventDetail, FingerprintSummary } from "../types";
import { formatDateTime } from "../lib/format";
import { ActionBadge, Button, CategoryDot } from "./ui/primitives";

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-border/60 py-2">
      <span className="shrink-0 text-[12px] text-text-3">{label}</span>
      <span className="text-right text-[13px]">{children}</span>
    </div>
  );
}

const Mono = ({ children }: { children: React.ReactNode }) => (
  <span className="font-mono text-[12px]">{children}</span>
);

export function DrillDown({
  event,
  onClose,
  onPivot,
  onFlagged,
}: {
  event: EventDetail;
  onClose: () => void;
  onPivot: (fp: string) => void;
  onFlagged: () => void;
}) {
  const [summary, setSummary] = useState<FingerprintSummary | null>(null);
  const [flag, setFlag] = useState<string | null>(event.flag);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    setFlag(event.flag);
    setSummary(null);
    if (event.value_fp) {
      api.fingerprint(event.value_fp).then(setSummary).catch(() => setSummary(null));
    }
  }, [event.id, event.value_fp, event.flag]);

  async function record(verdict: "false_positive" | "confirmed") {
    setBusy(true);
    try {
      await api.flag(event.id, verdict);
      setFlag(verdict);
      onFlagged();
    } finally {
      setBusy(false);
    }
  }

  return (
    <aside className="flex h-full w-[24rem] shrink-0 flex-col border-l border-border bg-bg-1">
      <header className="flex items-center justify-between border-b border-border px-4 py-3">
        <div className="flex items-center gap-2">
          <CategoryDot category={event.category} />
          <span className="text-[13px] font-medium">{event.entity_type}</span>
          <ActionBadge action={event.action} />
        </div>
        <button onClick={onClose} className="text-text-3 hover:text-text-1" title="Close">
          ✕
        </button>
      </header>

      <div className="flex-1 overflow-y-auto px-4 pb-6">
        <section className="pt-2">
          <h3 className="py-2 text-[11px] uppercase tracking-wide text-text-3">Detection</h3>
          <Field label="Recognizer">
            <Mono>{event.recognizer}</Mono>
          </Field>
          <Field label="Score">
            <Mono>{event.score?.toFixed(2) ?? "—"}</Mono>
          </Field>
          <Field label="Language">
            <Mono>{event.lang ?? "—"}</Mono>
          </Field>
          <Field label="Detection time">
            <Mono>{event.latency_ms ?? "—"} ms</Mono>
          </Field>
        </section>

        <section>
          <h3 className="py-2 text-[11px] uppercase tracking-wide text-text-3">Where</h3>
          {/* Offsets are shown as numbers, never by pointing at rendered text.
              A span at 10-24 does not appear "after" one at 0-9 in Arabic
              (brief §10), so position is reported numerically and the preview
              below is bidi-isolated. */}
          <Field label="Offsets">
            <Mono>
              {event.span_start ?? "—"} – {event.span_end ?? "—"}
            </Mono>
          </Field>
          <Field label="Length">
            <Mono>{event.value_len ?? "—"} chars</Mono>
          </Field>
          <Field label="Message">
            <Mono>
              {event.message_index ?? "—"}
              {event.message_role ? ` · ${event.message_role}` : ""}
            </Mono>
          </Field>
          <Field label="Field">
            <Mono>{event.field ?? "—"}</Mono>
          </Field>
          <Field label="Preview">
            {event.preview ? (
              <span dir="auto" className="bidi-auto font-mono text-[12px]">
                {event.preview}
              </span>
            ) : (
              <span className="text-[12px] text-text-3">withheld by policy</span>
            )}
          </Field>
        </section>

        <section>
          <h3 className="py-2 text-[11px] uppercase tracking-wide text-text-3">Who</h3>
          <Field label="User">
            <Mono>{event.user_id ?? "—"}</Mono>
          </Field>
          <Field label="Key alias">
            <Mono>{event.key_alias ?? "—"}</Mono>
          </Field>
          <Field label="End user">
            <Mono>{event.end_user_id ?? "—"}</Mono>
          </Field>
          <Field label="Model">
            <Mono>{event.model ?? "—"}</Mono>
          </Field>
          <Field label="Request">
            <Mono>{event.request_id.slice(0, 18)}…</Mono>
          </Field>
          <Field label="Time">
            <span className="text-[12px]">{formatDateTime(event.ts)}</span>
          </Field>
        </section>

        {event.value_fp && (
          <section className="mt-4 rounded-md border border-border bg-bg-0 p-3">
            <h3 className="text-[11px] uppercase tracking-wide text-text-3">Fingerprint</h3>
            <div className="mt-1 font-mono text-[13px] text-accent">{event.value_fp}</div>

            {summary ? (
              <>
                <div className="mt-3 grid grid-cols-3 gap-2 text-center">
                  <div>
                    <div className="font-mono text-lg">{summary.occurrences}</div>
                    <div className="text-[11px] text-text-3">times</div>
                  </div>
                  <div>
                    <div className="font-mono text-lg">{summary.distinct_users}</div>
                    <div className="text-[11px] text-text-3">people</div>
                  </div>
                  <div>
                    <div className="font-mono text-lg">{summary.distinct_requests}</div>
                    <div className="text-[11px] text-text-3">requests</div>
                  </div>
                </div>

                {/* The sentence the pepper buys. Same event count, opposite
                    meaning, and neither requires storing a digit. */}
                <p className="mt-3 text-[12px] leading-relaxed text-text-2">
                  {summary.distinct_users <= 1
                    ? `This value came from one person, ${summary.occurrences} time${
                        summary.occurrences === 1 ? "" : "s"
                      }.`
                    : `This value came from ${summary.distinct_users} different people across ${summary.distinct_requests} requests.`}
                </p>

                <div className="mt-3">
                  <Button variant="ghost" onClick={() => onPivot(event.value_fp!)}>
                    Show every occurrence
                  </Button>
                </div>
              </>
            ) : (
              <div className="mt-2 text-[12px] text-text-3">Loading correlation…</div>
            )}
          </section>
        )}

        <section className="mt-4">
          <h3 className="py-2 text-[11px] uppercase tracking-wide text-text-3">Review</h3>
          <p className="mb-2 text-[12px] leading-relaxed text-text-3">
            Verdicts feed the Arabic evaluation set. Confirmations matter as much as
            corrections — a set built only from mistakes teaches nothing about what works.
          </p>
          <div className="flex gap-2">
            <Button
              variant={flag === "false_positive" ? "primary" : "ghost"}
              onClick={() => void record("false_positive")}
              disabled={busy}
            >
              False positive
            </Button>
            <Button
              variant={flag === "confirmed" ? "primary" : "ghost"}
              onClick={() => void record("confirmed")}
              disabled={busy}
            >
              Correct
            </Button>
          </div>
        </section>
      </div>
    </aside>
  );
}
