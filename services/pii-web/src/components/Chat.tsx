/**
 * Chat with filtering feedback (brief §3).
 *
 * The rule that shapes this screen: showing people the PII *they themselves
 * just typed* is not a disclosure -- they wrote it. So their own bubble keeps
 * the original text with the detected spans underlined in category colours,
 * and a chip says exactly what was replaced before the message was forwarded.
 *
 * The masked path sends. The blocked path does not: the text stays in the
 * composer with its spans highlighted, plus a button to strip the flagged
 * runs, because a message that vanishes into an error is a message the person
 * has to retype from memory.
 */
import { useEffect, useRef, useState } from "react";
import { streamChat, api } from "../api";
import type { ChatAnalysis, ChatSpan, ChatTurn } from "../types";
import { colorFor, hasArabic } from "../lib/format";
import { Button } from "./ui/primitives";

/** Render text with detected spans underlined in their category colour.
 *
 *  Arabic needs `dir="auto"` and bidi isolation or a mixed-script line
 *  reorders around the spans (brief §10). The offsets are character indices
 *  into the original string and are never used for visual ordering. */
function SpannedText({ text, spans }: { text: string; spans: ChatSpan[] }) {
  const pieces: JSX.Element[] = [];
  let cursor = 0;

  for (const [index, span] of spans.entries()) {
    if (span.start > cursor) {
      pieces.push(<span key={`t${index}`}>{text.slice(cursor, span.start)}</span>);
    }
    pieces.push(
      <span
        key={`s${index}`}
        title={`${span.entity_type} → ${span.placeholder}`}
        className="rounded-sm px-0.5"
        style={{
          borderBottom: `2px solid ${colorFor(span.category)}`,
          background: `color-mix(in srgb, ${colorFor(span.category)} 14%, transparent)`,
        }}
      >
        {text.slice(span.start, span.end)}
      </span>,
    );
    cursor = span.end;
  }
  if (cursor < text.length) pieces.push(<span key="tail">{text.slice(cursor)}</span>);

  return (
    <span dir="auto" className={`bidi-auto ${hasArabic(text) ? "font-arabic" : ""}`}>
      {pieces}
    </span>
  );
}

/** The expandable "N items redacted" chip. */
function RedactionChip({ analysis }: { analysis: ChatAnalysis }) {
  const [open, setOpen] = useState(false);
  const count = analysis.spans.length;
  if (count === 0) return null;

  const blocked = analysis.blocked;
  return (
    <div className="mt-2">
      <button
        onClick={() => setOpen(!open)}
        className="inline-flex items-center gap-2 rounded-full border px-3 py-1 text-[12px] transition-colors"
        style={{
          borderColor: blocked ? "var(--blocked)" : "var(--masked)",
          color: blocked ? "var(--blocked)" : "var(--masked)",
        }}
      >
        <span>
          {count} item{count === 1 ? "" : "s"} {blocked ? "blocked" : "redacted"}
        </span>
        <span className="text-text-3">{open ? "▾" : "▸"}</span>
      </button>

      {open && (
        <div className="mt-2 overflow-hidden rounded-md border border-border bg-bg-0">
          <table className="w-full text-[12px]">
            <thead>
              <tr className="border-b border-border text-left text-text-3">
                <th className="px-3 py-1.5 font-normal">Type</th>
                <th className="px-3 py-1.5 font-normal">Matched</th>
                <th className="px-3 py-1.5 font-normal">Replaced with</th>
              </tr>
            </thead>
            <tbody>
              {analysis.spans.map((span, index) => (
                <tr key={index} className="border-b border-border/50 last:border-0">
                  <td className="px-3 py-1.5">
                    <span style={{ color: colorFor(span.category) }}>{span.entity_type}</span>
                  </td>
                  {/* The person's own text, back to their own browser. Isolated
                      so an Arabic value cannot reorder the row around it. */}
                  <td className="px-3 py-1.5">
                    <span dir="auto" className="bidi-auto font-mono">
                      {span.text}
                    </span>
                  </td>
                  <td className="px-3 py-1.5 font-mono text-text-2">{span.placeholder}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function Bubble({ turn }: { turn: ChatTurn }) {
  const mine = turn.role === "user";
  return (
    <div className={`flex ${mine ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-[46rem] ${mine ? "items-end" : "items-start"}`}>
        <div
          className={`whitespace-pre-wrap rounded-lg px-4 py-3 text-[14px] leading-relaxed ${
            mine ? "bg-bg-2 text-text-1" : "bg-bg-1 text-text-1 border border-border"
          }`}
        >
          {mine && turn.analysis ? (
            <SpannedText text={turn.content} spans={turn.analysis.spans} />
          ) : (
            <span dir="auto" className={`bidi-auto ${hasArabic(turn.content) ? "font-arabic" : ""}`}>
              {turn.content}
              {turn.streaming && <span className="ml-0.5 animate-pulse text-accent">▊</span>}
            </span>
          )}
          {turn.error && <div className="mt-2 text-[13px] text-blocked">{turn.error}</div>}
        </div>
        {mine && turn.analysis && <RedactionChip analysis={turn.analysis} />}
      </div>
    </div>
  );
}

export function Chat() {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [preview, setPreview] = useState<ChatAnalysis | null>(null);
  // Pre-send detection is a per-user toggle and OFF by default (brief §3).
  // On, it fires a debounced detection per pause in typing -- which is a real
  // request and a real audit row each time, not a free preview.
  const [liveDetect, setLiveDetect] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  useEffect(() => {
    if (!liveDetect || !draft.trim()) {
      setPreview(null);
      return;
    }
    const timer = setTimeout(() => {
      api
        .analyzeOnly(draft)
        .then((result) => setPreview(result.analysis))
        .catch(() => setPreview(null));
    }, 500);
    return () => clearTimeout(timer);
  }, [draft, liveDetect]);

  async function send() {
    const content = draft.trim();
    if (!content || sending) return;

    setSending(true);
    setPreview(null);
    const history: ChatTurn[] = [...turns, { role: "user", content }];
    setTurns(history);
    setDraft("");

    let blocked = false;

    try {
      await streamChat(
        history.map((turn) => ({ role: turn.role, content: turn.content })),
        {
          onAnalysis: (analysis) => {
            blocked = analysis.blocked;
            setTurns((current) => {
              const next = [...current];
              next[next.length - 1] = { ...next[next.length - 1], analysis };
              return next;
            });
            if (!analysis.blocked) {
              setTurns((current) => [
                ...current,
                { role: "assistant", content: "", streaming: true },
              ]);
            }
          },
          onDelta: (text) =>
            setTurns((current) => {
              const next = [...current];
              const last = next[next.length - 1];
              if (last?.role === "assistant") {
                next[next.length - 1] = { ...last, content: last.content + text };
              }
              return next;
            }),
          onError: (message) =>
            setTurns((current) => {
              const next = [...current];
              const last = next[next.length - 1];
              next[next.length - 1] = { ...last, error: message, streaming: false };
              return next;
            }),
          onDone: () =>
            setTurns((current) => {
              const next = [...current];
              const last = next[next.length - 1];
              if (last?.streaming) next[next.length - 1] = { ...last, streaming: false };
              return next;
            }),
        },
      );
    } catch {
      setTurns((current) => [
        ...current,
        { role: "assistant", content: "", error: "Could not reach the server." },
      ]);
    } finally {
      setSending(false);
      // A blocked message is not lost. It goes back in the composer with its
      // spans still highlighted, because the alternative is asking someone to
      // retype from memory the thing they were told not to send.
      if (blocked) setDraft(content);
    }
  }

  const blockedTurn = turns.at(-1);
  const showBlocked = blockedTurn?.role === "user" && blockedTurn.analysis?.blocked;

  function stripFlagged() {
    const spans = blockedTurn?.analysis?.spans ?? [];
    let text = draft;
    for (const span of [...spans].sort((a, b) => b.start - a.start)) {
      text = text.slice(0, span.start) + text.slice(span.end);
    }
    setDraft(text.replace(/\s{2,}/g, " ").trim());
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 overflow-y-auto px-6 py-6">
        <div className="mx-auto flex max-w-4xl flex-col gap-5">
          {turns.length === 0 && (
            <div className="mt-20 text-center">
              <div className="text-[15px] text-text-2">
                Messages are screened before they reach the model.
              </div>
              <div className="mt-2 text-[13px] text-text-3">
                Anything detected is replaced with a placeholder, and you will see exactly what
                was replaced.
              </div>
            </div>
          )}
          {turns.map((turn, index) => (
            <Bubble key={index} turn={turn} />
          ))}
          <div ref={endRef} />
        </div>
      </div>

      <div className="border-t border-border bg-bg-1 px-6 py-4">
        <div className="mx-auto max-w-4xl">
          {showBlocked && (
            <div className="mb-3 flex items-center justify-between rounded-md border border-blocked bg-blocked/10 px-3 py-2">
              <span className="text-[13px] text-blocked">
                This message was not sent. Remove the flagged text to continue.
              </span>
              <Button variant="danger" onClick={stripFlagged}>
                Remove flagged text
              </Button>
            </div>
          )}

          {liveDetect && preview && preview.spans.length > 0 && (
            <div className="mb-2 text-[12px] text-masked">
              {preview.spans.length} item{preview.spans.length === 1 ? "" : "s"} will be redacted
            </div>
          )}

          <div className="flex items-end gap-3">
            <textarea
              dir="auto"
              value={draft}
              rows={2}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void send();
                }
              }}
              placeholder="Type a message…"
              className="bidi-auto max-h-48 flex-1 resize-y rounded-md border border-border bg-bg-0 px-3 py-2 text-[14px] text-text-1 outline-none placeholder:text-text-3 focus:border-accent"
            />
            <Button onClick={() => void send()} disabled={sending || !draft.trim()}>
              {sending ? "Sending…" : "Send"}
            </Button>
          </div>

          <label className="mt-3 flex cursor-pointer items-center gap-2 text-[12px] text-text-3">
            <input
              type="checkbox"
              checked={liveDetect}
              onChange={(event) => setLiveDetect(event.target.checked)}
              className="accent-[var(--accent)]"
            />
            Check as I type
            <span className="text-text-3">
              — off by default; each pause sends the draft for detection
            </span>
          </label>
        </div>
      </div>
    </div>
  );
}
