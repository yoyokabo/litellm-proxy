import type {
  ChatAnalysis,
  EntityPolicyView,
  EntityUpsert,
  EventDetail,
  EventPage,
  FingerprintSummary,
  ReplacementStrategyInfo,
  SummaryStats,
  UserView,
} from "./types";

/** Thrown when the backend refuses because the bootstrap password stands. */
export class PasswordChangeRequired extends Error {}
export class Unauthorized extends Error {}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    // The session is an httpOnly cookie, so it never touches JS. Same-origin
    // in the container (nginx proxies /api), which is why there is no CORS
    // configuration anywhere in this app.
    credentials: "same-origin",
  });

  if (response.status === 401) throw new Unauthorized("not authenticated");
  if (response.status === 403) {
    const body = await response.json().catch(() => null);
    if (body?.detail?.code === "password_change_required") {
      throw new PasswordChangeRequired(body.detail.message);
    }
    throw new Error(body?.detail ?? "forbidden");
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(typeof body?.detail === "string" ? body.detail : `request failed (${response.status})`);
  }
  return response.status === 204 ? (undefined as T) : ((await response.json()) as T);
}

export const api = {
  me: () => request<UserView>("/api/auth/me"),

  login: (email: string, password: string) =>
    request<UserView>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),

  logout: () => request<void>("/api/auth/logout", { method: "POST" }),

  changePassword: (current_password: string, new_password: string) =>
    request<UserView>("/api/auth/password", {
      method: "POST",
      body: JSON.stringify({ current_password, new_password }),
    }),

  summary: (params: URLSearchParams) => request<SummaryStats>(`/api/admin/summary?${params}`),

  events: (params: URLSearchParams) => request<EventPage>(`/api/admin/events?${params}`),

  event: (id: number) => request<EventDetail>(`/api/admin/events/${id}`),

  fingerprint: (fp: string) => request<FingerprintSummary>(`/api/admin/fingerprints/${fp}`),

  flag: (id: number, verdict: "false_positive" | "confirmed", note?: string) =>
    request<void>(`/api/admin/events/${id}/flag`, {
      method: "POST",
      body: JSON.stringify({ verdict, note: note ?? null }),
    }),

  // -- entity policy ------------------------------------------------------
  //
  // These change what the gateway masks, for everyone, immediately -- no
  // restart and no redeploy. Each write returns the whole effective policy,
  // so the screen re-renders from the server's view rather than from a
  // locally patched copy that could disagree with it.

  entities: () => request<EntityPolicyView>("/api/admin/entities"),

  replacementStrategies: () =>
    request<{ strategies: ReplacementStrategyInfo[] }>("/api/admin/replacement-strategies"),

  saveEntity: (entity: EntityUpsert) =>
    request<EntityPolicyView>(`/api/admin/entities/${encodeURIComponent(entity.entity_type)}`, {
      method: "PUT",
      body: JSON.stringify(entity),
    }),

  resetEntity: (entityType: string) =>
    request<EntityPolicyView>(`/api/admin/entities/${encodeURIComponent(entityType)}`, {
      method: "DELETE",
    }),

  analyzeOnly: (content: string) =>
    request<{ analysis: ChatAnalysis }>("/api/chat/analyze", {
      method: "POST",
      body: JSON.stringify({ messages: [{ role: "user", content }] }),
    }),
};

export interface StreamHandlers {
  onAnalysis: (analysis: ChatAnalysis) => void;
  onDelta: (text: string) => void;
  onError: (message: string) => void;
  onDone: () => void;
}

/**
 * Read the chat SSE stream.
 *
 * Hand-rolled rather than EventSource for one reason that matters: EventSource
 * can only issue GET requests, and the message body belongs in a POST rather
 * than in a URL that lands in every access log between here and the backend.
 * A prompt in a query string is a prompt in nginx's log.
 */
export async function streamChat(
  messages: { role: "user" | "assistant"; content: string }[],
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "same-origin",
    body: JSON.stringify({ messages }),
    signal,
  });

  if (response.status === 401) throw new Unauthorized("not authenticated");
  if (!response.body) throw new Error("no response body");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line. A partial frame stays in the
    // buffer until the rest of it arrives -- without this, a token split
    // across two TCP reads would be dropped.
    let separator = buffer.indexOf("\n\n");
    while (separator !== -1) {
      const frame = buffer.slice(0, separator);
      buffer = buffer.slice(separator + 2);
      dispatch(frame, handlers);
      separator = buffer.indexOf("\n\n");
    }
  }
  handlers.onDone();
}

function dispatch(frame: string, handlers: StreamHandlers): void {
  let event = "";
  let data = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith("event: ")) event = line.slice(7);
    else if (line.startsWith("data: ")) data = line.slice(6);
  }
  if (!event) return;

  let payload: any = {};
  try {
    payload = data ? JSON.parse(data) : {};
  } catch {
    return;
  }

  if (event === "analysis") handlers.onAnalysis(payload as ChatAnalysis);
  else if (event === "delta") handlers.onDelta(payload.text ?? "");
  else if (event === "error") handlers.onError(payload.message ?? "stream error");
}
