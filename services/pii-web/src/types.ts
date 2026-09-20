export type Category = "id" | "person" | "contact" | "location" | "finance" | "other";

export interface UserView {
  id: number;
  email: string;
  must_change_password: boolean;
}

export interface TimelineBucket {
  bucket: string;
  entity_type: string;
  category: Category;
  total: number;
  blocked: number;
}

export interface SummaryStats {
  events: number;
  users: number;
  distinct_values: number;
  blocked: number;
  masked: number;
}

export interface TimelineResponse {
  since: string;
  until: string;
  bucket: "minute" | "hour" | "day";
  points: TimelineBucket[];
  summary: SummaryStats;
  categories: Record<string, Category>;
}

export interface EventDetail {
  id: number;
  ts: string;
  request_id: string;
  user_id: string | null;
  team_id: string | null;
  key_alias: string | null;
  end_user_id: string | null;
  entity_type: string;
  category: Category;
  recognizer: string;
  score: number | null;
  action: string;
  span_start: number | null;
  span_end: number | null;
  value_len: number | null;
  message_index: number | null;
  message_role: string | null;
  field: string | null;
  value_fp: string | null;
  preview: string | null;
  model: string | null;
  lang: string | null;
  latency_ms: number | null;
  flag: string | null;
}

export interface EventPage {
  total: number;
  offset: number;
  limit: number;
  events: EventDetail[];
}

export interface PerUserOccurrence {
  user_id: string | null;
  occurrences: number;
  last_seen: string | null;
}

export interface FingerprintSummary {
  value_fp: string;
  occurrences: number;
  distinct_users: number;
  distinct_requests: number;
  first_seen: string | null;
  last_seen: string | null;
  per_user: PerUserOccurrence[];
}

export interface ChatSpan {
  entity_type: string;
  category: Category;
  action: string;
  start: number;
  end: number;
  text: string;
  placeholder: string;
  score: number;
}

export interface ChatAnalysis {
  spans: ChatSpan[];
  masked_text: string;
  blocked: boolean;
  block_reason: Record<string, number> | null;
  counts: Record<string, number>;
  latency_ms: number;
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
  analysis?: ChatAnalysis;
  streaming?: boolean;
  error?: string;
}
