export type Category = "id" | "person" | "contact" | "location" | "finance" | "other";

export interface UserView {
  id: number;
  email: string;
  must_change_password: boolean;
}

/**
 * The admin panel no longer draws a chart, so nothing in this app consumes
 * these two. They describe /api/admin/timeline, which the backend still
 * serves -- kept so restoring the chart is a component and not an API change.
 */
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

// ---------------------------------------------------------------------------
// Entity policy administration
// ---------------------------------------------------------------------------

/**
 * How a masked span is rewritten. The names match pii-service's
 * ReplacementStrategy; the trade-offs are described by the backend and shown
 * in the dropdown rather than duplicated here, because the reasoning belongs
 * next to the implementation that has to honour it.
 */
export type ReplacementStrategyName =
  | "placeholder"
  | "constant"
  | "surrogate"
  | "redact"
  | "labelled_fingerprint";

export interface ReplacementStrategyInfo {
  name: ReplacementStrategyName;
  example: string;
  realistic: boolean;
  description: string;
}

export interface EntityView {
  entity_type: string;
  /** "baseline" = entities.yaml; "overlay" = changed or added by an operator. */
  source: "baseline" | "overlay";
  category: Category;
  action: string;
  score_threshold: number;
  placeholder: string;
  /** 1 = pattern, 2 = Arabic NER, 3 = a schema-conditioned label. */
  tier: number | null;
  /** The English phrase tier 3 is conditioned on, for the labels that have one. */
  gliner_prompt: string | null;
  replacement_strategy: ReplacementStrategyName;
  replacement_example: string;
  /** True when the replacement looks like real data rather than a placeholder. */
  replacement_is_realistic: boolean;
  enabled: boolean;
  note: string | null;
  updated_by: string | null;
}

export interface EntityPolicyView {
  entities: EntityView[];
  /** prompt -> entity type, exactly as tier 3 is configured. */
  tier3_labels: Record<string, string>;
  tier3_enabled: boolean;
  realistic_replacement_entities: string[];
  /** Consequences an operator should read before trusting the screen. */
  warnings: string[];
}

/** The body of a PUT. Every field but entity_type is optional. */
export interface EntityUpsert {
  entity_type: string;
  gliner_prompt?: string | null;
  category?: Category | null;
  action?: string | null;
  score_threshold?: number | null;
  placeholder?: string | null;
  replacement?: { strategy: ReplacementStrategyName; value?: string | null; pool?: string[] };
  enabled?: boolean;
  note?: string | null;
}
