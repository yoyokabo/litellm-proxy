import type { Category } from "../types";

/** Category -> CSS variable. One place, so the chart, the table, the chat
 *  underline and the drill-down badge can never drift apart. */
export const CATEGORY_COLOR: Record<Category, string> = {
  id: "var(--ent-id)",
  person: "var(--ent-person)",
  contact: "var(--ent-contact)",
  location: "var(--ent-location)",
  finance: "var(--ent-finance)",
  other: "var(--ent-other)",
};

export function colorFor(category: string): string {
  return CATEGORY_COLOR[(category as Category) ?? "other"] ?? CATEGORY_COLOR.other;
}

export function formatDateTime(iso: string | null): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

/** True when a string contains Arabic-script characters. Used only to pick a
 *  font stack and a text direction hint, never to route detection. */
export function hasArabic(text: string): boolean {
  return /[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]/.test(text);
}
