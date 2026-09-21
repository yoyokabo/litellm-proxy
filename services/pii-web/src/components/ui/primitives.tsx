/**
 * Local UI primitives.
 *
 * The brief names shadcn/ui, whose actual model is "copy the component into
 * your repo and own it" rather than "install a dependency". These are those
 * components, written against the §9 tokens directly. Keeping them local is
 * the point of that model; it also means the air-gapped build pulls no
 * component library at all.
 */
import type { ReactNode } from "react";

export function Button({
  children,
  onClick,
  variant = "primary",
  type = "button",
  disabled,
  title,
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "primary" | "ghost" | "danger";
  type?: "button" | "submit";
  disabled?: boolean;
  title?: string;
}) {
  const styles = {
    primary: "bg-accent text-bg-0 hover:bg-accent-hover",
    ghost: "bg-bg-2 text-text-1 hover:bg-border",
    danger: "bg-transparent text-blocked border border-blocked hover:bg-blocked/10",
  }[variant];

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`rounded-md px-3 py-2 text-[13px] font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${styles}`}
    >
      {children}
    </button>
  );
}

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div className={`rounded-lg border border-border bg-bg-1 ${className}`}>{children}</div>
  );
}

export function Stat({ label, value, tone }: { label: string; value: ReactNode; tone?: string }) {
  return (
    <Card className="px-4 py-3">
      <div className="text-[11px] uppercase tracking-wide text-text-3">{label}</div>
      <div className="mt-1 font-mono text-xl" style={tone ? { color: tone } : undefined}>
        {value}
      </div>
    </Card>
  );
}

export function Input({
  value,
  onChange,
  placeholder,
  type = "text",
  autoFocus,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  type?: string;
  autoFocus?: boolean;
}) {
  return (
    <input
      type={type}
      value={value}
      autoFocus={autoFocus}
      placeholder={placeholder}
      onChange={(event) => onChange(event.target.value)}
      className="w-full rounded-md border border-border bg-bg-0 px-3 py-2 text-[13px] text-text-1 outline-none placeholder:text-text-3 focus:border-accent"
    />
  );
}

/**
 * Action badge. Colour *and* shape differ, never colour alone (§9):
 * masked is filled, blocked is outlined. That survives a colour-blind reader
 * and a greyscale screenshot, both of which happen to audit evidence.
 */
export function ActionBadge({ action }: { action: string }) {
  if (action === "BLOCK") {
    return (
      <span className="rounded border border-blocked px-1.5 py-0.5 font-mono text-[11px] text-blocked">
        BLOCK
      </span>
    );
  }
  if (action === "ALLOW") {
    return (
      <span className="rounded border border-text-3 px-1.5 py-0.5 font-mono text-[11px] text-text-2">
        ALLOW
      </span>
    );
  }
  return (
    <span className="rounded bg-masked px-1.5 py-0.5 font-mono text-[11px] font-medium text-bg-0">
      MASK
    </span>
  );
}

export function CategoryDot({ category }: { category: string }) {
  return (
    <span
      className="inline-block h-2 w-2 shrink-0 rounded-full"
      style={{ background: `var(--ent-${category}, var(--ent-other))` }}
    />
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="px-4 py-10 text-center text-[13px] text-text-3">{children}</div>;
}

export function Select<T extends string>({
  value,
  onChange,
  options,
}: {
  value: T;
  onChange: (value: T) => void;
  options: { value: T; label: string }[];
}) {
  return (
    <select
      value={value}
      onChange={(event) => onChange(event.target.value as T)}
      className="w-full rounded-md border border-border bg-bg-0 px-3 py-2 text-[13px] text-text-1 outline-none focus:border-accent"
    >
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  );
}

export function Textarea({
  value,
  onChange,
  placeholder,
  rows = 4,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  rows?: number;
}) {
  return (
    <textarea
      value={value}
      rows={rows}
      placeholder={placeholder}
      onChange={(event) => onChange(event.target.value)}
      className="w-full resize-y rounded-md border border-border bg-bg-0 px-3 py-2 font-mono text-[12px] text-text-1 outline-none placeholder:text-text-3 focus:border-accent"
    />
  );
}

/** A form row: label above, control below, optional hint under it. */
export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: ReactNode;
  children: ReactNode;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-[11px] uppercase tracking-wide text-text-3">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-[12px] leading-snug text-text-3">{hint}</span>}
    </label>
  );
}

/**
 * The tier a label is detected by. Tier 3 is the one an operator can add, so
 * it is the one that needs to be visually distinct in a list of fifteen.
 */
export function TierBadge({ tier }: { tier: number | null }) {
  if (tier === null) return <span className="text-[11px] text-text-3">—</span>;
  const label = { 1: "pattern", 2: "Arabic NER", 3: "label" }[tier] ?? `tier ${tier}`;
  return (
    <span className="rounded border border-border px-1.5 py-0.5 font-mono text-[11px] text-text-2">
      T{tier} {label}
    </span>
  );
}

/**
 * Marks a replacement that looks like real data.
 *
 * The one consequence of "John Doe" an operator must not discover later: a
 * masked prompt stops looking masked, and a real person named John Doe is
 * never masked as a person. Warning-coloured, and never colour alone.
 */
export function RealisticBadge() {
  return (
    <span
      className="rounded border border-masked px-1.5 py-0.5 text-[11px] text-masked"
      title="Looks like real data: a masked prompt no longer looks masked, and a value matching this replacement is left alone."
    >
      ⚠ realistic
    </span>
  );
}
