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
