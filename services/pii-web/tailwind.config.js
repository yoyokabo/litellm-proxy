/** @type {import('tailwindcss').Config} */
// The palette is brief §9, exposed to Tailwind by name so a component writes
// `bg-bg-1` / `text-ent-person` rather than an arbitrary hex. Every value
// resolves to a CSS variable defined in index.css, which is what lets the
// light theme swap without any class changing.
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        "bg-0": "var(--bg-0)",
        "bg-1": "var(--bg-1)",
        "bg-2": "var(--bg-2)",
        border: "var(--border)",
        "text-1": "var(--text-1)",
        "text-2": "var(--text-2)",
        "text-3": "var(--text-3)",
        accent: "var(--accent)",
        "accent-hover": "var(--accent-hover)",
        "accent-dim": "var(--accent-dim)",
        ok: "var(--ok)",
        masked: "var(--masked)",
        blocked: "var(--blocked)",
        info: "var(--info)",
        "ent-id": "var(--ent-id)",
        "ent-person": "var(--ent-person)",
        "ent-contact": "var(--ent-contact)",
        "ent-location": "var(--ent-location)",
        "ent-finance": "var(--ent-finance)",
        "ent-other": "var(--ent-other)",
      },
      fontFamily: {
        sans: ["Inter", "IBM Plex Sans Arabic", "system-ui", "sans-serif"],
        arabic: ["IBM Plex Sans Arabic", "Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      // 4px base grid, 8px rhythm (§9).
      spacing: { 1: "4px", 2: "8px", 3: "12px", 4: "16px", 6: "24px", 8: "32px" },
    },
  },
  plugins: [],
};
