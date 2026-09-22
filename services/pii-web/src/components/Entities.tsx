/**
 * Entities: what the gateway detects, and what it writes in place of it.
 *
 * Two jobs, and the screen is laid out around the fact that they are different
 * kinds of change:
 *
 *  - **Add a custom label.** An English phrase -- "internal project codename",
 *    "employee badge number" -- that tier 3 is conditioned on at inference.
 *    New capability, so it gets its own panel rather than hiding behind a "+"
 *    in a table header.
 *  - **Change a replacement.** Per entity, so it belongs in the row for that
 *    entity, and the editor opens beside the list rather than as a route
 *    change, keeping the other fourteen entities visible as context.
 *
 * Every write returns the whole effective policy and the screen re-renders
 * from that, never from a locally patched copy. A screen that disagrees with
 * what the gateway is actually doing is worse than one that takes a round
 * trip to find out.
 *
 * The screen leads with consequences rather than burying them in a tooltip.
 * "Replace PERSON with John Doe" sounds harmless and is not: the prompt stops
 * looking masked to a human reader, and a real person named John Doe stops
 * being masked at all. The backend computes those warnings -- this renders
 * them where the operator cannot miss them.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type {
  EntityAction,
  Category,
  EntityPolicyView,
  EntityUpsert,
  EntityView,
  ReplacementStrategyInfo,
  ReplacementStrategyName,
} from "../types";
import {
  Button,
  Card,
  CategoryDot,
  Empty,
  Field,
  Input,
  RealisticBadge,
  Select,
  Textarea,
  TierBadge,
} from "./ui/primitives";

const CATEGORIES: Category[] = ["id", "person", "contact", "location", "finance", "other"];

/** Editor state, kept as strings so a half-typed pool is not an error yet. */
interface Draft {
  entity_type: string;
  gliner_prompt: string;
  category: Category;
  action: EntityAction;
  placeholder: string;
  strategy: ReplacementStrategyName;
  value: string;
  pool: string;
  note: string;
}

/** The three things policy can do with a detected span.
 *
 * Named for what an operator wants, not for the enum: "detect only" is the
 * answer to "turn masking off for this one", and it is ALLOW on the wire.
 * Kept as three choices rather than a switch because BLOCK is a real third
 * state, and a two-way toggle would silently throw it away.
 */
const ACTION_OPTIONS: { value: EntityAction; label: string }[] = [
  { value: "MASK", label: "Mask it" },
  { value: "ALLOW", label: "Detect only — do not mask" },
  { value: "BLOCK", label: "Block the request" },
];

const ACTION_HINT: Record<EntityAction, string> = {
  MASK: "The span is replaced before the text reaches the model.",
  ALLOW: "Detected and audited, but the text is passed through untouched.",
  BLOCK: "The request is refused. The caller is told the type and count, never the value.",
};

function draftFrom(entity: EntityView): Draft {
  return {
    entity_type: entity.entity_type,
    gliner_prompt: entity.gliner_prompt ?? "",
    category: entity.category,
    action: (entity.action as EntityAction) ?? "MASK",
    placeholder: entity.placeholder,
    strategy: entity.replacement_strategy,
    // The server does not send back what a constant or a pool was set to --
    // the policy view reports the rendered example, which is the thing an
    // operator needs to read. Re-entering them on edit is the cost of that,
    // and it is the right trade: a stale echo of a pool would be worse.
    value: "",
    pool: "",
    note: entity.note ?? "",
  };
}

const BLANK: Draft = {
  entity_type: "",
  gliner_prompt: "",
  category: "other",
  action: "MASK",
  placeholder: "",
  strategy: "placeholder",
  value: "",
  pool: "",
  note: "",
};

function toPayload(draft: Draft, { isNew }: { isNew: boolean }): EntityUpsert {
  const pool = draft.pool
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  const payload: EntityUpsert = {
    entity_type: draft.entity_type.trim().toUpperCase(),
    replacement: {
      strategy: draft.strategy,
      value: draft.value.trim() || null,
      pool,
    },
  };
  payload.action = draft.action;
  if (draft.gliner_prompt.trim()) payload.gliner_prompt = draft.gliner_prompt.trim();
  if (draft.placeholder.trim()) payload.placeholder = draft.placeholder.trim();
  if (draft.note.trim()) payload.note = draft.note.trim();
  // Category only matters when introducing a label: for an existing entity,
  // sending it would overwrite a value the operator did not touch.
  if (isNew) payload.category = draft.category;
  return payload;
}

export function Entities() {
  const [policy, setPolicy] = useState<EntityPolicyView | null>(null);
  const [strategies, setStrategies] = useState<ReplacementStrategyInfo[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    setError(null);
    try {
      const [policyData, strategyData] = await Promise.all([
        api.entities(),
        api.replacementStrategies(),
      ]);
      setPolicy(policyData);
      setStrategies(strategyData.strategies);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "failed to load the policy");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const entity = useMemo(
    () => policy?.entities.find((candidate) => candidate.entity_type === selected) ?? null,
    [policy, selected],
  );

  async function run(action: () => Promise<EntityPolicyView>, onDone?: () => void) {
    setBusy(true);
    setError(null);
    try {
      setPolicy(await action());
      onDone?.();
    } catch (caught) {
      // Upstream refusals are the useful ones and arrive with their reason
      // intact -- "PROJECT_CODENAME is not in the baseline policy, so it needs
      // a gliner_prompt". Shown verbatim.
      setError(caught instanceof Error ? caught.message : "the change was refused");
    } finally {
      setBusy(false);
    }
  }

  if (!policy) {
    return (
      <div className="p-6 text-[13px] text-text-3">
        {error ? <span className="text-blocked">{error}</span> : "Loading policy…"}
      </div>
    );
  }

  return (
    <div className="flex h-full">
      <div className="flex min-w-0 flex-1 flex-col overflow-y-auto">
        <div className="shrink-0 border-b border-border px-6 py-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="text-[14px] font-semibold">Entities and replacements</h2>
              <p className="mt-0.5 text-[12px] text-text-3">
                What the gateway detects, and what it writes in place of it. Changes take effect
                immediately, for every client of the proxy.
              </p>
            </div>
            <div className="flex gap-2">
              <Button variant="ghost" onClick={() => void load()}>
                Refresh
              </Button>
              <Button
                onClick={() => {
                  setAdding(true);
                  setSelected(null);
                }}
              >
                Add a custom label
              </Button>
            </div>
          </div>

          {(error || policy.warnings.length > 0) && (
            <div className="mt-3 space-y-2">
              {error && (
                <div className="rounded-md border border-blocked px-3 py-2 text-[12px] text-blocked">
                  {error}
                </div>
              )}
              {policy.warnings.map((warning) => (
                <div
                  key={warning}
                  className="rounded-md border border-masked px-3 py-2 text-[12px] leading-snug text-masked"
                >
                  ⚠ {warning}
                </div>
              ))}
            </div>
          )}
        </div>

        <EntityList
          entities={policy.entities}
          selected={selected}
          onSelect={(name) => {
            setSelected(name);
            setAdding(false);
          }}
        />
      </div>

      {(adding || entity) && (
        <aside className="flex w-[26rem] shrink-0 flex-col overflow-y-auto border-l border-border bg-bg-1">
          {adding ? (
            <NewLabelPanel
              strategies={strategies}
              busy={busy}
              tier3Enabled={policy.tier3_enabled}
              onClose={() => setAdding(false)}
              onSave={(draft) =>
                void run(
                  () => api.saveEntity(toPayload(draft, { isNew: true })),
                  () => {
                    setAdding(false);
                    setSelected(draft.entity_type.trim().toUpperCase());
                  },
                )
              }
            />
          ) : (
            entity && (
              <EntityPanel
                key={entity.entity_type}
                entity={entity}
                strategies={strategies}
                busy={busy}
                onClose={() => setSelected(null)}
                onSave={(draft) => void run(() => api.saveEntity(toPayload(draft, { isNew: false })))}
                onReset={() =>
                  void run(() => api.resetEntity(entity.entity_type), () => setSelected(null))
                }
              />
            )
          )}
        </aside>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------

function EntityList({
  entities,
  selected,
  onSelect,
}: {
  entities: EntityView[];
  selected: string | null;
  onSelect: (entityType: string) => void;
}) {
  if (entities.length === 0) return <Empty>No entities configured.</Empty>;

  return (
    <table className="w-full border-collapse text-[13px]">
      <thead className="sticky top-0 bg-bg-0 text-[11px] uppercase tracking-wide text-text-3">
        <tr className="border-b border-border">
          <th className="px-6 py-2 text-left font-medium">Entity</th>
          <th className="px-3 py-2 text-left font-medium">Detected by</th>
          <th className="px-3 py-2 text-left font-medium">Label prompt</th>
          <th className="px-3 py-2 text-left font-medium">Replaced with</th>
        </tr>
      </thead>
      <tbody>
        {entities.map((entity) => (
          <tr
            key={entity.entity_type}
            onClick={() => onSelect(entity.entity_type)}
            className={`cursor-pointer border-b border-border/60 transition-colors hover:bg-bg-1 ${
              selected === entity.entity_type ? "bg-bg-2" : ""
            }`}
          >
            <td className="px-6 py-2.5">
              <div className="flex items-center gap-2">
                <CategoryDot category={entity.category} />
                <span className="font-mono text-[12px]">{entity.entity_type}</span>
                {entity.source === "overlay" && (
                  <span
                    className="rounded border border-accent px-1.5 py-0.5 text-[11px] text-accent"
                    title={
                      entity.updated_by
                        ? `changed by ${entity.updated_by}`
                        : "changed from the shipped policy"
                    }
                  >
                    custom
                  </span>
                )}
              </div>
            </td>
            <td className="px-3 py-2.5">
              <TierBadge tier={entity.tier} />
            </td>
            <td className="px-3 py-2.5 text-[12px] text-text-2">
              {entity.gliner_prompt ?? <span className="text-text-3">—</span>}
            </td>
            <td className="px-3 py-2.5">
              <div className="flex items-center gap-2">
                <code className="rounded bg-bg-2 px-1.5 py-0.5 font-mono text-[12px] text-text-1">
                  {entity.replacement_example || "(removed)"}
                </code>
                {entity.replacement_is_realistic && <RealisticBadge />}
              </div>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ---------------------------------------------------------------------------

/** Strategy dropdown plus the fields that strategy needs, and nothing else. */
function ReplacementEditor({
  draft,
  onChange,
  strategies,
}: {
  draft: Draft;
  onChange: (draft: Draft) => void;
  strategies: ReplacementStrategyInfo[];
}) {
  const chosen = strategies.find((candidate) => candidate.name === draft.strategy);

  return (
    <>
      <Field
        label="Replace with"
        hint={
          chosen && (
            <>
              {chosen.description}{" "}
              <span className="text-text-2">Example: {chosen.example || "(removed)"}</span>
            </>
          )
        }
      >
        <Select
          value={draft.strategy}
          onChange={(strategy) => onChange({ ...draft, strategy })}
          options={
            strategies.length > 0
              ? strategies.map((strategy) => ({
                  value: strategy.name,
                  label: strategy.realistic ? `${strategy.name} ⚠ realistic` : strategy.name,
                }))
              : [{ value: draft.strategy, label: draft.strategy }]
          }
        />
      </Field>

      {draft.strategy === "constant" && (
        <Field
          label="Replacement text"
          hint="Every match becomes this. Two different people in one message become the same name, so prefer surrogate when that matters."
        >
          <Input
            value={draft.value}
            onChange={(value) => onChange({ ...draft, value })}
            placeholder="John Doe"
          />
        </Field>
      )}

      {draft.strategy === "surrogate" && (
        <Field
          label="Surrogate pool, one per line"
          hint="Picked by the value's fingerprint: the same person is the same fake name in every turn, and two people stay two people. No mapping is stored anywhere."
        >
          <Textarea
            value={draft.pool}
            onChange={(pool) => onChange({ ...draft, pool })}
            placeholder={"John Doe\nJane Roe\nSam Poe\nAlex Coe"}
            rows={5}
          />
        </Field>
      )}

      {(draft.strategy === "placeholder" || draft.strategy === "labelled_fingerprint") && (
        <Field label="Placeholder" hint="The token written into the text.">
          <Input
            value={draft.placeholder}
            onChange={(placeholder) => onChange({ ...draft, placeholder })}
            placeholder="<PERSON>"
          />
        </Field>
      )}
    </>
  );
}

function PanelHeader({
  title,
  subtitle,
  onClose,
}: {
  title: string;
  subtitle?: string;
  onClose: () => void;
}) {
  return (
    <div className="flex items-start justify-between border-b border-border px-5 py-4">
      <div>
        <div className="font-mono text-[13px]">{title}</div>
        {subtitle && <div className="mt-0.5 text-[12px] text-text-3">{subtitle}</div>}
      </div>
      <button onClick={onClose} className="text-[13px] text-text-3 hover:text-text-1">
        ✕
      </button>
    </div>
  );
}

function EntityPanel({
  entity,
  strategies,
  busy,
  onSave,
  onReset,
  onClose,
}: {
  entity: EntityView;
  strategies: ReplacementStrategyInfo[];
  busy: boolean;
  onSave: (draft: Draft) => void;
  onReset: () => void;
  onClose: () => void;
}) {
  const [draft, setDraft] = useState<Draft>(() => draftFrom(entity));

  return (
    <>
      <PanelHeader
        title={entity.entity_type}
        subtitle={`${entity.category} · ${entity.action} · threshold ${entity.score_threshold}`}
        onClose={onClose}
      />

      <div className="space-y-4 px-5 py-4">
        <Card className="px-3 py-2 text-[12px] text-text-2">
          Currently masked as{" "}
          <code className="font-mono text-text-1">{entity.replacement_example || "(removed)"}</code>
          {entity.updated_by && (
            <div className="mt-1 text-text-3">Last changed by {entity.updated_by}</div>
          )}
        </Card>

        {entity.tier === 3 && (
          <Field
            label="Label prompt"
            hint="The English phrase the model is conditioned on. Describe the thing in words, not as an identifier."
          >
            <Input
              value={draft.gliner_prompt}
              onChange={(gliner_prompt) => setDraft({ ...draft, gliner_prompt })}
              placeholder="internal project codename"
            />
          </Field>
        )}

        <Field
          label="When this is found"
          hint={ACTION_HINT[draft.action]}
        >
          <Select
            value={draft.action}
            onChange={(action) => setDraft({ ...draft, action })}
            options={ACTION_OPTIONS}
          />
        </Field>

        {draft.action !== "MASK" && (
          <Card className="border-masked px-3 py-2 text-[12px]">
            {draft.action === "ALLOW" ? (
              <>
                <strong>Masking is off for this entity.</strong> It is still detected and still
                written to the audit log, so you keep the record of who sent what — the text
                simply reaches the model unchanged.
              </>
            ) : (
              <>
                <strong>Requests carrying this are refused.</strong> The caller gets entity types
                and counts, never the value. Nothing reaches the model.
              </>
            )}
          </Card>
        )}

        {draft.action === "MASK" && (
          <ReplacementEditor draft={draft} onChange={setDraft} strategies={strategies} />
        )}

        <Field label="Note" hint="Why this was changed. Shown to the next operator.">
          <Input
            value={draft.note}
            onChange={(note) => setDraft({ ...draft, note })}
            placeholder="Requested by legal, ticket OPS-214"
          />
        </Field>

        <div className="flex items-center gap-2 pt-1">
          <Button onClick={() => onSave(draft)} disabled={busy}>
            {busy ? "Saving…" : "Save"}
          </Button>
          {entity.source === "overlay" && (
            <Button
              variant="danger"
              onClick={onReset}
              disabled={busy}
              title="Discard this override and go back to the shipped policy"
            >
              Reset to default
            </Button>
          )}
        </div>
      </div>
    </>
  );
}

function NewLabelPanel({
  strategies,
  busy,
  tier3Enabled,
  onSave,
  onClose,
}: {
  strategies: ReplacementStrategyInfo[];
  busy: boolean;
  tier3Enabled: boolean;
  onSave: (draft: Draft) => void;
  onClose: () => void;
}) {
  const [draft, setDraft] = useState<Draft>(BLANK);
  const ready = /^[A-Za-z][A-Za-z0-9_]{1,63}$/.test(draft.entity_type.trim())
    && draft.gliner_prompt.trim().length >= 2;

  return (
    <>
      <PanelHeader
        title="New label"
        subtitle="Detected by describing it in English -- no retraining"
        onClose={onClose}
      />

      <div className="space-y-4 px-5 py-4">
        {!tier3Enabled && (
          <div className="rounded-md border border-masked px-3 py-2 text-[12px] leading-snug text-masked">
            ⚠ Tier 3 is disabled on this deployment, so a new label is saved but detects nothing
            until <code className="font-mono">PII_ENABLE_TIER3_GLINER</code> is turned on.
          </div>
        )}

        <Field label="Entity type" hint="Uppercase, underscores. How it appears in the audit log.">
          <Input
            value={draft.entity_type}
            onChange={(entity_type) => setDraft({ ...draft, entity_type })}
            placeholder="EMPLOYEE_BADGE"
            autoFocus
          />
        </Field>

        <Field
          label="What to look for, in English"
          hint="A description, not an identifier: “employee badge number”, not “EMPLOYEE_BADGE”. The model is conditioned on this phrase at inference."
        >
          <Input
            value={draft.gliner_prompt}
            onChange={(gliner_prompt) => setDraft({ ...draft, gliner_prompt })}
            placeholder="employee badge number"
          />
        </Field>

        <Field label="Category" hint="Decides the colour it gets in the audit timeline.">
          <Select
            value={draft.category}
            onChange={(category) => setDraft({ ...draft, category })}
            options={CATEGORIES.map((category) => ({ value: category, label: category }))}
          />
        </Field>

        <Field
          label="When this is found"
          hint={ACTION_HINT[draft.action]}
        >
          <Select
            value={draft.action}
            onChange={(action) => setDraft({ ...draft, action })}
            options={ACTION_OPTIONS}
          />
        </Field>

        {draft.action !== "MASK" && (
          <Card className="border-masked px-3 py-2 text-[12px]">
            {draft.action === "ALLOW" ? (
              <>
                <strong>Masking is off for this entity.</strong> It is still detected and still
                written to the audit log, so you keep the record of who sent what — the text
                simply reaches the model unchanged.
              </>
            ) : (
              <>
                <strong>Requests carrying this are refused.</strong> The caller gets entity types
                and counts, never the value. Nothing reaches the model.
              </>
            )}
          </Card>
        )}

        {draft.action === "MASK" && (
          <ReplacementEditor draft={draft} onChange={setDraft} strategies={strategies} />
        )}

        <Field label="Note">
          <Input
            value={draft.note}
            onChange={(note) => setDraft({ ...draft, note })}
            placeholder="Why this label exists"
          />
        </Field>

        <div className="pt-1">
          <Button onClick={() => onSave(draft)} disabled={busy || !ready}>
            {busy ? "Saving…" : "Create label"}
          </Button>
        </div>
      </div>
    </>
  );
}
