import { useState } from "react";
import { api } from "../api";
import type { UserView } from "../types";
import { Button, Card, Input } from "./ui/primitives";

export function Login({ onAuthed }: { onAuthed: (user: UserView) => void }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit() {
    setBusy(true);
    setError(null);
    try {
      onAuthed(await api.login(email, password));
    } catch {
      // One message for every failure. The backend does not distinguish
      // "no such account" from "wrong password", and neither does this.
      setError("Invalid email or password.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex h-full items-center justify-center px-4">
      <Card className="w-full max-w-sm p-6">
        <h1 className="text-[16px] font-semibold">PII Guardrail</h1>
        <p className="mt-1 text-[13px] text-text-3">Sign in to the chat and admin console.</p>

        <form
          className="mt-5 flex flex-col gap-3"
          onSubmit={(event) => {
            event.preventDefault();
            void submit();
          }}
        >
          <Input value={email} onChange={setEmail} placeholder="Email" autoFocus />
          <Input value={password} onChange={setPassword} placeholder="Password" type="password" />
          {error && <div className="text-[12px] text-blocked">{error}</div>}
          <Button type="submit" disabled={busy || !email || !password}>
            {busy ? "Signing in…" : "Sign in"}
          </Button>
        </form>
      </Card>
    </div>
  );
}

export function ChangePassword({ onDone }: { onDone: (user: UserView) => void }) {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit() {
    if (next !== confirm) {
      setError("The two new passwords do not match.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      onDone(await api.changePassword(current, next));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not change the password.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex h-full items-center justify-center px-4">
      <Card className="w-full max-w-sm p-6">
        <h1 className="text-[16px] font-semibold">Change your password</h1>
        {/* Says why, not just that. A gate with no stated reason reads as a
            bug, and people work around bugs. */}
        <p className="mt-2 text-[13px] leading-relaxed text-text-2">
          This account still uses the password from the install configuration, which means it is
          written down somewhere. The console stays locked until it is changed.
        </p>

        <form
          className="mt-5 flex flex-col gap-3"
          onSubmit={(event) => {
            event.preventDefault();
            void submit();
          }}
        >
          <Input
            value={current}
            onChange={setCurrent}
            placeholder="Current password"
            type="password"
            autoFocus
          />
          <Input
            value={next}
            onChange={setNext}
            placeholder="New password (12+ characters)"
            type="password"
          />
          <Input
            value={confirm}
            onChange={setConfirm}
            placeholder="Repeat new password"
            type="password"
          />
          {error && <div className="text-[12px] text-blocked">{error}</div>}
          <Button type="submit" disabled={busy || next.length < 12 || !current}>
            {busy ? "Saving…" : "Change password"}
          </Button>
        </form>
      </Card>
    </div>
  );
}
