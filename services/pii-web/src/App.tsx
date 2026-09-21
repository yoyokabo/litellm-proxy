import { useEffect, useState } from "react";
import { api, PasswordChangeRequired, Unauthorized } from "./api";
import type { UserView } from "./types";
import { Admin } from "./components/Admin";
import { Chat } from "./components/Chat";
import { Entities } from "./components/Entities";
import { ChangePassword, Login } from "./components/Login";

type Tab = "chat" | "admin" | "entities";

export default function App() {
  const [user, setUser] = useState<UserView | null>(null);
  const [ready, setReady] = useState(false);
  const [tab, setTab] = useState<Tab>("chat");
  const [theme, setTheme] = useState<"dark" | "light">("dark");

  useEffect(() => {
    api
      .me()
      .then(setUser)
      .catch((caught) => {
        if (!(caught instanceof Unauthorized || caught instanceof PasswordChangeRequired)) {
          // Nothing to do but show the login screen.
        }
        setUser(null);
      })
      .finally(() => setReady(true));
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle("light", theme === "light");
    document.documentElement.classList.toggle("dark", theme === "dark");
  }, [theme]);

  if (!ready) return <div className="p-6 text-[13px] text-text-3">Loading…</div>;
  if (!user) return <Login onAuthed={setUser} />;
  if (user.must_change_password) return <ChangePassword onDone={setUser} />;

  return (
    <div className="flex h-full flex-col">
      <header className="flex shrink-0 items-center justify-between border-b border-border bg-bg-1 px-6 py-3">
        <div className="flex items-center gap-6">
          <span className="text-[14px] font-semibold">PII Guardrail</span>
          <nav className="flex gap-1">
            {(["chat", "admin", "entities"] as const).map((option) => (
              <button
                key={option}
                onClick={() => setTab(option)}
                className={`rounded-md px-3 py-1.5 text-[13px] capitalize transition-colors ${
                  tab === option ? "bg-bg-2 text-text-1" : "text-text-3 hover:text-text-1"
                }`}
              >
                {option}
              </button>
            ))}
          </nav>
        </div>

        <div className="flex items-center gap-4">
          <button
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            title="Toggle theme"
            className="text-[13px] text-text-3 hover:text-text-1"
          >
            {theme === "dark" ? "☾" : "☀"}
          </button>
          <span className="font-mono text-[12px] text-text-3">{user.email}</span>
          <button
            onClick={() => void api.logout().then(() => setUser(null))}
            className="text-[13px] text-text-3 hover:text-text-1"
          >
            Sign out
          </button>
        </div>
      </header>

      <main className="min-h-0 flex-1">
        {tab === "chat" && <Chat />}
        {tab === "admin" && <Admin />}
        {tab === "entities" && <Entities />}
      </main>
    </div>
  );
}
