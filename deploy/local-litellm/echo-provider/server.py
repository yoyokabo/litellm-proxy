"""A deterministic OpenAI-compatible provider that echoes what it received.

Test fixture for the local stand-in deployment, not product.

Done-criterion 3 is "comes back masked", and the honest way to check that is to
look at what the *provider* was handed, not at what a language model chose to
say about it. An 8B model asked to repeat a message verbatim mostly complies,
which makes it a poor witness: a masking regression would show up as the model
paraphrasing, which is indistinguishable from the model being a model.

So this returns the request's messages verbatim as its completion content. An
assertion against that is an assertion about the wire, and it fails loudly if
the guardrail ever stops mapping its edits back onto the request.

Standard library only; it runs on the python:3.11-slim image with no build.
"""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 8099


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")

        # The whole point: hand back exactly what arrived.
        echoed = json.dumps(
            {"messages": body.get("messages", []), "tools": body.get("tools")},
            ensure_ascii=False,
        )
        self._send(
            {
                "id": "echo-1",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", "echo"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": echoed},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        )

    def do_GET(self) -> None:
        self._send({"status": "ok"})

    def _send(self, payload: dict[str, object]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt: str, *args: object) -> None:
        """Silence the access log.

        It would contain no bodies, but a fixture that prints request lines
        next to a PII test is a habit worth not forming.
        """


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()  # noqa: S104
