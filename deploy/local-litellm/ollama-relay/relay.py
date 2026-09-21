"""TCP relay from a Docker bridge address to Ollama on the host's loopback.

Ollama listens on 127.0.0.1:11434, which a container cannot reach: resolving
`host.docker.internal` gets it to the host's bridge address and no further.

The alternative is setting OLLAMA_HOST=0.0.0.0 on the host's ollama service,
which publishes a model server on every interface the machine has in order to
fix a container-networking problem. This relay is the smaller change: it runs
with network_mode: host, binds one Docker bridge address and forwards to
loopback, so nothing new is reachable from the LAN and no system service is
touched.

Only the local stand-in stack needs it. A real deployment points LiteLLM at
vLLM on the GPU host over the network and has nothing to relay.

Standard library only, so it runs on a base image that is already present --
which matters on a host whose containers have no working DNS for `apk add`.
"""

from __future__ import annotations

import asyncio
import os

BIND = os.environ.get("RELAY_BIND", "172.17.0.1")
PORT = int(os.environ.get("RELAY_PORT", "11435"))
TARGET_HOST = os.environ.get("RELAY_TARGET_HOST", "127.0.0.1")
TARGET_PORT = int(os.environ.get("RELAY_TARGET_PORT", "11434"))


async def _pump(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while chunk := await reader.read(65536):
            writer.write(chunk)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        writer.close()


async def _handle(client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter) -> None:
    try:
        upstream_reader, upstream_writer = await asyncio.open_connection(TARGET_HOST, TARGET_PORT)
    except OSError:
        client_writer.close()
        return
    # Both directions, and the first to finish does not cancel the other:
    # streamed completions hold the downstream leg open long after the request
    # body is done.
    await asyncio.gather(
        _pump(client_reader, upstream_writer),
        _pump(upstream_reader, client_writer),
    )


async def main() -> None:
    server = await asyncio.start_server(_handle, BIND, PORT)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
