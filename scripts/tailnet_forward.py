"""Expose the local Capture Studio server (127.0.0.1:8642) on this machine's Tailscale address only.

    python scripts/tailnet_forward.py            # http://<tailscale-ip>:8642/web/live/index.html
The server itself keeps listening on localhost; nothing is opened on the LAN or the public internet.
"""
import asyncio
import subprocess
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8642


async def pipe(r, w):
    try:
        while data := await r.read(65536):
            w.write(data)
            await w.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        w.close()


async def handle(cr, cw):
    try:
        sr, sw = await asyncio.open_connection("127.0.0.1", PORT)
    except OSError:
        cw.close()
        return
    await asyncio.gather(pipe(cr, sw), pipe(sr, cw))


async def main():
    ip = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True).stdout.split()[0]
    server = await asyncio.start_server(handle, ip, PORT)
    print(f"forwarding http://{ip}:{PORT}/ -> 127.0.0.1:{PORT}", flush=True)
    async with server:
        await server.serve_forever()


asyncio.run(main())
