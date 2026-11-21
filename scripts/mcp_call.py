"""Call Capture Studio MCP tools from the shell (what an agent does, one call at a time).

    python scripts/mcp_call.py --list
    python scripts/mcp_call.py get_model
    python scripts/mcp_call.py add_wall '{"start":[4,0],"end":[4,3]}'
    python scripts/mcp_call.py validate  add_space '{...}'  validate      # several calls, in order
"""
import asyncio
import json
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

import os
URL = os.environ.get("CAPTURE_STUDIO_MCP", "http://127.0.0.1:8642/mcp")


async def main(argv):
    async with streamablehttp_client(URL) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            if argv and argv[0] == "--list":
                for t in (await s.list_tools()).tools:
                    print(f"{t.name:22s} {(t.description or '').splitlines()[0]}")
                return
            i = 0
            while i < len(argv):
                name, args = argv[i], {}
                if i + 1 < len(argv) and argv[i + 1].lstrip().startswith("{"):
                    args = json.loads(argv[i + 1])
                    i += 1
                i += 1
                res = await s.call_tool(name, args)
                text = "".join(c.text for c in res.content if hasattr(c, "text"))
                try:
                    text = json.dumps(json.loads(text), indent=1 if "--pretty" in sys.argv else None)
                except ValueError:
                    pass
                print(f"── {name} {'(error)' if res.isError else ''}\n{text}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
