
#  attach your MCP servers here, following the example below. You can have as many servers as you want, and they can be local or remote.

SERVERS= {
        # "expenses_manager": {
        # "transport": "stdio",
        # "command": "uv",
        # "args": [
        #     "run",
        #     "fastmcp",
        #     "run",
        #     "O:/MCP-tutorial/main.py"
        #     ]
        # },
        # "feather_fables": {
        #     "transport": "streamable_http",
        #     "url": "https://feather-fables-mcp.fastmcp.app/mcp",
        #     "headers": {
        #         "Authorization": "Bearer fmcp_jrM_ztT1u30UAXroQ37SNqXVNpypb6sU8LjutvMbCJo"
        #     }
        # }
        "feather_fables": {
        "transport": "stdio",
        "command": "uv",
        "args": [
            "run",
            "fastmcp",
            "run",
            "O:/feather-fables-mcp/server.py"
            ]
        },
        
    }
