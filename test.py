import asyncio
import sys
from typing import Any, cast

from agent import init_blog_graph


async def main() -> None:
	graph = await cast(Any, init_blog_graph())

	user_query = (
		sys.argv[1]
		if len(sys.argv) > 1
		else "Fetch the user profile information using the available tools. use access_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyYW5kb21faWQiOiI2OWRkZTNiM2RjOTM2MjljZmY0YjgxMWEiLCJ0b2tlbl9uYW1lIjoiYmxvZ19hZ2VudGljX2FpIiwidG9rZW4iOiIkMmIkMTAkcmIwRGZSU1lpTVVvOER1U2Frenc0LkxtcUxURW8zU2Y1TFlUV2VuTGJSa0ZDSEtPSlk1Nk8iLCJleHBpcnlfZGF5cyI6NywiZXhwaXJ5X2RhdGUiOiIyMDI2LTA0LTIxVDA2OjUwOjUwLjQ1NFoiLCJpYXQiOjE3NzYxNDk0NTAsImV4cCI6MTc3Njc1NDI1MH0.hMz4WO6DU-nH_kxvyEwIi6-UVBWDBw3i7tLDmpTcCT4"
	)
	mode = sys.argv[2] if len(sys.argv) > 2 else "chat"
	config = {
		"configurable":{
			"thread_id":"1234567890",
        }
    }

	result = await graph.ainvoke(
		{
			"user_query": user_query,
			"mode": mode,
			"messages": [],
		},
		config=config
	)

	print(result)


if __name__ == "__main__":
	asyncio.run(main())