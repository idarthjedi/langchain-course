import asyncio
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
#from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv()

llm = ChatOllama(model="gpt-oss:120b")

async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": [
                    f"{PROJECT_ROOT.as_posix()}/servers/math_server.py"
                ],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            },
        }
    )
    tools = await client.get_tools()
    agent = create_agent(llm, tools)
    # result = await agent.ainvoke({"messages": "What is 2 + 2?"})
    result = await agent.ainvoke(
        {"messages": "What is the weather in San Francisco?"}
    )

    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())