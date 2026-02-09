from dotenv import load_dotenv

load_dotenv(dotenv_path="environment")

import asyncio
import json
import os
import sys

from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def mcp_to_openai_tools(mcp_tools):
    """Convert MCP tool definitions to OpenAI function tool format."""
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema,
        }
        for tool in mcp_tools.tools
    ]


async def handle_tool_calls(response, session, openai_tools):
    """Execute MCP tool calls requested by the model and return the final response."""
    function_calls = [item for item in response.output if item.type == "function_call"]

    while function_calls:
        tool_outputs = []
        for fc in function_calls:
            result = await session.call_tool(fc.name, json.loads(fc.arguments))
            tool_outputs.append({
                "type": "function_call_output",
                "call_id": fc.call_id,
                "output": result.content[0].text,
            })

        response = client.responses.create(
            model="gpt-4o-mini",
            instructions="You are a helpful assistant. Use the available tools when relevant.",
            input=tool_outputs,
            tools=openai_tools,
            previous_response_id=response.id,
            temperature=0.7,
            top_p=1.0,
        )
        function_calls = [
            item for item in response.output if item.type == "function_call"
        ]

    return response


async def main():
    # Connect to the MCP server via stdio (spawns it as a subprocess)
    server_params = StdioServerParameters(
        command=sys.executable, args=["mcp_server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Discover tools from MCP server and convert to OpenAI format
            mcp_tools = await session.list_tools()
            openai_tools = mcp_to_openai_tools(mcp_tools)
            tool_names = [t.name for t in mcp_tools.tools]
            print(f"MCP tools available: {tool_names}\n")

            welcome_message = "Hello! I'm your chatbot with MCP tools. Ask me about the weather or to do calculations."
            print(f"Chatbot: {welcome_message}")

            previous_response_id = None

            while True:
                print("\nYou: ", end="")
                user_input = input()

                if any(
                    kw in user_input.lower()
                    for kw in ["exit", "quit", "bye", "goodbye"]
                ):
                    print("Chatbot: Goodbye! Chat session is over.")
                    break

                response = client.responses.create(
                    model="gpt-4o-mini",
                    instructions="You are a helpful assistant. Use the available tools when relevant.",
                    input=user_input,
                    tools=openai_tools,
                    previous_response_id=previous_response_id,
                    temperature=0.7,
                    top_p=1.0,
                )

                # If the model requested tool calls, execute them via MCP
                response = await handle_tool_calls(response, session, openai_tools)
                print(f"Chatbot: {response.output_text}")
                previous_response_id = response.id


if __name__ == "__main__":
    asyncio.run(main())
