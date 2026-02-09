# A Gentle Introduction to ChatBots

* `bot_openai_basic.py` - Stateless chatbot using the OpenAI Responses API. Each message is independent with no conversation memory.
* `bot_openai_persistent.py` - Stateful chatbot using the OpenAI Conversations API. The model remembers all previous messages in the session.
* `bot_openai_tools.py` - Stateful chatbot with web search tool. The model can search the internet for current information and cites its sources.
* `bot_openai_mcp.py` + `mcp_server.py` - Stateful chatbot with custom tools served via [MCP](https://modelcontextprotocol.io/) (Model Context Protocol). The client discovers tools from the MCP server at runtime and bridges them with OpenAI function calling.
* `bot_openai_agents.py` - Multi-agent system using the OpenAI Agents SDK. Demonstrates triage routing, handoffs, function tools, and input guardrails.
* `bot_openai_gradio.py` - Chat interface with Gradio.
* `bot_langchain_rag.ipynb` - RAG with LangChain and FAISS over a text document.

## Installation

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```
