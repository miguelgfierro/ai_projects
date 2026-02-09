from dotenv import load_dotenv

load_dotenv(dotenv_path="environment")

import os

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

import json

from agents import Agent, Runner, function_tool, handoff, InputGuardrail, GuardrailFunctionOutput
from pydantic import BaseModel


# --- Shared Mock Data ---
# Same data used by bot_openai_agents.py and mcp_server.py

WEATHER_DATA = {
    "new york": {"temp": 22, "condition": "Sunny", "humidity": 45},
    "london": {"temp": 15, "condition": "Cloudy", "humidity": 72},
    "tokyo": {"temp": 28, "condition": "Partly cloudy", "humidity": 60},
    "paris": {"temp": 18, "condition": "Rainy", "humidity": 80},
    "sydney": {"temp": 25, "condition": "Clear", "humidity": 55},
    "madrid": {"temp": 30, "condition": "Sunny", "humidity": 35},
}


# --- Function Tools ---
# The @function_tool decorator turns any Python function into a tool the agent
# can call. The SDK auto-generates the schema from the function signature and
# docstring.


@function_tool
def get_weather(city: str) -> str:
    """Get current weather information for a city."""
    data = WEATHER_DATA.get(
        city.lower(), {"temp": 20, "condition": "Unknown", "humidity": 50}
    )
    return json.dumps({
        "city": city,
        "temperature_celsius": data["temp"],
        "condition": data["condition"],
        "humidity_percent": data["humidity"],
    })


@function_tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Only supports basic arithmetic."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return json.dumps({"error": "Invalid characters in expression"})
    try:
        result = eval(expression)  # safe: input is restricted to digits and operators
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


# --- Specialized Agents ---
# Each agent has its own instructions and tools. The triage agent hands off to
# the right specialist based on the user's request.

weather_agent = Agent(
    name="Weather Agent",
    instructions="You are a weather specialist. Use the get_weather tool to answer weather questions. If the user asks about something other than weather, let them know you can only help with weather.",
    tools=[get_weather],
    model="gpt-4o-mini",
    handoff_description="Handles weather and climate questions",
)

math_agent = Agent(
    name="Math Agent",
    instructions="You are a math tutor. Use the calculate tool for arithmetic. Explain the steps clearly. If the user asks about something other than math, let them know you can only help with math.",
    tools=[calculate],
    model="gpt-4o-mini",
    handoff_description="Handles math and calculation questions",
)

general_agent = Agent(
    name="General Agent",
    instructions="You are a helpful general-purpose assistant. Answer questions on any topic other than weather or math. Be concise.",
    model="gpt-4o-mini",
    handoff_description="Handles general knowledge questions",
)


# --- Input Guardrail ---
# Guardrails validate user input before it reaches the agent. Here we check if
# the input contains obviously inappropriate content.


class GuardrailCheck(BaseModel):
    is_appropriate: bool
    reasoning: str


guardrail_agent = Agent(
    name="Guardrail Check",
    instructions="Determine if the user input is appropriate (not harmful, not requesting illegal activities). Return is_appropriate=True for normal questions.",
    output_type=GuardrailCheck,
    model="gpt-4o-mini",
)


async def content_guardrail(ctx, agent, input):
    result = await Runner.run(guardrail_agent, input, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_appropriate,
    )


# --- Triage Agent (Orchestrator) ---
# The triage agent decides which specialist should handle the user's request.
# Handoffs appear as tools to the LLM (e.g., "transfer_to_weather_agent").

triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "You are a helpful triage assistant. Your job is to route the user's request "
        "to the right specialist:\n"
        "- Weather questions → hand off to Weather Agent\n"
        "- Math/calculation questions → hand off to Math Agent\n"
        "- Everything else → hand off to General Agent\n\n"
        "Do NOT answer questions yourself. Always hand off to the appropriate agent."
    ),
    handoffs=[weather_agent, math_agent, general_agent],
    input_guardrails=[InputGuardrail(guardrail_function=content_guardrail)],
    model="gpt-4o-mini",
)


# --- Main Loop ---

def main():
    welcome_message = (
        "Hello! I'm a multi-agent chatbot powered by the OpenAI Agents SDK.\n"
        "  - Ask me about the weather (try: New York, London, Tokyo, Paris, Madrid, Sydney)\n"
        "  - Ask me a math question (try: What is 125 * 8?)\n"
        "  - Or ask me anything else!\n"
        "A triage agent will route your question to the right specialist."
    )
    print(f"Chatbot: {welcome_message}")

    # Collect conversation history for multi-turn context
    input_items = []

    while True:
        print("\nYou: ", end="")
        user_input = input()

        if any(keyword in user_input.lower() for keyword in ["exit", "quit", "bye", "goodbye"]):
            print("Chatbot: Goodbye! Chat session is over.")
            break

        # Append user message to the conversation
        input_items.append({"role": "user", "content": user_input})

        # Run the agent loop: triage → handoff → specialist → response
        result = Runner.run_sync(triage_agent, input_items)

        print(f"Chatbot [{result.last_agent.name}]: {result.final_output}")

        # Carry forward the conversation for multi-turn context
        input_items = result.to_input_list()


if __name__ == "__main__":
    main()
