import json

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Demo Tools")


@mcp.tool()
def get_weather(city: str) -> str:
    """Get current weather information for a city.

    Args:
        city: Name of the city to get weather for.
    """
    weather_data = {
        "new york": {"temp": 22, "condition": "Sunny", "humidity": 45},
        "london": {"temp": 15, "condition": "Cloudy", "humidity": 72},
        "tokyo": {"temp": 28, "condition": "Partly cloudy", "humidity": 60},
        "paris": {"temp": 18, "condition": "Rainy", "humidity": 80},
        "sydney": {"temp": 25, "condition": "Clear", "humidity": 55},
        "madrid": {"temp": 30, "condition": "Sunny", "humidity": 35},
    }
    data = weather_data.get(
        city.lower(), {"temp": 20, "condition": "Unknown", "humidity": 50}
    )
    return json.dumps({
        "city": city,
        "temperature_celsius": data["temp"],
        "condition": data["condition"],
        "humidity_percent": data["humidity"],
    })


@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate (e.g., '2 + 3 * 4').
    """
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return json.dumps({"error": "Invalid characters in expression"})
    try:
        result = eval(expression)
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    mcp.run()
