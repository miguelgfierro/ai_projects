from dotenv import load_dotenv

load_dotenv(dotenv_path="environment")

import os
from openai import OpenAI


# Set your OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_INSTRUCTIONS = "You are a helpful assistant."


def generate_response(user_input, previous_response_id=None):
    # Generate a response from OpenAI Responses API
    response = client.responses.create(
        model="gpt-4o-mini",  # Models: https://platform.openai.com/docs/models/overview
        instructions=SYSTEM_INSTRUCTIONS,
        input=user_input,
        previous_response_id=previous_response_id,
        temperature=0.7,  # The temperature can range from 0 to 2.
        top_p=1.0,
    )
    return response


def main():
    welcome_message = "Hello! I'm your chatbot. Ask me anything, and I'll do my best to help you."
    print(f"Chatbot: {welcome_message}")

    previous_response_id = None

    while True:
        # Get user input
        print("You: ", end="")
        user_input = input()

        # Check if the conversation is complete
        if any(exit_keyword in user_input.lower() for exit_keyword in ["exit", "quit", "bye", "goodbye"]):
            print("Chatbot: Goodbye! Chat session is over.")
            break

        # Generate and display the bot's response
        response = generate_response(user_input, previous_response_id)
        bot_response = response.output_text
        print(f"Chatbot: {bot_response}")

        # Keep track of conversation state
        previous_response_id = response.id


if __name__ == "__main__":
    main()
