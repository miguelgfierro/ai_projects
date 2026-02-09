from dotenv import load_dotenv

load_dotenv(dotenv_path="environment")

import os
from openai import OpenAI


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_response(user_input):
    # Each call is independent, no conversation memory
    response = client.responses.create(
        model="gpt-4o-mini",  # Models: https://platform.openai.com/docs/models
        instructions="You are a helpful assistant.",
        input=user_input,
        temperature=0.7,  # The temperature can range from 0 to 2.
        top_p=1.0,
    )
    return response


def main():
    welcome_message = "Hello! I'm your chatbot. Ask me anything, and I'll do my best to help you."
    print(f"Chatbot: {welcome_message}")

    while True:
        # Get user input
        print("You: ", end="")
        user_input = input()

        # Check if the conversation is complete
        if any(exit_keyword in user_input.lower() for exit_keyword in ["exit", "quit", "bye", "goodbye"]):
            print("Chatbot: Goodbye! Chat session is over.")
            break

        # Generate and display the bot's response
        response = generate_response(user_input)
        print(f"Chatbot: {response.output_text}")


if __name__ == "__main__":
    main()
