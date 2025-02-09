from dotenv import load_dotenv

load_dotenv(dotenv_path="environment")

import os
from openai import OpenAI


# Set your OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def setup_message(welcome_message):
    system_message = "You are a helpful assistant."
    messages = [
        {"role": "system", "content": system_message},
        {"role": "assistant", "content": welcome_message},
    ]
    return messages


def format_response(response):
    return response.choices[0].message.content


def generate_response(message_history):
    # Generate a response from OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Models: https://platform.openai.com/docs/models/overview
        messages=message_history,
        temperature=0.7,  # The temperature can range from 0 to 2.
        # max_tokens=150,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    return response


def main():
    welcome_message = "Hello! I'm your chatbot. Ask me anything, and I'll do my best to help you."
    print(f"Chatbot: {welcome_message}")

    message_history = setup_message(welcome_message)

    while True:
        # Get user input
        print("You: ", end="")
        user_input = input()
        message_history.append({"role": "user", "content": user_input})

        # Check if the conversation is complete
        if any(exit_keyword in user_input.lower() for exit_keyword in ["exit", "quit", "bye", "goodbye"]):
            print("Chatbot: Goodbye! Chat session is over.")
            break

        # Generate and display the bot's response
        response = generate_response(message_history)
        bot_response = format_response(response)
        print(f"Chatbot: {bot_response}")

        # Add the bot's response to the chat history
        message_history.append({"role": "assistant", "content": bot_response})


if __name__ == "__main__":
    main()
