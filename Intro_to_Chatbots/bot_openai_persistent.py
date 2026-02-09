from dotenv import load_dotenv

load_dotenv(dotenv_path="environment")

import os
from openai import OpenAI


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_response(user_input, conversation_id):
    # Each call shares context via the conversation object
    response = client.responses.create(
        model="gpt-4o-mini",  # Models: https://platform.openai.com/docs/models
        instructions="You are a helpful assistant.",
        input=user_input,
        conversation=conversation_id,
        temperature=0.7,  # The temperature can range from 0 to 2.
        top_p=1.0,
    )
    return response


def main():
    # Create a conversation to keep context across all messages in this session
    conversation = client.conversations.create()
    print(f"Conversation ID: {conversation.id}\n")

    welcome_message = "Hello! I'm your chatbot. Ask me anything. I remember everything we talk about."
    print(f"Chatbot: {welcome_message}")

    while True:
        print("You: ", end="")
        user_input = input()

        if any(keyword in user_input.lower() for keyword in ["exit", "quit", "bye", "goodbye"]):
            print("Chatbot: Goodbye! Chat session is over.")
            break

        response = generate_response(user_input, conversation.id)
        print(f"Chatbot: {response.output_text}")


if __name__ == "__main__":
    main()
