from dotenv import load_dotenv

load_dotenv(dotenv_path="environment")

import os
import time
from openai import OpenAI


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create a new thread for each conversation
thread = client.beta.threads.create()


def submit_message(assistant_id, thread, user_message):
    client.beta.threads.messages.create(thread_id=thread.id, role="user", content=user_message)
    return client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )


def get_response(thread):
    return client.beta.threads.messages.list(thread_id=thread.id, order="desc")


def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.1)
    return run


def print_assistant_response(messages):
    """Pretty printing helper for the latest assistant response"""
    for m in messages:
        if m.role == "assistant":
            print(f"{m.role}: {m.content[0].text.value}")
            break


def decode_assistant_response(messages):
    response = ""
    for m in messages:
        if m.role == "assistant":
            response = m.content[0].text.value
            break
    return response


if __name__ == "__main__":

    exit_keywords = [
        "exit",
        "bye",
        "goodbye",
        "quit",
    ]

    while True:
        user_input = input("You: ")
        if any(keyword in user_input.lower() for keyword in exit_keywords):
            print("Conversation ended.")
            break

        # Submit the user's message to the existing thread
        run = submit_message(os.getenv("OPENAI_ASSISTANT_ID"), thread, user_input)

        # Wait for the assistant to respond
        run = wait_on_run(run, thread)

        # Display the assistant's response
        assistant_response = get_response(thread)
        print_assistant_response(assistant_response)
