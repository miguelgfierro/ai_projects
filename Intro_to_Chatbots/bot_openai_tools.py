from dotenv import load_dotenv

load_dotenv(dotenv_path="environment")

import os
from openai import OpenAI


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_response(user_input, conversation_id):
    # Persistent conversation with web search tool
    response = client.responses.create(
        model="gpt-4o-mini",  # Models: https://platform.openai.com/docs/models
        instructions="You are a helpful assistant. Use web search when the user asks about current events, recent news, or anything that requires up-to-date information.",
        input=user_input,
        conversation=conversation_id,
        tools=[{"type": "web_search_preview"}],
        temperature=0.7,
        top_p=1.0,
    )
    return response


def format_response(response):
    """Extract text and citations from the response."""
    text = response.output_text

    # Collect unique citations
    citations = []
    for item in response.output:
        if item.type == "message":
            for content in item.content:
                if hasattr(content, "annotations"):
                    for annotation in content.annotations:
                        if annotation.type == "url_citation" and annotation.url not in citations:
                            citations.append(annotation.url)

    if citations:
        text += "\n\nSources:"
        for url in citations:
            text += f"\n  - {url}"

    return text


def main():
    conversation = client.conversations.create()
    print(f"Conversation ID: {conversation.id}\n")

    welcome_message = "Hello! I'm your chatbot with web search. Ask me about current events, news, or anything else."
    print(f"Chatbot: {welcome_message}")

    while True:
        print("\nYou: ", end="")
        user_input = input()

        if any(keyword in user_input.lower() for keyword in ["exit", "quit", "bye", "goodbye"]):
            print("Chatbot: Goodbye! Chat session is over.")
            break

        response = generate_response(user_input, conversation.id)
        print(f"Chatbot: {format_response(response)}")


if __name__ == "__main__":
    main()
