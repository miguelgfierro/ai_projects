from dotenv import load_dotenv

load_dotenv(dotenv_path="environment")

import os
from openai import OpenAI
import gradio as gr


# Set your OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Initialize an empty conversation history
messages = []

# Add a system input
system_message = "You are a helpful assistant."
messages.append({"role": "system", "content": system_message})


# Define the OpenAI chat function
def openai_chat(user_input):
    messages.append({"role": "user", "content": user_input})

    # Call OpenAI's Chat API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )

    assistant_reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_reply})

    # Return the assistant's reply
    return assistant_reply


# Create the Gradio interface
iface = gr.Interface(
    fn=openai_chat,
    inputs="text",
    outputs="text",
    live=True,
    title="OpenAI Chat",
    description="Type a message and click Send to chat with the assistant.",
)


# Launch the Gradio app
iface.launch(width=600)
