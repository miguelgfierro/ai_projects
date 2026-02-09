from dotenv import load_dotenv

load_dotenv(dotenv_path="environment")

import os
from openai import OpenAI
import gradio as gr

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def chat(user_input, history, previous_response_id):
    """Send a message and get a response using OpenAI Responses API."""
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions="You are a helpful assistant.",
        input=user_input,
        previous_response_id=previous_response_id,
        temperature=0.7,
        top_p=1.0,
    )

    assistant_reply = response.output_text
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": assistant_reply})

    return "", history, response.id


def clear_chat():
    """Reset the conversation."""
    return [], None


with gr.Blocks(title="OpenAI Chatbot") as app:
    gr.Markdown("# OpenAI Chatbot\nPowered by the Responses API")

    chatbot = gr.Chatbot(
        height=500,
        show_copy_button=True,
        placeholder="Start a conversation...",
    )
    previous_response_id = gr.State(None)

    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Type your message here...",
            show_label=False,
            scale=9,
            container=False,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)

    clear_btn = gr.Button("Clear Chat")

    send_btn.click(
        fn=chat,
        inputs=[user_input, chatbot, previous_response_id],
        outputs=[user_input, chatbot, previous_response_id],
    )
    user_input.submit(
        fn=chat,
        inputs=[user_input, chatbot, previous_response_id],
        outputs=[user_input, chatbot, previous_response_id],
    )
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, previous_response_id],
    )

app.launch(theme=gr.themes.Soft())
