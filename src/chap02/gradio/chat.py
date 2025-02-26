import fire
import gradio as gr
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, AIMessage


def proc(model="qwen2.5:14b", base_url="http://localhost:11434"):
    llm = ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0.0,
    )

    def history_to_messages(history):
        messages = []
        for user, assistant in history:
            messages.append(HumanMessage(content=user))
            messages.append(AIMessage(content=assistant))
        return messages

    def chat(message, history):
        messages = history_to_messages(history)
        messages.append(HumanMessage(content=message))
        response = llm.invoke(message)
        return response.content

    demo = gr.ChatInterface(chat)
    demo.launch()


fire.Fire(proc)
