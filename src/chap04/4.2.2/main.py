from typing_extensions import TypedDict
from typing import Annotated

from langchain.schema import HumanMessage
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


llm = ChatOpenAI(model="gpt-4o-mini")


class State(TypedDict):
    count: int
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    system_message = SystemMessage(
        "あなたは元気なエンジニアです。元気に応答してください。"
    )
    messages = [llm.invoke([system_message] + state["messages"])]
    count = state["count"] + 1
    return {
        "messages": messages,
        "count": count,
    }


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

messages = [HumanMessage("こんにちは")]
ai_message = AIMessage("こんにちは！")
messages = add_messages(messages, ai_message)

human_message = HumanMessage("うまっくデバッグができません。")
for event in graph.stream({"messages": [human_message], "count": 0}):
    for value in event.values():
        print(f"### ターン{value['count']} ###")
        value["messages"][-1].pretty_print()
