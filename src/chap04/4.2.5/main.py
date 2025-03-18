import json
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_core.messages import ToolMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]
    count: int


class ToolNode:
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, state: State):
        if messages := state.get("messages", []):
            messages = messages[-1]
        else:
            raise ValueError("入力にメッセージが見つかりません")
        tool_messages = []
        for tool_call in messages.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            tool_messages.append(
                ToolMessage(
                    content=json.dumps(tool_result, ensure_ascii=False),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {
            "messages": tool_messages,
            "count": state.get("count") + 1,
        }


def chatbot(state: State) -> State:
    messages = [llm_with_tool.invoke(state["messages"])]
    return {
        "messages": messages,
        "count": state.get("count") + 1,
    }


def route_tools(state: State) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    ai_message = messages[-1]
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"


tavily_tool = TavilySearchResults(max_results=2)
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tool = llm.bind_tools([tavily_tool])
tool_node = ToolNode([tavily_tool])

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    ["tools", "__end__"],
)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("tools", "chatbot")
graph = graph_builder.compile()

human_message = {
    "messages": [HumanMessage("今日の東京の天気を教えてください。")],
    "count": 0,
}
for event in graph.stream(human_message):
    for value in event.values():
        value["messages"][-1].pretty_print()
