import os
import re
import tomllib
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_experimental.utilities import PythonREPL
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages


repl = PythonREPL()
llm = ChatOpenAI(model="gpt-4o", model_kwargs={"temperature": 0})
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(THIS_DIR, "config.toml"), "rb") as f:
    settings = tomllib.load(f)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    problem: str
    first_flag: bool
    end_flag: bool


def extract_code(input_string: str):
    ptn = r"```(.*?)```"
    match = re.findall(ptn, input_string, flags=re.DOTALL)
    queries = ""
    for m in match:
        query = m.replace("python", "").strip()
        queries += query + "\n"
    return queries.strip()


def user_proxy_agent(state: State):
    if state["first_flag"]:
        message = settings["INITIAL_PROMPT"].format(problem=state["problem"])
    else:
        last_message = state["messages"][-1].content
        code = extract_code(last_message)
        if code:
            message = repl.run(code)
        else:
            message = "続けてください。クエリが必要になるまで問題を解き続けてください（答えが出た場合は、\\boxed{{}}にいれてください）。"
    message = HumanMessage(message)
    return {
        "messages": [message],
        "first_flag": False,
    }


def extract_boxed(input_string: str):
    ptn = r"\\boxed\{[^\}]*\}"
    matches = re.findall(ptn, input_string)
    return [m.replace("\\boxed{", "").replace("}", "") for m in matches]


def llm_agent(state: State):
    message = llm.invoke(state["messages"])
    content = message.content
    boxed = extract_boxed(content)
    return {
        "messages": [message],
        "end_flag": True if boxed else False,
    }


graph_builder = StateGraph(State)
graph_builder.add_node("llm_agent", llm_agent)
graph_builder.add_node("user_proxy_agent", user_proxy_agent)
graph_builder.add_edge(START, "user_proxy_agent")
graph_builder.add_conditional_edges(
    "llm_agent",
    lambda state: state["end_flag"],
    {
        True: END,
        False: "user_proxy_agent",
    },
)
graph_builder.add_edge("user_proxy_agent", "llm_agent")

graph = graph_builder.compile()
problem = settings["problem"]
for event in graph.stream({"problem": problem, "first_flag": True}):
    for value in event.values():
        value["messages"][-1].pretty_print()
