import functools
from typing import Annotated, Literal
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain.prompts import SystemMessagePromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


llm = ChatOpenAI(model="gpt-4o-mini")


class State(TypedDict):
    messages: Annotated[list, add_messages]
    next: str


class RouteSchema(BaseModel):
    next: Literal["kenta", "mari", "yuta"] = Field(..., description="次に発言する人")


def agent_with_persona(state: State, name: str, traits: str):
    system_message_template = SystemMessagePromptTemplate.from_template(
        "あなたの名前は{name}です。\nあなたの性格は以下の通りです。\n\n{traits}"
    )
    system_message = system_message_template.format(name=name, traits=traits)
    message = HumanMessage(
        content=llm.invoke([system_message, *state["messages"]]).content,
        name=name,
    )
    return {"messages": [message]}


def supervisor(state: State):
    system_message = SystemMessagePromptTemplate.from_template(
        "あなたは以下の作業者間の会話を監督する監督者です：\n{members}\n\n"
        "各メンバーの性格は以下の通りです：\n {traits_description}\n\n"
        "与えられたユーザーリクエストに対して、次に発言する人を選択してください。"
    )
    members = ", ".join(list(member_dict.keys()))
    traits_description = "\n".join(
        [f"**{name}**\n{traits}" for name, traits in member_dict.items()]
    )
    system_message = system_message.format(
        members=members, traits_description=traits_description
    )
    llm_with_format = llm.with_structured_output(RouteSchema)
    next_person = llm_with_format.invoke([system_message] + state["messages"]).next
    return {"next": next_person}


kenta_traits = """\
* アクティブで冒険好き
* 新しい経験を求める
* アウトドア活動を好む
* SNSでの共有を楽しむ
* エネルギッシュで社交的
"""

mari_traits = """\
* 穏やかでリラックス指向
* 家族を大切にする
* 静かな趣味を楽しむ
* 心身の休養を重視
* 丁寧な生活を好む
"""

yuta_traits = """\
* バランス重視
* 柔軟性がある
* 自己啓発に熱心
* 伝統と現代の融合を好む
* 多様な経験を求める
"""

member_dict = {
    "kenta": kenta_traits,
    "mari": mari_traits,
    "yuta": yuta_traits,
}


kenta = functools.partial(agent_with_persona, name="kenta", traits=kenta_traits)
mari = functools.partial(agent_with_persona, name="mari", traits=mari_traits)
yuta = functools.partial(agent_with_persona, name="yuta", traits=yuta_traits)


graph_builder = StateGraph(State)
graph_builder.add_node("supervisor", supervisor)
graph_builder.add_node("kenta", kenta)
graph_builder.add_node("mari", mari)
graph_builder.add_node("yuta", yuta)

graph_builder.add_edge(START, "supervisor")
graph_builder.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {"kenta": "kenta", "mari": "mari", "yuta": "yuta"},
)
for member in ["kenta", "mari", "yuta"]:
    graph_builder.add_edge(member, END)

graph = graph_builder.compile()

human_message = HumanMessage("休日のまったりした過ごし方を教えてください。")
for event in graph.stream({"messages": [human_message]}):
    for value in event.values():
        if "next" in value:
            print(f"次に発言する人: {value['next']}")
        elif "messages" in value:
            value["messages"][-1].pretty_print()
