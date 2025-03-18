import functools
from typing import Annotated
from typing_extensions import TypedDict

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

kenta = functools.partial(agent_with_persona, name="kenta", traits=kenta_traits)
mari = functools.partial(agent_with_persona, name="mari", traits=mari_traits)
yuta = functools.partial(agent_with_persona, name="yuta", traits=yuta_traits)


graph_builder = StateGraph(State)
graph_builder.add_node("kenta", kenta)
graph_builder.add_node("mari", mari)
graph_builder.add_node("yuta", yuta)

graph_builder.add_edge(START, "kenta")
graph_builder.add_edge(START, "mari")
graph_builder.add_edge(START, "yuta")
graph_builder.add_edge("kenta", END)
graph_builder.add_edge("mari", END)
graph_builder.add_edge("yuta", END)

graph = graph_builder.compile()

human_message = HumanMessage("休日の過ごし方について、建設的に議論してください。")
for event in graph.stream({"messages": [human_message]}):
    for value in event.values():
        value["messages"][-1].pretty_print()
