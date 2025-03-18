import functools
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field


class JudgeScheme(BaseModel):
    judged: bool = Field(..., description="勝者が決まったかどうか")
    answer: str = Field(description="議題に対する結論とその理由")


class State(TypedDict):
    messages: Annotated[list, add_messages]
    debate_topic: str
    judged: bool
    round: int


llm = ChatOpenAI(model="gpt-4o-mini")


def cot_agent(state: State) -> State:
    system_message = (
        "与えられた議題に対してステップバイステップで考えてから応答してください。"
        "議題：{debate_topic}"
    )
    system_message = SystemMessage(
        system_message.format(debate_topic=state["debate_topic"])
    )
    message = HumanMessage(content=llm.invoke([system_message]).content, name="CoT")
    return {"messages": [message]}


def debater(
    state: State,
    name: str,
    position: str,
) -> State:
    system_message = (
        "あなたはディベーターです。ディベート大会へようこそ。"
        "私たちの目的は正しい答えを見つけることですので、お互いの視点に完全に同意する必要はありません。"
        "ディベートのテーマは以下の通りです：{debate_topic}\n"
        "{position}"
    )
    debate_topic = state["debate_topic"]
    system_message = SystemMessage(
        system_message.format(
            debate_topic=debate_topic,
            position=position,
        ),
    )
    message = HumanMessage(
        content=llm.invoke([system_message, *state["messages"]]).content,
        name=name,
    )
    return {"messages": [message]}


def judger(state: State):
    system_message = (
        "あなたは司会者です。"
        "ディベート大会に２名のディべーターが参加します。"
        "彼らは{debate_topic}について自分の応答を発表し、それぞれの視点について議論します。"
        "各ラウンドの終わりに、あなたは両社の応答を評価していき、ディベートの勝者を判断します。"
        "判定が難しい場合は、次のラウンドで判断してください。"
    )
    system_message = SystemMessage(
        system_message.format(debate_topic=state["debate_topic"])
    )
    llm_with_format = llm.with_structured_output(JudgeScheme)
    res: JudgeScheme = llm_with_format.invoke([system_message, *state["messages"]])
    messages = []
    if res.judged:
        message = HumanMessage(res.answer)
        messages.append(message)
    return {
        "messages": messages,
        "judged": res.judged,
    }


def round_monitor(state: State, max_round: int):
    round_ = state["round"] + 1
    if state["round"] < max_round:
        return {"round": round_}
    return {
        "messages": [
            HumanMessage(
                "最終ラウンドなので勝者を決定し、議論に対する結論とその理由を述べてください。"
            )
        ],
        "round": round_,
    }


affirmative_debater = functools.partial(
    debater,
    name="Affirmative_Debater",
    position="あなたは肯定側です。あなたの見解を簡潔に述べてください。否定側の意見が与えられた場合はそれに反対して理由を簡潔に述べてください。",
)
negative_debater = functools.partial(
    debater,
    name="Negative_Debater",
    position="あなたは否定側です。肯定側の意見に反対し、あなたの理由を簡潔に述べてください。",
)
round_monitor = functools.partial(round_monitor, max_round=3)

graph_builder = StateGraph(State)
graph_builder.add_node("cot_agent", cot_agent)
graph_builder.add_node("affirmative_debater", affirmative_debater)
graph_builder.add_node("negative_debater", negative_debater)
graph_builder.add_node("judger", judger)
graph_builder.add_node("round_monitor", round_monitor)

graph_builder.add_edge(START, "cot_agent")
graph_builder.add_edge("cot_agent", "affirmative_debater")
graph_builder.add_edge("affirmative_debater", "negative_debater")
graph_builder.add_edge("negative_debater", "round_monitor")
graph_builder.add_edge("round_monitor", "judger")
graph_builder.add_conditional_edges(
    "judger",
    lambda state: state["judged"],
    {
        True: END,
        False: "affirmative_debater",
    },
)

graph = graph_builder.compile()


inputs = {
    "messages": [],
    "debate_topic": "戦争は必要か？",
    "judged": False,
    "round": 0,
}

for event in graph.stream(inputs):
    for value in event.values():
        try:
            value["messages"][-1].pretty_print()
        except Exception as _:
            pass
