import os
from typing import List

import gradio as gr
from gradio import ChatMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    MessagesPlaceholder,
)
from langchain_openai import ChatOpenAI
from langchain_openai.output_parsers.tools import PydanticToolsParser
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()


class ActionItem(BaseModel):
    action_name: str = Field(description="Name of the action item")
    action_description: str = Field(description="Description of the action item")


class Plan(BaseModel):
    problem: str = Field(description="Description of the problem to be solved")
    actions: List[ActionItem] = Field(
        description="List of action items that need to be performed to solve the problem"
    )


class ActionResult(BaseModel):
    thoughts: str = Field(description="Thoughts about the action item")
    result: str = Field(description="Result of the action item")


ACTION_PROMPT = """\
問題をアクションプランに分解して解いています。
これまでのアクションの結果と、次に行うべきアクションを示すので、実際にアクションを実行してその結果を報告してください。

# 問題
{problem}

# アクションプラン
{action_items}

# これまでのアクションの結果
{action_results}

# 次のアクション
{next_action}
"""

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)
llm_action = llm.bind_tools([ActionResult], tool_choice="ActionResult")
action_parser = PydanticToolsParser(tools=[ActionResult], first_tool_only=True)
action_prompt = PromptTemplate.from_template(ACTION_PROMPT)
action_runnable = action_prompt | llm_action | action_parser
llm_plan = llm.bind_tools(tools=[Plan])
chat_prompt = ChatPromptTemplate(
    [
        AIMessage(
            content="ユーザーの質問が複雑な場合は、アクションプランを作成し、そのあとにひとつずつ実行するPlan-and-Solve形式となります。これが必要と判断した場合は、Planツールによってアクションプランを保存してください。"
        ),
        MessagesPlaceholder(variable_name="history"),
    ]
)
planning_runnable = chat_prompt | llm_plan
plan_parser = PydanticToolsParser(
    tools=[Plan],
    first_tool_only=True,
)


def action_loop(action_plan: Plan):
    problem = action_plan.problem
    actions = action_plan.actions

    action_items = "\n".join([f"* {action.action_name}" for action in actions])
    action_results = []
    action_results_str = ""
    for action in actions:
        next_action = f"* {action.action_name} \n{action.action_description}"
        response: ActionResult = action_runnable.invoke(
            {
                "problem": problem,
                "action_items": action_items,
                "action_results": action_results_str,
                "next_action": next_action,
            }
        )
        action_results.append(response)
        action_results_str += f"{action.action_name}\t\n{response.result}\n"
        yield (response.thoughts, response.result)


def chat(prompt, messages, history):
    messages.append(ChatMessage(role="user", content=prompt))
    history.append(HumanMessage(content=prompt))
    response = planning_runnable.invoke({"history": history})
    if response.response_metadata["finish_reason"] != "tool_calls":
        messages.append(("assistant", response.content))
        history.append(("assistant", response.content))
        yield "", messages, history
    else:
        action_plan: Plan = plan_parser.invoke(response)
        action_items = "\n".join(
            [f"* {action.action_name}" for action in action_plan.actions]
        )
        messages.append(
            ChatMessage(
                role="assistant",
                content=action_items,
                metadata={"title": "Action Items"},
            )
        )
        yield "", messages, history

        action_results_str = ""
        for i, (thoughts, result) in enumerate(action_loop(action_plan)):
            action_name = action_plan.actions[i].action_name
            action_results_str += f"* {action_name} \n{result}\n"
            text = f"## {action_name}\n### 思考過程\n{thoughts}\n### 結果\n{result}"
            messages.append(
                ChatMessage(
                    role="assistant",
                    content=text,
                )
            )
            yield "", messages, history
        history.append(("assistant", action_results_str))
        yield "", messages, history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="assistant", type="messages", height=700)
    history = gr.State([])
    with gr.Row():
        with gr.Column(scale=9):
            user_input = gr.Textbox(lines=1, label="Chat Message")
        with gr.Column(scale=1):
            submit = gr.Button("Submit")
            clear = gr.ClearButton([user_input, chatbot, history])
        submit.click(
            chat,
            inputs=[user_input, chatbot, history],
            outputs=[user_input, chatbot, history],
        )
demo.launch()
