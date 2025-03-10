from langchain_openai import ChatOpenAI
from langchain.agents import load_tools, AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

store = {}


def get_by_session_id(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


input_variables = ["agent_scratchpad", "input", "tool_names", "tools"]
template = """\
Answer the following questions as best you can. YUou have access to the following tools:
{tools}

use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Previous conversation history: {chat_history}
Question: {input}
Thought:{agent_scratchpad}
"""

prompt = PromptTemplate(input_variables=input_variables, template=template)
model = ChatOpenAI(model="gpt-4o-mini")
tools = load_tools(["serpapi"], llm=model)
agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_by_session_id,
    input_messages_key="input",
    history_messages_key="chat_history",
)

response = agent_with_chat_history.invoke(
    {
        "input": "株式会社Elithの住所を教えてください。最新の公式情報として公開されているものを教えてください。"
    },
    config={"configurable": {"session_id": "test-session1"}},
)
response = agent_with_chat_history.invoke(
    {"input": "先ほど尋ねた会社はなんの会社でしょうか？"},
    config={"configurable": {"session_id": "test-session1"}},
)
print(response)

print(get_by_session_id("test-session1"))
