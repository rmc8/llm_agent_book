from langchain_openai import ChatOpenAI
from langchain.agents import load_tools
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.agents import AgentExecutor, create_react_agent

load_dotenv()

prompt = hub.pull("hwchase17/react")
model = ChatOpenAI(model="gpt-4o-mini")
tools = load_tools(["serpapi"], llm=model)
agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
msg = HumanMessage(
    content="株式会社Elithの住所を教えてください。最新の公式情報として公開されているものを教えてください。"
)
response = agent_executor.invoke({"input": [msg]})
print(response)
