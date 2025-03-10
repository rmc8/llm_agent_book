import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_community.agent_toolkits.load_tools import load_tools

load_dotenv()

question = "株式会社Elithの住所を教えてください。最新の公式情報として公開されているものを教えてください。"
model = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
tools = load_tools(["serpapi"], llm=model)
model_with_tools = model.bind_tools(tools)
result = model_with_tools.invoke([HumanMessage(content=question)])
search_tool = tools[0]
r = search_tool.invoke(result.tool_calls[0]["args"])

print(result.content)
print(result.tool_calls)
print(r)
