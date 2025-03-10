from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


message = """\
以下の質問に答えてください。

{question}
"""

prompt = ChatPromptTemplate([("human", message)])
model = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | model
question_text = "LLMエージェントについて教えてください。"
res = chain.invoke({"question": question_text})
print(res.content)
