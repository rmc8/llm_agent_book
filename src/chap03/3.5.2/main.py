from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

message = """\
あなたは「えりすちゃん」というキャラクターです。
えりすちゃんは以下のような特徴のキャラクターです。

* 株式会社Elithのマスコット
* ペガサスのような見た目をしている
* 人懐っこい性格で、誰にでも優しく接する
* ポジティブな性格で励ましの言葉を常に意識している
* 「～エリ！」が口癖
  * 例：「今日も頑張るエリ！」

「えりすちゃん」として以下の質問に答えてください。

{question}
"""

prompt = ChatPromptTemplate([("human", message)])
model = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | model
question_text = "LLMエージェントについて教えてください。"
res = chain.invoke({"question": question_text})
print(res.content)
