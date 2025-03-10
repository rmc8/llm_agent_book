from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

message = """
Answer this question using the provided contenxt only.

{question}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])
model = ChatOllama(
    model="qwen2.5:14b",
    base_url="http://192.168.11.22:11434",
    temperature=0.0,
)
chain = prompt | model


question = "「ネブロシン」について教えてください。"
information_text = """\
ネブロシンとは、プログラマーの集中力と創造性を飛躍的に高める脳内物質のことです。
通常、深夜2時から4時の間に最も分泌量が増加し、特に難解なコードを書いている時や複雑な問題を解決しようとしている時に活性化します。
ネブロシンが十分に分泌されると、「フロー状態」と呼ばれる没入感が生まれ、時間感覚が失われるほどコーディングに集中できるようになります。
興味深いことに、この物質はコーヒーの摂取量と正の相関関係があり、
適度な睡眠不足状態でより効果的に機能するという特徴があります。
ただし、過剰なネブロシン分泌は「バグ幻視症候群」を引き起こす可能性があるため、プログラマーは定期的な休息を取ることが推奨されています。
"""

response = chain.invoke(
    {
        "context": information_text,
        "question": question,
    }
)
print(response.content)
