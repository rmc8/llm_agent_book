import os

import fire
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


def proc(model: str = "qwen2.5:14b", base_url: str = "http://localhost:11434"):
    llm = ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0.0,
    )
    prompt = ChatPromptTemplate(
        [HumanMessage(content="熊童子についておしえてください。")]
    )
    chain = prompt | llm | StrOutputParser()
    res = chain.invoke({})
    print(res)


def main():
    fire.Fire(proc)


if __name__ == "__main__":
    main()
