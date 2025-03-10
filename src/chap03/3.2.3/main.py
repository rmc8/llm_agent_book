import os
import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()


def get_date(date) -> str:
    date_now = datetime.now(ZoneInfo("Asia/Tokyo"))
    if "今日" in date:
        date_delta = 0
    elif "明日" in date:
        date_delta = 1
    elif "明後日" in date:
        date_delta = 2
    else:
        return "サポートしていません"
    return (date_now + timedelta(days=date_delta)).strftime("m%月%日")


def get_fortune(date_string) -> str:
    try:
        date = datetime.strptime(date_string, "%m月%d日")
    except ValueError:
        return "無効な日付形式です'X月X日'の形式で入力してください。"
    fortunes = ["大吉", "中吉", "小吉", "吉", "末吉", "凶", "大凶"]
    weights = [1, 3, 3, 4, 3, 2, 1]
    random.seed(date.month * 100 + date.day)
    fortune = random.choices(fortunes, weights=weights)[0]
    return f"{date_string}の運勢は【{fortune}】です。"


class GetDate(BaseTool):
    name: str = "Get_date"
    description: str = (
        "今日の日付を取得する。インプットは'date'です。'date'は日付を取得する対象の日で、'今日','明日','明後日'という3種類の文字列から指定します「今日」のように入力し、「'今日'」のように余計な文字をつけてはいけません。"
    )

    def _run(self, date) -> str:
        return get_date(date)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Does not support async")


class GetFortune(BaseTool):
    name: str = "Get_fortune"
    description: str = (
        "特定の日付の運勢を占う。インプットは'date_string'です。'date_string'は占う日付で、mm月dd日という形式です。「1月1日」のように入力し「'1月1日'」のように余計な文字を付けてはいけません。"
    )

    def _run(self, date_string) -> str:
        return get_fortune(date_string)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Dose not support async")


tools = [GetDate(), GetFortune()]
model = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)
model_with_tools = model.bind_tools(tools)
question = "今日の運勢を教えてください。"
response = model_with_tools.invoke([HumanMessage(content=question)])

print(response.content)
print(response.tool_calls)
