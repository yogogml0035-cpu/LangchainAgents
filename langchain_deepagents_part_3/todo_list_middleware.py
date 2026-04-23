from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_deepseek import ChatDeepSeek

load_dotenv()

model = ChatDeepSeek(
    model="deepseek-chat",
)

agent = create_agent(
    model=model,
    tools=[],
    middleware=[TodoListMiddleware()],
)

res = agent.invoke({"messages":["""你要一步一步的详细规划以下内容再进行回答。
请分析美国加利福尼亚中央谷地的杏仁种植业在未来30年面临的气候变化风险,并估算其经济影响。
具体需要回答,:
“假设当前气候趋势持续,到2050年,加利福尼亚中央谷地杏仁产量可能减少的百分比及其对该州经
济的潜在年度损失是多少美元?这些美元按照2025年11月的汇率能够购买多少比特币?
"""]})

