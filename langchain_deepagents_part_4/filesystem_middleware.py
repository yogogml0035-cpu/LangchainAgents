from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from deepagents import FilesystemMiddleware
from deepagents.backends import FilesystemBackend, StateBackend, StoreBackend, CompositeBackend
from langchain.messages import HumanMessage
from langgraph.store.memory import InMemoryStore

load_dotenv()

model = ChatDeepSeek(
    model="deepseek-chat",
)

'''
agent_local = create_agent(
    model=model,
    tools=[],
    middleware=[
        FilesystemMiddleware(
            backend=FilesystemBackend(root_dir="./test_dir", virtual_mode=True)
        )
    ]
)

res = agent_local.invoke(
    {
        'messages': [HumanMessage("调用工具写入一个文件，文件名为:测试.txt, 内容为: '测试'")]
    }
)

print(res)
'''

'''
agent_local = create_agent(
    model=model,
    tools=[],
    middleware=[
        FilesystemMiddleware(
            backend=lambda runtime: StateBackend(runtime)
        )
    ]
)

res = agent_local.invoke(
    {
        'messages': [
            HumanMessage("调用工具写入一个文件，文件名为:测试.txt, 内容为: '你好帅'"),
            HumanMessage('调用工具读取名为测试.txt的文件，告诉我里面的内容')
        ]
    },
)

print(res['messages'][-1].content)
'''

'''
store = InMemoryStore()
agent_local1 = create_agent(
    model=model,
    tools=[],
    store=store,
    middleware=[
        FilesystemMiddleware(
            backend=lambda runtime: StoreBackend(runtime)
        )
    ]
)

agent_local1.invoke({
    'messages':[
        HumanMessage("调用工具写入一个文件，文件名为:测试.txt, 内容为: '你好帅'"),
    ]
})

agent_local2 = create_agent(
    model=model,
    tools=[],
    store=store, # 同一个store实例
    middleware=[
        FilesystemMiddleware(
            backend=lambda runtime: StoreBackend(runtime)
        )
    ]
)

res = agent_local2.invoke(
    {
        'messages': [
            HumanMessage('调用工具读取名为测试.txt的文件，告诉我里面的内容')
        ]
    },
)

print(res['messages'][-1].content)
'''

store = InMemoryStore()

composite_backend = lambda runtime: CompositeBackend(
    default=StateBackend(runtime),
    routes={
        "/memories/": StoreBackend(runtime)
    }
)

agent = create_agent(
    model=model,
    store=store,
    middleware=[
        FilesystemMiddleware(
            backend=composite_backend
        )
    ]
)

config1 = {"configurable": {"thread_id": '1'}}

# 智能体将 "preferences.txt" 写入 /memories/ 路径
agent.invoke({
    "messages": [{"role": "user", "content": "我最爱的水果是草莓, 请把我的偏好保存在/memories/preferences.txt"}]
}, config=config1)

config2 = {"configurable": {"thread_id": '2'}}

res = agent.invoke({
    "messages": [{"role": "user", "content": "请从/memories/获取我最爱的水果是什么?"}]
}, config=config2)

print(res['messages'][-1].content)
