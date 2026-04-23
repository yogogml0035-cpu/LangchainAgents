from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from deepagents.backends.filesystem import FilesystemBackend

load_dotenv()

model = ChatDeepSeek(
    model="deepseek-chat",
)

checkpointer = MemorySaver()

agent = create_deep_agent(
    model=model,
    backend=FilesystemBackend(root_dir="./", virtual_mode=True),
    skills=["./skills/"],
    checkpointer=checkpointer,  # Required!
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is langgraph?",
            }
        ]
    },
    config={"configurable": {"thread_id": "12345"}},
)

print(result)