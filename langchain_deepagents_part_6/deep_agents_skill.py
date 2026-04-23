from pathlib import Path
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langchain_deepseek import ChatDeepSeek
from deepagents.backends import LocalShellBackend
from deepagents import create_deep_agent

load_dotenv()

model = ChatDeepSeek(
    model="deepseek-chat",
)

# 上下文记忆
checkpointer = MemorySaver()

root_dir = Path.cwd().as_posix()

print(root_dir)

backend = LocalShellBackend(
    root_dir=root_dir,
    inherit_env=True,
    timeout=120,  # 命令超时秒数
    max_output_bytes=100000
)

system_prompt = '''
## 角色设定
你是一位专业、高效、多领域的超级智能助手，具备强大的知识整合与问题解决能力。你善于理解用户意图，提供准确、清晰、有温度的回答。

## 核心任务
- 根据用户提问，结合你的专业知识库与可用工具（skills），提供高质量解答
- 回答需遵循：准确性 > 实用性 > 简洁性 > 友好性 的优先级原则
- 遇到模糊问题时，主动澄清需求；遇到复杂问题时，分步骤拆解说明

## 注意事项
read_file工具使用注意点: 不支持Windows绝对地址, 如: 错误写法 D:\\xxx\\xxx\\SKILL.md, 正确写法为 /xxx/xxx/SKILL.md
'''

agent = create_deep_agent(
    model=model,
    backend=backend,
    skills=[root_dir + r'/skills'],
    system_prompt=system_prompt,
    checkpointer = checkpointer,
)

while True:
    question = input('请输入:')
    if not question:
        continue
    if question == 'q':
        break
    for type, chunk in agent.stream({
        'messages': [
            {
                'role': 'user',
                'content': question
            },
        ]
    },
            config={"configurable": {"thread_id": "12345"}},
            stream_mode=["updates"]
    ):
        if "SkillsMiddleware.before_agent" in chunk and chunk["SkillsMiddleware.before_agent"]:
            skills = chunk['SkillsMiddleware.before_agent']['skills_metadata']
            print(">" * 10, "加载Skills", "<" * 30)
            for skill in skills:
                print('Load Skill:', skill['name'])

        if 'model' in chunk:
            message = chunk['model']['messages'][0]
            content = message.content
            if content:
                print(">" * 10, "AIMessage", "<" * 30)
                print(content)
            tool_calls = message.tool_calls
            if tool_calls:
                print(">" * 10, "Call Tools", "<" * 30)
                for t in tool_calls:
                    print(f'Tool:{t['name']}, Args:{t['args']}')

        if 'tools' in chunk:
            print(">" * 20, "Tools Output", "<" * 20)
            for m in chunk['tools']['messages']:
                print(f"Tool:{m.name}, Output: \n{m.content}")
