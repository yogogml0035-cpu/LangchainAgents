from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import (
    LLMToolSelectorMiddleware,
    TodoListMiddleware,
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_deepseek import ChatDeepSeek
from langgraph.config import get_stream_writer
from typing import Callable
from pathlib import Path
import json


# =========================
# 环境变量
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
env_path = PROJECT_ROOT / "langchain_deepagents_part_1" / "deep_research" / ".env"
load_dotenv(dotenv_path=env_path)


# =========================
# 模型
# =========================
model = ChatDeepSeek(
    model="deepseek-chat",
)


# =========================
# 自定义中间件：打印并流式输出 wrap_model_call
# =========================
class DebugWrapModelMiddleware(AgentMiddleware):
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """
        包住每一次模型调用：
        1) 控制台打印 before / after
        2) 往 custom stream 写事件，便于在 agent.stream() 里看到
        """

        writer = None
        try:
            writer = get_stream_writer()
        except Exception:
            # 如果当前上下文拿不到 writer，不影响正常执行
            writer = None

        # ===== BEFORE =====
        before_payload = {
            "node": "DebugWrapModelMiddleware.wrap_model_call",
            "phase": "before",
            "message_count": len(request.messages) if request.messages else 0,
            "tool_count": len(request.tools) if request.tools else 0,
            "tool_names": [
                getattr(t, "name", str(t)) for t in (request.tools or [])
            ],
        }

        print("=" * 80)
        print("节点名称: DebugWrapModelMiddleware.wrap_model_call")
        print("节点类型: [middleware-wrap]")
        print("阶段: before")
        print(f"消息数量: {before_payload['message_count']}")
        print(f"工具数量: {before_payload['tool_count']}")
        print(f"工具列表: {before_payload['tool_names']}")
        print("-" * 80)

        if writer:
            writer(before_payload)

        # 真正调用模型
        response = handler(request)

        # ===== AFTER =====
        result_messages = response.result if response and response.result else []
        ai_message_count = len(result_messages)

        tool_call_count = 0
        last_content = None

        if result_messages:
            last_msg = result_messages[-1]
            last_content = getattr(last_msg, "content", None)
            tool_calls = getattr(last_msg, "tool_calls", None)
            if tool_calls:
                tool_call_count = len(tool_calls)

        after_payload = {
            "node": "DebugWrapModelMiddleware.wrap_model_call",
            "phase": "after",
            "result_message_count": ai_message_count,
            "tool_call_count": tool_call_count,
            "last_content": last_content,
        }

        print("=" * 80)
        print("节点名称: DebugWrapModelMiddleware.wrap_model_call")
        print("节点类型: [middleware-wrap]")
        print("阶段: after")
        print(f"返回消息数量: {after_payload['result_message_count']}")
        print(f"tool_calls 数量: {after_payload['tool_call_count']}")
        print(f"最后一条内容: {after_payload['last_content']}")
        print("-" * 80)

        if writer:
            writer(after_payload)

        return response


# =========================
# 工具
# =========================
@tool
def tool_1(input: str) -> str:
    """
    This is a useless tool, intended solely as an example.
    """
    return "This is a useless tool, intended solely as an example."


@tool
def tool_2(input: str) -> str:
    """
    This is a useless tool, intended solely as an example.
    """
    return "This is a useless tool, intended solely as an example."


@tool
def tool_3(input: str) -> str:
    """
    This is a useless tool, intended solely as an example.
    """
    return "This is a useless tool, intended solely as an example."


@tool
def tool_4(input: str) -> str:
    """
    This is a useless tool, intended solely as an example.
    """
    return "This is a useless tool, intended solely as an example."


@tool
def calculate(expression: str) -> str:
    """Perform mathematical calculations and return the result.
    Args:
        expression: Mathematical expression to evaluate
        (e.g., "2 + 3 * 4", "sqrt(16)", "sin(pi/2)")
    Returns:
        The calculated result as a string
    """
    expression = expression.replace("^", "**")
    result = str(eval(expression))
    return result


# =========================
# 创建 Agent
# =========================
agent = create_agent(
    model=model,
    tools=[tool_1, tool_2, tool_3, tool_4, calculate],
    middleware=[
        # 放最前面，让它尽量包住后续模型调用链
        DebugWrapModelMiddleware(),
        LLMToolSelectorMiddleware(
            model=model,
            max_tools=2,
            always_include=["tool_1"],
        ),
        TodoListMiddleware(),
    ],
)


# =========================
# 打印函数：处理 updates 流里的 data
# =========================
def format_update_chunk(chunk_data):
    """格式化 updates 模式下的 data"""
    if not isinstance(chunk_data, dict):
        print(f"节点类型: {type(chunk_data).__name__}")
        print(f"节点内容: {chunk_data}")
        return

    for key, value in chunk_data.items():
        print("=" * 80)

        if key == "model":
            print(f"节点名称: {key}")
            print("节点类型: [model] - AI消息 (AIMessage)")
            if isinstance(value, dict) and "messages" in value:
                messages = value["messages"]
                print(f"  源消息: {messages}")
                for msg in messages:
                    msg_type = msg.__class__.__name__.replace("Message", "")
                    print(f"  消息类型: {msg_type}")

                    if hasattr(msg, "content") and msg.content:
                        if isinstance(msg.content, str):
                            print(f"  内容: {msg.content}")
                        elif isinstance(msg.content, list):
                            print(f"  内容列表: {len(msg.content)} 项")

                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        print(f"\n  🔧 Tool Calls ({len(msg.tool_calls)}个):")
                        for i, tc in enumerate(msg.tool_calls, 1):
                            print(f"    [{i}] 工具: {tc.get('name', 'N/A')}")
                            print(f"        ID: {tc.get('id', 'N/A')}")
                            args = tc.get("args", {})
                            if "todos" in args:
                                print(f"        Todos数量: {len(args['todos'])}")
                                for j, todo in enumerate(args["todos"], 1):
                                    status = todo.get("status", "unknown")
                                    content = todo.get("content", "")
                                    print(f"          [{j}] [{status}] {content}")
                            else:
                                print(f"        参数: {json.dumps(args, ensure_ascii=False)}")
            else:
                print(f"  值: {value}")

        elif key.endswith(".after_model"):
            print(f"节点名称: {key}")
            print("节点类型: [middleware-after_model]")
            print(f"  值: {value}")

        elif key == "tools":
            print("节点类型: [tools] - 工具执行结果")
            if isinstance(value, dict):
                if "todos" in value:
                    print("  Todos更新:")
                    for todo in value["todos"]:
                        status = todo.get("status", "unknown")
                        content = todo.get("content", "")
                        print(f"    [{status}] {content}")

                if "messages" in value:
                    print(f"  源消息: {value['messages']}")
                    print(f"\n  消息列表 ({len(value['messages'])}条):")
                    for msg in value["messages"]:
                        msg_type = msg.__class__.__name__.replace("Message", "")
                        print(f"    消息类型: {msg_type}")
                        if hasattr(msg, "name"):
                            print(f"    工具名: {msg.name}")
                        if hasattr(msg, "content"):
                            content = str(msg.content)
                            print(f"    内容: {content}")

        else:
            print(f"节点类型: [{key}]")
            print(f"  值: {value}")

        print("-" * 80)


# =========================
# 打印函数：处理 v2 stream part
# =========================
def format_stream_part(part):
    """
    v2 格式下，每个 chunk 统一是：
    {
        "type": "updates" | "custom" | ...
        "ns": (),
        "data": ...
    }
    """
    if not isinstance(part, dict):
        print(f"[未知chunk] {part}")
        return

    chunk_type = part.get("type")
    data = part.get("data")
    ns = part.get("ns", ())

    if chunk_type == "updates":
        format_update_chunk(data)

    elif chunk_type == "custom":
        print("=" * 80)
        print("流类型: [custom]")
        print(f"命名空间: {ns}")

        if isinstance(data, dict) and data.get("node") == "DebugWrapModelMiddleware.wrap_model_call":
            print(f"节点名称: {data.get('node')}")
            print("节点类型: [middleware-wrap]")
            print(f"阶段: {data.get('phase')}")

            if data.get("phase") == "before":
                print(f"消息数量: {data.get('message_count')}")
                print(f"工具数量: {data.get('tool_count')}")
                print(f"工具列表: {data.get('tool_names')}")
            elif data.get("phase") == "after":
                print(f"返回消息数量: {data.get('result_message_count')}")
                print(f"tool_calls 数量: {data.get('tool_call_count')}")
                print(f"最后一条内容: {data.get('last_content')}")
        else:
            print(f"custom 数据: {data}")

        print("-" * 80)

    else:
        print("=" * 80)
        print(f"流类型: [{chunk_type}]")
        print(f"命名空间: {ns}")
        print(f"数据: {data}")
        print("-" * 80)


# =========================
# 主程序
# =========================
if __name__ == "__main__":
    from IPython.display import Image, display

    # 画图并保存到本地
    img_data = agent.get_graph().draw_mermaid_png()
    display(Image(img_data))
    with open("agent_graph.png", "wb") as f:
        f.write(img_data)

    # 重点：
    # 1) 开启 updates + custom
    # 2) 使用 version="v2"
    res = agent.stream(
        {
            "messages": [
                "你要一步一步的详细规划以下内容再进行回答。请回答2+3*4的值。"
            ]
        },
        stream_mode=["updates", "custom"],
        version="v2",
    )

    for part in res:
        if isinstance(part, tuple) and len(part) == 2:
            mode, chunk = part
            if mode == "updates":
                format_update_chunk(chunk)
            elif mode == "custom":
                print("=" * 80)
                print("流类型: [custom]")
                print(chunk)
                print("-" * 80)
        else:
            format_stream_part(part)
