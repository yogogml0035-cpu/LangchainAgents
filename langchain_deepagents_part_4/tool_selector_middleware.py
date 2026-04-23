from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolSelectorMiddleware
from langchain_deepseek import ChatDeepSeek

load_dotenv()

model = ChatDeepSeek(
    model="deepseek-chat",
)

@tool
def tool_1(input:str) -> str:
    """
    This is a useless tool, intended solely as an example.
    """
    return "This is a useless tool, intended solely as an example."
@tool
def tool_2(input:str) -> str:
    """
    This is a useless tool, intended solely as an example.
    """
    return "This is a useless tool, intended solely as an example."
@tool
def tool_3(input:str) -> str:
    """
    This is a useless tool, intended solely as an example.
    """
    return "This is a useless tool, intended solely as an example."
@tool
def tool_4(input:str) -> str:
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
    result = str(eval(expression))
    return result

agent = create_agent(
    model=model,
    tools=[tool_1, tool_2, tool_3, tool_4,calculate],
    middleware=[
        LLMToolSelectorMiddleware(
            model=model,
            max_tools=2,
            always_include=['tool_1'],
        ),
    ],
)

status = {
    'messages': '请计算2+3*4的值'
}

result = agent.invoke(status)
print(result)