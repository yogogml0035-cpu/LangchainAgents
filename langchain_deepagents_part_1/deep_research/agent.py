"""Research Agent - Standalone script for LangGraph deployment.

This module creates a deep research agent with custom tools and prompts
for conducting web research with strategic thinking and context management.
"""

from datetime import datetime
import os

from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

from research_agent.prompts import (
    RESEARCHER_INSTRUCTIONS,
    RESEARCH_WORKFLOW_INSTRUCTIONS,
    SUBAGENT_DELEGATION_INSTRUCTIONS,
)
from research_agent.tools import tavily_search, think_tool

os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY", "sk-c045276d62aa4d00bc71f000ec925497")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "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")

max_concurrent_research_units = 3
max_researcher_iterations = 3

current_date = datetime.now().strftime("%Y-%m-%d")

INSTRUCTIONS = (
    RESEARCH_WORKFLOW_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_research_units=max_concurrent_research_units,
        max_researcher_iterations=max_researcher_iterations,
    )
)

research_sub_agent = {
    "name": "research-agent",
    "description": "Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.",
    "system_prompt": RESEARCHER_INSTRUCTIONS.format(date=current_date),
    "tools": [tavily_search, think_tool],
}

model = init_chat_model(
    model="deepseek-chat",
    model_provider="openai",  # 使用 OpenAI 适配器
    openai_api_key="sk-c045276d62aa4d00bc71f000ec925497",
    openai_api_base="https://api.deepseek.com", # 官方 API 地址
)

agent = create_deep_agent(
    model=model,
    tools=[tavily_search, think_tool],
    system_prompt=INSTRUCTIONS,
    subagents=[research_sub_agent],
)

from utils import format_messages

result = agent.invoke(                  
    {
        "messages": [
            {
                "role": "user",
                "content": "探究Langgraph怎么写一个智能体",
            }
        ],
    }, 
)
format_messages(result["messages"])
