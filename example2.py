
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class ReasoningEngine:
    """推理引擎：负责任务分解与行动策略制定"""
    def __init__(self, model_name: str = "gpt-4-turbo"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.parser = JsonOutputParser()
        # 初始化推理分析提示词
        prompt_template = """你是一个任务规划专家。请将复杂任务分解为可执行的子任务。
任务：{task}
请返回 JSON 格式：{{"sub_tasks": ["步骤1", "步骤2"], "strategy": "分解策略"}}
"""
self.chain = ChatPromptTemplate.from_template(prompt_template) | self.llm | self.parser

def decompose_task(self, task: str) -> dict:
    """执行任务分解"""
    return self.chain.invoke({"task": task})

# 使用示例
engine = ReasoningEngine()
result = engine.decompose_task("分析企业 Q4 销售数据并制定营销策略")

from datetime import datetime
from typing import List, Dict

class ShortTermMemory:
    """短期记忆：维护会话级别的上下文信息和交互历史"""
    def __init__(self, max_history_length: int = 50):
        self.conversation_history = []
        self.max_history = max_history_length

    def add_interaction(self, user_input: str, agent_response: str):
        """记录交互过程到短期记忆"""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "agent_response": agent_response
        }
