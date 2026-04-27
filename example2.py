
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
