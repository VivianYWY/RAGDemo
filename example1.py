from langchain_community.document_loaders import PyPDFLoader
pdf_path = "sample_rag_document.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"成功加载了{len(documents)}页文档。")
print(f"第一页的元数据：{documents[0].metadata}")

from langchain.text_splitter import CharacterTextSplitter
plain_text = "段落一：这是第一个段落。\n\n段落二：这是第二个段落，内容稍长一些，用于演示分割效果。"

# 1. 初始化 CharacterTextSplitter，指定分隔符
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=100,
    chunk_overlap=20,
)

# 2. 对文本进行分块
chunks = text_splitter.split_text(plain_text)

# 3. 打印结果
print(f"切分出了 {len(chunks)} 个块。")
for i, chunk in enumerate(chunks):
    print(f"块 {i+1}: '{chunk}'")

# 输出示例：
# 切分出了 2 个块。
# 块 1: '段落一：这是第一个段落。'
# 块 2: '段落二：这是第二个段落，内容稍长一些，用于演示分割效果。'

from langchain.text_splitter import RecursiveCharacterTextSplitter

# 一个更复杂的文本示例，包含多种潜在的分隔符
complex_text = "RAG系统介绍。\n核心组件包括：检索器和生成器。检索器负责从知识库中获取信息。生成器则负责利用这些信息回答问题。\n\n这是一个新的段落。"

# 1. 初始化RecursiveCharacterTextSplitter
# 无须指定分隔符，它会使用默认的优先列表

# 假设 'chunks' 对象已通过上一节的代码准备好
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 1. 准备演示用的 Document 对象（代替上一节的输出）
chunks_as_docs = [
    Document(page_content="RAG系统的核心是检索与生成。"),
    Document(page_content="FAISS是一个高效的向量相似性搜索库。"),
    Document(page_content="Embedding模型负责将文本转换为向量。")
]

# 2. 初始化开源的 Embedding 模型
# model_name 指定了要使用的模型，device='cpu'表示在CPU上运行
model_name = "BAAI/bge-small-zh-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embeddings_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# 3. 使用 FAISS.from_documents() 类方法，一步完成向量化与索引构建
# 该方法会遍历所有文本块，使用指定的 embedding 模型将其向量化，
# 然后将所有向量存入 FAISS 索引中。
vector_store = FAISS.from_documents(chunks_as_docs, embeddings_model)

# 4. 验证索引构建是否成功
print(f"向量索引已成功构建，包含 {vector_store.index.ntotal} 个向量。")
# 输出示例：向量索引已成功构建，包含 3 个向量。

# 假设 'vector_store' 对象已通过上一节的代码构建完成
# (为保持代码独立, 此处包含了上一节构建过程的简化版本)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --- 前置准备 ---
chunks_as_docs = [Document(page_content="FAISS 是一个高效的向量相似性搜索库。")]
embeddings_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'}
)
vector_store = FAISS.from_documents(chunks_as_docs, embeddings_model)
# --- 前置准备结束 ---

# 1. 定义用户查询
query = "什么是 FAISS? "

# 2. 调用 similarity_search 方法执行检索
# k=1 表示我们希望返回最相似的 1 个结果
retrieved_docs = vector_store.similarity_search(query, k=1)

# 3. 打印检索结果
print(f"查询: '{query}'")

from langchain.prompts import PromptTemplate
# 1. 定义一个包含'context'和'question'两个输入变量的提示词模板字符串
prompt_template_str = """
请基于以下提供的上下文信息来回答问题。
如果上下文中没有足够的信息来回答问题，请直接说"根据提供的资料，我无法回答该问题。"
上下文:
{context}
问题:
{question}
回答:
"""

# 2. 使用该字符串初始化 PromptTemplate 对象
rag_prompt_template = PromptTemplate.from_template(prompt_template_str)

# 3. 准备示例的上下文和问题
sample_context = "FAISS 是一个由 Facebook AI 研究院开发的高效向量相似性搜索库。"
sample_question = "FAISS 是由哪个公司开发的？"

# 4. 使用 .format() 方法填充模板，生成最终的提示
final_prompt = rag_prompt_template.format(
    context=sample_context,
    question=sample_question
)

# 5. 打印生成的最终提示
print(final_prompt)

def mock_llm_call(prompt):
    # 这是一个模拟调用，实际应用中应替换为真实的 LLM 调用
    if "优点" in prompt and "RAG" in prompt:
        return "RAG技术有哪些优点？"
    return "无法改写查询"

# 1. 初始化对话历史
chat_history = ChatMessageHistory()
chat_history.add_user_message("什么是 RAG 技术？")
chat_history.add_ai_message("RAG 是检索增强生成技术的缩写。")

# 2. 用户的最新追问
latest_question = "它有哪些优点？"

# 3. 构建用于查询改写的提示词模板
rewrite_prompt_template = PromptTemplate.from_template(
    """根据以下的对话历史和用户提出的后续问题，
请将后续问题改写成一个独立的、无须依赖对话历史就能被理解的新问题。
对话历史：
{chat_history}
后续问题：
{question}
改写后的独立问题：
"""
)
