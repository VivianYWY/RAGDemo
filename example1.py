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
