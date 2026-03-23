from langchain_community.document_loaders import PyPDFLoader
pdf_path = "sample_rag_document.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"成功加载了{len(documents)}页文档。")
print(f"第一页的元数据：{documents[0].metadata}")

from langchain.text_splitter import CharacterTextSplitter
