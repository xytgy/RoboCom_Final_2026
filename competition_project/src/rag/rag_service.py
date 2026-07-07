import os
import ssl

# 禁用 SSL 验证
ssl._create_default_https_context = ssl._create_unverified_context

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from .document_loader import split_text, DocumentLoader
from .vector_store import VectorStore

class RAGService:
    def __init__(self):
        self.vector_store = VectorStore()

    def document_loader(self, document_path: str) -> None:
        """加载文档"""
        document_loader = DocumentLoader(document_path)
        text = document_loader.load()
        chunks = split_text(text)
        for chunk in chunks:
            self.vector_store.add(chunk)
    
    def query(self, question: str) -> str:
        """查询"""
        relevant_chunks = self.vector_store.search(question, k=3)
        context = "\n".join(relevant_chunks)
        prompt = f"根据以下信息回答问题：\n{context}\n\n问题：{question}"
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from model_service import generate_response
        return generate_response(prompt)
