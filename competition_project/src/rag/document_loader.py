import os
from typing import List
import fitz
from docx import Document

class DocumentLoader:
    def __init__(self, document_dir: str):
        self.document_dir = document_dir

    def load(self) -> str:
        ext = os.path.splitext(self.document_dir)[1].lower()

        if ext == ".txt" or ext == ".md":
            return self._load_text()
        elif ext == ".docx":
            return self._load_docx()
        elif ext == ".pdf":
            return self._load_pdf()
        else:
            raise ValueError(f" 不支持的文件格式: {ext}")
        
    def _load_text(self) -> str:
        with open(self.document_dir, "r", encoding="utf-8") as f:
            return f.read()
    
    def _load_docx(self) -> str:
        doc = Document(self.document_dir)
        text = ""
        for para in doc.paragraphs:
            text += para.text
        return text
    
    def _load_pdf(self) -> str:
        doc = fitz.open(self.document_dir)
        text = ""
        for page in doc:
            text += page.get_text()
        return text



def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """把长文本切分成小块"""
    # chunk_size: 每块多少字
    # overlap: 重叠多少字（保证上下文连贯）
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
