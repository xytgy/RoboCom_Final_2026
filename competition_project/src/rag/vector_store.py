import os
import ssl

# 禁用 SSL 验证
ssl._create_default_https_context = ssl._create_unverified_context

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import faiss
import numpy as np
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch

class VectorStore:
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        """加载向量模型"""
        self.texts = []
        self.index = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.model.eval()
    
    def add(self, text: str) -> None:
        """添加文本到向量存储"""
        self.texts.append(text)
        vectors = self.encode([text])
        if self.index is None:
            self.create_index(vectors)
        else:
            self.index.add(vectors)
    
    def encode(self, text: List[str]) -> np.ndarray:
        """将文本转换为向量"""
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings.astype('float32')
    
    def create_index(self, vectors: np.ndarray) -> None:
        """创建向量索引"""
        if vectors.shape[0] == 0:
            raise ValueError("输入的向量数组不能为空")
        dimension = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(vectors)
    
    def search(self, query: str, k: int = 5) -> List[str]:
        """根据查询向量搜索相似的文本"""
        if not self.texts or self.index is None:
            return []
        query_embeddings = self.encode([query])
        _, indices = self.index.search(query_embeddings, min(k, len(self.texts)))
        return [self.texts[i] for i in indices[0]]
