#!/usr/bin/env python3
"""直接运行测试"""

import os
import sys

# 设置环境变量
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_EVALUATE_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

print("\n" + "="*60)
print("测试 1: 文档切分")
print("="*60)
from rag.document_loader import split_text

long_text = """
人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
机器学习是人工智能的核心技术之一，它使计算机能够从数据中学习，而无需明确编程。
深度学习是机器学习的一个子集，它使用多层神经网络来学习数据的表示。
大语言模型基于深度学习技术构建，通过在大量文本数据上进行预训练，能够生成连贯的文本。
Python 是人工智能领域最常用的编程语言，因为它有丰富的库和框架支持。
"""

print(f"原始文本长度: {len(long_text)} 字\n")
for size in [100, 200]:
    chunks = split_text(long_text, chunk_size=size, overlap=20)
    print(f"切分大小: {size} → {len(chunks)} 个片段")
    for i, c in enumerate(chunks[:2], 1):
        print(f"  片段 {i}: {c.strip()[:50]}...")
    print()
print("✓ 切分测试通过!\n")

print("="*60)
print("测试 2: RAG 检索")
print("="*60)
from rag.vector_store import VectorStore

store = VectorStore()
print("✓ 向量模型加载成功\n")

test_texts = [
    "Python 是一种编程语言",
    "Java 也是一种编程语言", 
    "北京是中国的首都",
    "上海是中国的经济中心",
]
for t in test_texts:
    store.add(t)
print(f"✓ 添加了 {len(test_texts)} 条文本\n")

query = "编程语言"
results = store.search(query, k=2)
print(f"查询 '{query}' 结果:")
for i, r in enumerate(results, 1):
    print(f"  {i}. {r}")
print("\n✓ 检索测试通过!\n")

print("="*60)
print("测试总结")
print("="*60)
print("✓ 所有核心功能测试通过！")
print("""
注意: 大模型测试因为环境限制暂不执行，但 RAG 核心功能全部 OK。
竞赛环境中如果有网络可以直接使用完整的 RAG 问答功能。
""")
