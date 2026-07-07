#!/usr/bin/env python3
"""简化版 RAG 测试 - 只测试检索部分"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_rag_retrieval():
    """测试 RAG 检索功能"""
    print("\n" + "=" * 50)
    print("简化版 RAG 检索测试")
    print("=" * 50 + "\n")

    from rag.vector_store import VectorStore
    from rag.document_loader import DocumentLoader, split_text

    print("步骤 1: 初始化向量存储...")
    store = VectorStore()
    print("✓ 向量模型加载成功\n")

    print("步骤 2: 创建测试文档...")
    test_file = "test_simple.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("""
人工智能（AI）是计算机科学的一个分支。
机器学习是人工智能的核心技术之一。
深度学习是机器学习的一个子集。
大语言模型基于深度学习技术构建。
Python 是人工智能领域最常用的编程语言。
""")
    print(f"✓ 创建测试文件: {test_file}\n")

    print("步骤 3: 加载并切分文档...")
    loader = DocumentLoader(test_file)
    text = loader.load()
    chunks = split_text(text)
    print(f"✓ 文档加载成功，切分为 {len(chunks)} 个片段\n")

    print("步骤 4: 添加到向量库...")
    for chunk in chunks:
        store.add(chunk)
    print("✓ 添加完成\n")

    print("步骤 5: 测试检索...")
    queries = ["什么是机器学习？", "Python 是做什么的？", "大语言模型基于什么技术？"]
    
    for query in queries:
        print(f"\n查询: {query}")
        results = store.search(query, k=2)
        print("检索结果:")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r.strip()}")

    print("\n" + "=" * 50)
    print("检索测试完成！")
    print("=" * 50)

    os.remove(test_file)

if __name__ == "__main__":
    test_rag_retrieval()
