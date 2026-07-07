#!/usr/bin/env python3
"""RAG 系统测试脚本"""

import os
# 在最顶部设置所有环境变量
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_EVALUATE_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_document_loader():
    """测试文档加载器"""
    print("=" * 50)
    print("测试 1: 文档加载器")
    print("=" * 50)

    from rag.document_loader import DocumentLoader, split_text

    # 创建测试文件
    test_file = "test_doc.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("这是一个测试文档。\n" * 100)

    try:
        loader = DocumentLoader(test_file)
        text = loader.load()
        print(f"✓ 文档加载成功，长度: {len(text)} 字符")

        chunks = split_text(text, chunk_size=100, overlap=20)
        print(f"✓ 文本切分成功，共 {len(chunks)} 个片段")
        print(f"  第一个片段: {chunks[0][:50]}...")
    finally:
        os.remove(test_file)

    print()


def test_vector_store():
    """测试向量存储"""
    print("=" * 50)
    print("测试 2: 向量存储")
    print("=" * 50)

    from rag.vector_store import VectorStore

    store = VectorStore()
    print("✓ 向量模型加载成功")

    # 添加测试文本
    test_texts = [
        "Python 是一种编程语言",
        "Java 也是一种编程语言",
        "北京是中国的首都",
        "上海是中国的经济中心",
    ]

    for text in test_texts:
        store.add(text)
    print(f"✓ 添加了 {len(test_texts)} 条文本")

    # 测试搜索
    query = "编程语言"
    results = store.search(query, k=2)
    print(f"✓ 搜索 '{query}' 成功，结果:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r}")

    print()


def test_rag_service():
    """测试完整 RAG 流程"""
    print("=" * 50)
    print("测试 3: RAG 完整流程")
    print("=" * 50)

    from rag.rag_service import RAGService

    # 创建测试文档
    test_file = "test_rag.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("""
人工智能（AI）是计算机科学的一个分支。
机器学习是人工智能的核心技术之一。
深度学习是机器学习的一个子集。
大语言模型基于深度学习技术构建。
""")

    try:
        service = RAGService()
        print("✓ RAGService 初始化成功")

        service.document_loader(test_file)
        print("✓ 文档加载到向量库成功")

        question = "什么是机器学习？"
        print(f"\n提问: {question}")
        answer = service.query(question)
        print(f"回答: {answer[:200]}...")
    finally:
        os.remove(test_file)

    print()


if __name__ == "__main__":
    print("\n开始 RAG 系统测试...\n")

    try:
        test_document_loader()
        test_vector_store()
        test_rag_service()

        print("=" * 50)
        print("所有测试通过！")
        print("=" * 50)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
