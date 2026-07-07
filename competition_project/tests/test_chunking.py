#!/usr/bin/env python3
"""测试文档切分功能"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag.document_loader import split_text

print("\n" + "=" * 50)
print("文档切分测试")
print("=" * 50 + "\n")

# 创建一个更长的测试文本
long_text = """
人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
机器学习是人工智能的核心技术之一，它使计算机能够从数据中学习，而无需明确编程。
深度学习是机器学习的一个子集，它使用多层神经网络来学习数据的表示。
大语言模型基于深度学习技术构建，通过在大量文本数据上进行预训练，能够生成连贯的文本。
Python 是人工智能领域最常用的编程语言，因为它有丰富的库和框架支持。
自然语言处理（NLP）是人工智能的一个重要方向，专注于让计算机理解和生成人类语言。
计算机视觉则让计算机能够从图像或视频中获取信息，识别物体、场景和活动。
强化学习是一种让智能体通过与环境交互来学习最优策略的机器学习方法。
迁移学习允许将在一个任务上学到的知识应用到其他相关任务上，提高训练效率。
"""

print(f"原始文本长度: {len(long_text)} 字\n")

# 测试不同的切分大小
chunk_sizes = [100, 200, 300]

for size in chunk_sizes:
    chunks = split_text(long_text, chunk_size=size, overlap=20)
    print(f"切分大小: {size} 字 → 生成 {len(chunks)} 个片段")
    for i, chunk in enumerate(chunks[:2], 1):  # 只显示前2个
        print(f"  片段 {i}: {chunk.strip()[:60]}...")
    print()

print("=" * 50)
print("切分测试完成！")
print("=" * 50)
