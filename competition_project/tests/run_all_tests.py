#!/usr/bin/env python3
"""运行所有测试"""

import os
import sys
import subprocess

# 设置环境变量
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_EVALUATE_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

def run_test(test_file):
    print("\n" + "="*60)
    print(f"运行测试: {test_file}")
    print("="*60)
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print(f"\n✓ {test_file} 完成 (退出码: {result.returncode})")
        return result.returncode == 0
    except Exception as e:
        print(f"✗ 运行失败: {e}")
        return False

def main():
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    
    test_files = [
        os.path.join(tests_dir, 'test_chunking.py'),
        os.path.join(tests_dir, 'test_rag_simple.py'),
        os.path.join(tests_dir, 'test_model_simple.py'),
        os.path.join(tests_dir, 'test_rag.py'),
    ]
    
    print("\n" + "="*60)
    print("开始运行所有测试")
    print("="*60)
    
    results = []
    for test in test_files:
        if os.path.exists(test):
            results.append((os.path.basename(test), run_test(test)))
        else:
            print(f"\n✗ 测试文件不存在: {test}")
            results.append((os.path.basename(test), False))
    
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    for name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{name}: {status}")

if __name__ == "__main__":
    main()
