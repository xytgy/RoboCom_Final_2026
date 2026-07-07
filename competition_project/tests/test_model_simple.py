#!/usr/bin/env python3
"""最简单的大模型测试"""

import os
import ssl

# 先禁用 SSL
ssl._create_default_https_context = ssl._create_unverified_context

# 禁用网络请求
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

print("\n" + "=" * 60)
print("大模型加载测试 (简化版)")
print("=" * 60 + "\n")

# 直接加载，不做任何多余检查
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2-0.5B"

print("正在加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=True,
    trust_remote_code=True
)
print("✓ Tokenizer 加载成功\n")

print("正在加载 model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    local_files_only=True,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
print("✓ Model 加载成功\n")

print("正在设置 pad token...")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("✓ Pad token 设置完成\n")

print("正在测试生成...")
prompt = "你好，世界！"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=30, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("=" * 60)
print(f"输入: {prompt}")
print(f"输出: {response}")
print("=" * 60)
print("\n✓ 大模型测试完成！")
