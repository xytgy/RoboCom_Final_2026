#!/usr/bin/env python3
"""只测试大模型加载"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

print("\n" + "=" * 50)
print("大模型加载测试")
print("=" * 50 + "\n")

print("正在加载模型...")

from unittest.mock import patch

# 先 mock 掉 adapter 检查
def mock_find_adapter(*args, **kwargs):
    return None

with patch('transformers.utils.peft_utils.find_adapter_config_file', side_effect=mock_find_adapter):
    from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2-0.5B"

print("正在加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, 
    local_files_only=True,
    trust_remote_code=True,
    use_fast=True
)
print("✓ Tokenizer 加载成功！\n")

print("正在加载 model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    local_files_only=True,
    trust_remote_code=True,
    device_map=None,
    low_cpu_mem_usage=True
)
print("✓ Model 加载成功！\n")

print("测试生成...")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer("你好，请介绍一下自己。", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("=" * 50)
print(f"生成结果:")
print(response)
print("=" * 50)
print("\n测试完成！")
