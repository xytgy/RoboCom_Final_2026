import os
import ssl

# 禁用 SSL 验证
ssl._create_default_https_context = ssl._create_unverified_context

# 完全禁用 Hugging Face 网络请求
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_EVALUATE_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# 在导入 transformers 前 mock 掉 adapter 检查
from unittest.mock import patch

# Mock find_adapter_config_file 避免网络请求
def mock_find_adapter_config(*args, **kwargs):
    return None

# 先应用 mock
with patch('transformers.utils.peft_utils.find_adapter_config_file', side_effect=mock_find_adapter_config):
    from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2-0.5B"
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, 
    local_files_only=True,
    trust_remote_code=True,
    use_fast=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    local_files_only=True,
    trust_remote_code=True,
    device_map=None,
    low_cpu_mem_usage=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_response(message: str) -> str:
    inputs = tokenizer(message, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
