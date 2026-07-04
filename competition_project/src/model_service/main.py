import os
# 禁用 Hugging Face 网络请求，强制使用本地缓存
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

# 定义请求体模型
class ChatRequest(BaseModel):
    message: str

# 加载本地缓存的模型
MODEL_NAME = "Qwen/Qwen2-0.5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True)

# 设置 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        inputs = tokenizer(request.message, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}
