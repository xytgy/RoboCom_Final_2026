from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-zh-v1.5", local_files_only=True)
model = AutoModel.from_pretrained("BAAI/bge-small-zh-v1.5", local_files_only=True)

inputs = tokenizer(["测试文本"], return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

print("模型输出正常")
embeddings = outputs.last_hidden_state[:, 0, :].numpy()
print(f"向量维度: {embeddings.shape}")