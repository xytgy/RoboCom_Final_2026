from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from model_service import generate_response
from rag.rag_service import RAGService
import os
import shutil

class ChatRequest(BaseModel):
    message: str
    use_rag: bool = False

app = FastAPI()
rag_service = RAGService()

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传文档到 RAG 系统"""
    # 创建临时目录
    os.makedirs("temp_docs", exist_ok=True)
    file_path = os.path.join("temp_docs", file.filename)
    
    # 保存文件
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 加载到 RAG
    rag_service.document_loader(file_path)
    
    return {"message": f"文档 {file.filename} 已加载"}

@app.post("/chat")
async def chat(request: ChatRequest):
    """统一对话接口"""
    if request.use_rag:
        response = rag_service.query(request.message)
    else:
        response = generate_response(request.message)
    return {"response": response}