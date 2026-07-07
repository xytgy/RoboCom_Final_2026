# RAG 系统 - 2026 睿抗竞赛

这是一个完整的 RAG（检索增强生成）系统，用于 2026 睿抗竞赛。

## 项目结构

```
competition_project/
├── README.md                 # 项目说明
├── requirements.txt          # Python 依赖
├── src/                      # 源代码
│   ├── main.py               # 主入口（FastAPI）
│   ├── model_service.py      # 大模型服务
│   └── rag/                  # RAG 模块
│       ├── __init__.py
│       ├── document_loader.py  # 文档加载器
│       ├── vector_store.py    # 向量存储
│       └── rag_service.py     # RAG 服务
├── tests/                    # 测试文件
│   ├── test_chunking.py      # 文档切分测试
│   ├── test_rag_simple.py    # 简化 RAG 测试
│   ├── test_model_simple.py  # 大模型测试
│   ├── test_rag.py          # 完整 RAG 测试
│   ├── test_model.py
│   ├── test_model_only.py
│   ├── run_all_tests.py     # 运行所有测试
│   ├── run_tests_here.py    # 直接运行测试
│   └── run_and_save.py      # 运行并保存结果
└── docs/                    # 文档
    └── test_summary.md      # 测试总结
```

## 环境配置

### 必需环境变量

在运行代码前，确保以下环境变量已设置（已在代码中自动设置）：

```python
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

### 模型缓存

确保以下模型已下载到本地 Hugging Face 缓存：
- `Qwen/Qwen2-0.5B` - 大语言模型
- `BAAI/bge-small-zh-v1.5` - 中文 Embedding 模型

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行测试

```bash
# 运行所有测试
cd tests
python run_all_tests.py

# 或直接运行简单测试
python run_tests_here.py
```

### 3. 启动服务

```bash
cd src
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 功能模块

### 文档加载器
- 支持格式：`.txt`, `.md`, `.docx`, `.pdf`
- 可配置切分大小和重叠

### 向量存储
- 使用 BGE 中文 Embedding
- FAISS 索引，快速检索

### RAG 服务
- 完整流程：加载 → 切分 → 向量化 → 检索 → 生成回答

## 注意事项

1. **SSL 问题**：已在代码中禁用 SSL 验证，适配中国网络环境
2. **离线模式**：默认使用离线模式，依赖本地模型缓存
3. **竞赛环境**：确保竞赛环境中有网络或正确配置的本地模型

## 测试

详见 `docs/test_summary.md`。
