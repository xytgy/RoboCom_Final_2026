# 2026 睿抗决赛 — Mac 过流程记录

> 日期：2026-07-04
> 目的：在 Mac 上先过一遍完整流程，熟悉每一步做什么，之后到 Windows（有 NVIDIA GPU）上实战。

---

## 整体计划

12 个任务是一条完整链路：

```
任务1-2：让模型能跑、能被调用（环境搭建 + 模型服务部署）
任务3-4：让模型更懂特定行业（数据准备 + 微调）
任务5-8：把模型变成用户能用的应用（提示词、RAG、前端、多轮对话）
任务9-11：把应用升级为能协作的智能体系统（工具调用、多智能体、跨场景编排）
任务12：给系统加上安全控制并完成综合展示
```

**评分权重：**
- 模块一（大模型安装部署与微调）：30%
- 模块二（大模型应用开发）：35%
- 模块三（智能体应用与开发）：35%
- 职业素养（团队协作、操作规范）：5%

---

## 任务一：开发环境与基础依赖准备（10%）

### 目标
在服务器上搭好 Python 环境，装好所有依赖，跑通 demo，截图证明。

### Mac 上的实操步骤

**1. 安装 Miniconda**

Mac 上没有 conda，需要先装：

```bash
# 下载并安装（Apple Silicon / arm64）
curl -s -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda3

# 初始化 shell
$HOME/miniconda3/bin/conda init zsh

# 重新打开终端，或手动 source
source $HOME/miniconda3/etc/profile.d/conda.sh
```

验证安装：
```bash
conda --version   # 应显示 conda 26.x.x
```

**2. 创建虚拟环境**

```bash
conda create -n robocom python=3.10 -y
conda activate robocom
```

> 虚拟环境就是一个隔离空间，装的东西不会影响系统。`robocom` 是环境名，可以随便取。
> 激活后终端前面会出现 `(robocom)` 提示符。

**3. 安装依赖包**

```bash
# PyTorch（Mac CPU 版，Windows 上换成 GPU 版）
pip install torch torchvision torchaudio

# 核心依赖
pip install transformers accelerate sentencepiece protobuf
pip install fastapi uvicorn
pip install vllm          # 高性能模型推理服务（需要 GPU）
pip install peft trl      # 微调相关
pip install datasets       # 数据集处理

# 应用开发相关
pip install gradio         # 前端界面（推荐）
pip install faiss-cpu      # RAG 向量检索

# 导出依赖清单
pip freeze > requirements.txt
```

> **Windows 实战时注意：** PyTorch 要装 GPU 版，命令参考 https://pytorch.org/get-started/locally/
> 例如：`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

**4. 验证安装**

```bash
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
```

> Windows 上还要验证 GPU：
> ```bash
> python -c "import torch; print(torch.cuda.is_available())"
> ```
> 输出 `True` 就说明 GPU 可用。

### 项目目录结构

```
competition_project/
├── configs/              # 配置文件（模型参数、训练参数等）
├── data/
│   ├── raw/              # 原始数据
│   ├── processed/        # 清洗后的数据
│   └── finetune/         # 微调用的数据
├── src/
│   ├── model_service/    # 模型服务部署（FastAPI）
│   ├── rag/              # RAG 知识库
│   ├── app/              # 前端应用（Gradio/Streamlit）
│   ├── agent/            # 智能体
│   └── security/         # 安全模块
├── tests/                # 测试代码
├── outputs/              # 输出结果
└── docs/                 # 文档
```

> **注意：** Git 不跟踪空文件夹。如果需要在 GitHub 上保留空目录结构，需要在每个空文件夹里放一个 `.gitkeep` 文件（内容可以为空）。

### 关键概念速记

| 包名 | 干什么用的 |
|------|-----------|
| torch | PyTorch，跑模型的核心框架 |
| transformers | Hugging Face 的模型加载和推理库 |
| accelerate | 分布式训练/推理加速 |
| sentencepiece | 分词器（模型需要它来切分文本） |
| protobuf | Google 的数据序列化格式，一些模型依赖它 |
| fastapi | 快速搭建 RESTful API 服务 |
| uvicorn | ASGI 服务器，用来运行 FastAPI |
| vllm | 高性能大模型推理引擎（需要 GPU） |
| peft | 参数高效微调（LoRA 等） |
| trl | 基于 Transformer 的强化学习训练 |
| datasets | Hugging Face 的数据集加载和处理 |
| gradio | 快速搭建模型演示前端界面 |
| faiss-cpu | Facebook 的向量相似度检索库 |

---

## 任务二：模型服务部署（10%）

### 目标
用 FastAPI 把大模型包装成一个 HTTP 接口，让后面的应用和智能体可以调用。

### 核心思路
就像开一家餐厅：模型是厨师，FastAPI 是服务员，客人（前端/智能体）通过点单（HTTP 请求）让厨师做菜（推理）。

### 示例代码框架

```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

# 加载模型和分词器
model_path = "/path/to/model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.post("/chat")
async def chat(req: ChatRequest):
    inputs = tokenizer(req.message, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=512)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}
```

启动服务：
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 任务三：数据准备与清洗（10%）

### 目标
拿到一批行业数据（如客服对话记录），做清洗、脱敏、格式转换，变成微调能用的 JSONL 格式。

### 核心步骤
1. **清洗**：去掉乱码、重复、太短的内容
2. **脱敏**：把手机号、身份证号、姓名等替换成占位符
3. **格式转换**：转成微调需要的对话格式

### 微调数据 JSONL 格式示例

```json
{"instruction": "用户的问题或指令", "input": "可选的额外上下文", "output": "模型应该给出的回答"}
```

> **微调 vs RAG 的区别：**
> - 微调 = 让模型去上课，学完之后脑子里就有新知识了（改变模型权重）
> - RAG = 考试的时候给你一本参考书，模型本身没变，只是能查资料（不改变模型权重）

---

## 任务四：模型微调（10%）

### 目标
用清洗好的数据，对大模型做 LoRA 微调，让它在特定行业表现更好。

### 核心思路
全量微调太贵（需要巨大显存），LoRA 只训练一小部分参数，效果接近但资源消耗小很多。

### 关键参数
- `lora_r`：LoRA 秩，一般 8~64，越大能力越强但越慢
- `lora_alpha`：缩放系数，一般设为 r 的 2 倍
- `learning_rate`：学习率，LoRA 一般用 1e-4 ~ 3e-4
- `num_epochs`：训练轮数，数据少的话 2~5 轮就够

---

## 任务五：提示词工程（5%）

### 目标
为不同功能场景设计合适的 System Prompt，让模型按要求回答。

### 要点
- 明确角色：告诉模型"你是 XX 领域的专家"
- 约束行为：比如"只回答相关问题，不回答无关内容"
- 格式要求：比如"请用 JSON 格式输出"
- 不同功能用不同提示词（客服模式、分析模式、摘要模式等）

---

## 任务六：RAG 知识库问答（15%）

### 目标
搭建一个检索增强生成系统，让模型能基于企业内部文档回答问题。

### 核心流程
```
文档 → 切分成小块 → 用 Embedding 模型转成向量 → 存入向量数据库（FAISS）
                                                    ↓
用户提问 → 问题也转成向量 → 在 FAISS 里找最相关的几个块 → 把问题和这些块一起喂给模型 → 生成回答
```

### 关键技术点
- 文本切分：按段落或固定长度切，注意重叠（overlap）
- Embedding 模型：可以用 `bge-large-zh`、`m3e` 等中文模型
- 向量数据库：FAISS（轻量，适合比赛）
- 检索策略：可以结合关键词检索（BM25）+ 向量检索做混合检索

---

## 任务七：多轮对话应用（10%）

### 目标
做一个有上下文记忆的多轮对话应用，用户不用每次重复背景信息。

### 核心思路
维护一个 `history` 列表，每次对话把之前的问答都带上：

```python
history = []

def chat(message):
    history.append({"role": "user", "content": message})
    response = model.chat(history)
    history.append({"role": "assistant", "content": response})
    return response
```

### 前端推荐
- **Gradio**：`gr.ChatInterface` 几行代码就能搭出聊天界面
- **Streamlit**：`st.chat_message` + `st.chat_input` 也很方便

---

## 任务八：前端界面开发（5%）

### 目标
做一个美观易用的 Web 界面，把前面的模型服务、RAG、对话功能都集成进去。

### 推荐组合
- **后端**：FastAPI（性能好，自带 API 文档）
- **前端**：Gradio（最适合模型演示，上手极快）

> Flask 也可以做后端，但 FastAPI 更现代、更快。
> Streamlit 也可以做前端，但 Gradio 对模型场景更友好。

---

## 任务九：工具调用智能体（10%）

### 目标
让模型能调用外部工具（查日志、查知识库、查告警系统等）。

### 核心思路
1. 定义工具函数和它们的描述（JSON Schema）
2. 模型根据用户问题，决定调用哪个工具
3. 执行工具，拿到结果
4. 模型根据工具结果生成最终回答

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "query_log",
            "description": "查询设备运行日志",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_id": {"type": "string", "description": "设备编号"},
                    "time_range": {"type": "string", "description": "时间范围"}
                },
                "required": ["device_id"]
            }
        }
    }
]
```

---

## 任务十：多智能体协作（10%）

### 目标
设计多个智能体分工协作，完成复杂任务。

### 常见角色
- **规划智能体（Planner）**：分析任务，拆解成子任务
- **执行智能体（Executor）**：执行具体操作
- **质检智能体（Checker）**：检查执行结果是否合格
- **路由智能体（Router）**：根据用户意图分发到不同处理流程

---

## 任务十一：跨场景智能体编排（15%）

### 目标
让智能体能自动识别用户意图，跨不同业务场景调度。

### 核心思路
```
用户输入 → 意图识别（路由智能体）→ 分发到对应场景
                                      ├── 质检场景 → 质检智能体
                                      ├── 运维场景 → 运维智能体
                                      └── 客服场景 → 客服智能体
```

---

## 任务十二：安全控制与综合展示（10%）

### 目标
给系统加上安全防护，并完成最终展示。

### 安全要点
- **敏感指令检测**：过滤注入攻击、越权请求
- **权限控制**：不同角色看到不同内容
- **审计日志**：记录所有操作，方便追溯

### 综合展示
准备答辩 PPT，重点讲：
1. 系统架构（画个架构图）
2. 每个模块怎么实现的
3. 遇到了什么问题、怎么解决的
4. 演示系统功能

---

## Mac 实操进度记录

### 7月4日 — 环境搭建

- [x] 安装 Miniconda（conda 26.3.2）
  - 下载地址：`https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh`
  - 安装命令：`bash /tmp/miniconda.sh -b -p $HOME/miniconda3`
  - 初始化：`$HOME/miniconda3/bin/conda init zsh`
- [x] 创建虚拟环境 `robocom`（Python 3.10）
  - `conda create -n robocom python=3.10 -y`
  - `conda activate robocom`
- [x] 建立项目目录结构 `competition_project/`
- [x] 安装 PyTorch CPU 版
- [x] 安装 transformers, accelerate, sentencepiece, protobuf, fastapi, uvicorn, peft, trl, datasets, gradio, faiss-cpu
- [x] 了解 Git 空文件夹问题（需要 .gitkeep）

### 7月4日 — 模型下载

- [x] 下载 Qwen2-0.5B（Hugging Face 版本）
  - 使用国内镜像：`HF_ENDPOINT=https://hf-mirror.com`
  - 模型缓存位置：`~/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B`
- [x] 已有 bge-small-zh-v1.5（Embedding 模型，用于 RAG 向量化）

### 7月4日 — FastAPI 服务开发

- [x] 跑通 FastAPI Hello World
- [ ] 在 FastAPI 中加载 Qwen2-0.5B 模型
- [ ] 实现 /chat 接口（接收消息 → 模型推理 → 返回回答）

### 关键踩坑记录

1. **pip 安装时包名之间要有空格**：`trl gradio` 不能写成 `trlgra`
2. **Hugging Face 直连会 SSL 报错**：需要加 `HF_ENDPOINT=https://hf-mirror.com` 走国内镜像
3. **Git 不跟踪空文件夹**：需要放 `.gitkeep` 文件
4. **Ollama 模型格式不通用**：Ollama 是 GGUF 格式，transformers 需要 Hugging Face 格式，不能混用

## Windows 实战待办

- [x] 安装 Miniconda / Anaconda
- [x] 创建 conda 环境，安装 GPU 版 PyTorch
- [x] 验证 GPU 可用（`torch.cuda.is_available()`）
- [ ] 下载比赛指定模型
- [ ] 用 FastAPI 部署模型服务
- [ ] 准备数据并清洗脱敏
- [ ] LoRA 微调
- [ ] 设计提示词
- [ ] 搭建 RAG 知识库
- [ ] 实现多轮对话
- [ ] 开发前端界面
- [ ] 实现工具调用智能体
- [ ] 实现多智能体协作
- [ ] 实现跨场景编排
- [ ] 添加安全控制
- [ ] 准备答辩材料

---

## Windows 环境搭建记录（2026-07-04）

### 1. 检查 Miniconda
- 确认 Windows 上未安装 conda

### 2. 下载安装 Miniconda
- 下载 Miniconda3-latest-Windows-x86_64.exe（约 95MB）
- 静默安装至 `C:\Users\xytgy\miniconda3`
- 安装版本：conda 26.3.2，与 Mac 端一致
- 接受了 conda 服务条款（pkgs/main、pkgs/r、pkgs/msys2）

### 3. 检查 GPU 环境
- GPU：NVIDIA GeForce RTX 4070 Laptop，8GB 显存
- 驱动版本：610.62
- CUDA UMD Version：13.3

### 4. 创建虚拟环境
- 环境名：robocom
- Python 版本：3.10.20
- 路径：`C:\Users\xytgy\miniconda3\envs\robocom`

### 5. 安装依赖（进行中）
- PyTorch GPU 版安装遇到问题：`cu121` 索引无法找到包
- 尝试方案：升级 pip 后换用 `cu124` 索引，或直接安装默认版本

### Windows 当前待完成
- [ ] PyTorch GPU 版安装成功
- [ ] 安装 transformers, accelerate, sentencepiece, protobuf, fastapi, uvicorn, peft, trl, datasets, gradio, faiss-cpu
- [ ] 验证 GPU 可用（`torch.cuda.is_available()`）
- [ ] 建立项目目录结构
- [ ] 导出 requirements.txt

---

## 附录：FastAPI 开发指南（Java 工程师视角）

> 用 Java/Spring Boot 的概念类比，帮助理解 Python/FastAPI。

### 概念对照

| Java / Spring Boot | Python / FastAPI |
|---|---|
| `@RestController` | `app = FastAPI()` |
| `@PostMapping("/chat")` | `@app.post("/chat")` |
| `@RequestBody ChatRequest req` | `class ChatRequest(BaseModel): message: str` |
| `return ResponseEntity.ok(result)` | `return {"response": result}` |
| `java -jar app.jar` 启动 | `uvicorn main:app --reload` 启动 |
| Maven `pom.xml` | `requirements.txt` |
| Spring Bean 初始化 | 模块顶部加载模型 |

### 开发步骤

1. **创建 `main.py`**：相当于创建 Spring Boot 主类
2. **加载模型**：相当于 `@Bean` 初始化，用 `AutoTokenizer` 和 `AutoModelForCausalLM`
3. **定义请求体**：相当于写一个 DTO 类，继承 `BaseModel`
4. **写接口**：用 `@app.post("/chat")` 定义，接收请求体，返回 JSON
5. **模型推理**：`tokenizer.encode()` → `model.generate()` → `tokenizer.decode()`
6. **启动服务**：`uvicorn main:app --host 0.0.0.0 --port 8000`

### 模型推理三件套

```
tokenizer(text, return_tensors="pt")   # 文字 → 数字（模型能懂的）
model.generate(**inputs, max_new_tokens=100)  # 模型生成回答
tokenizer.decode(output, skip_special_tokens=True)  # 数字 → 文字
```
