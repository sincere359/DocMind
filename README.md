# DocMind 📄 — AI 文档智能助手

> 基于 MiMo 大模型 + RAG 技术的企业级文档智能分析平台

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![MiMo V2.5](https://img.shields.io/badge/MiMo-V2.5-green.svg)](https://xiaomimimo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 项目定位

DocMind 是一款面向企业和个人用户的 **AI 文档智能助手**，通过 RAG（检索增强生成）技术，让用户能够与文档"对话"，实现智能问答、自动摘要、关键信息提取和多文档对比分析。

**核心价值：将静态文档变为可交互的动态知识库。**

### 市场对标

| 产品 | 定位 | DocMind 差异 |
|------|------|-------------|
| ChatPDF | PDF 问答 | 多格式 + 中文优化 + 信息提取 |
| 智谱清言 | 通用 AI | 专注文档场景，深度 RAG |
| Kimi | 长文本 | 本地部署 + 多文档对比 |

---

## ✨ 核心功能

| 功能 | 描述 | MiMo 模型 |
|------|------|-----------|
| 💬 **智能问答** | 基于文档内容精准回答，标注来源 | mimo-v2.5-pro (推理) |
| 📝 **文档摘要** | 自动生成多风格摘要（简短/详细/学术） | mimo-v2.5-pro |
| 🔍 **信息提取** | 提取实体/数据/结论/风险/建议 | mimo-v2.5-pro |
| 📊 **文档对比** | 多文档异同点对比分析 | mimo-v2.5-pro |
| ⚡ **流式输出** | 实时 token 流式响应 | mimo-v2.5-pro |

---

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────────┐
│                    Streamlit UI                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │ 智能问答  │ │ 文档摘要  │ │ 信息提取  │ │文档对比│ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └───┬────┘ │
└───────┼─────────────┼────────────┼────────────┼──────┘
        │             │            │            │
┌───────▼─────────────▼────────────▼────────────▼──────┐
│                    RAG Engine                        │
│  ┌──────────────┐  ┌───────────────┐                │
│  │   Retriever   │  │   Generator   │                │
│  │  (FAISS+Emb)  │  │ (MiMo V2.5)   │                │
│  └──────┬───────┘  └───────┬───────┘                │
│         │                  │                         │
│  ┌──────▼───────┐  ┌──────▼───────┐                │
│  │ Vector Store  │  │ MiMo Client  │                │
│  │  (FAISS IP)   │  │ (OpenAI SDK) │                │
│  └──────────────┘  └──────────────┘                │
└─────────────────────────────────────────────────────┘
        │                  │
┌───────▼──────┐  ┌───────▼──────────────┐
│  Embedding   │  │  MiMo Token Plan API │
│ text2vec-    │  │  mimo-v2.5-pro       │
│ base-chinese │  │  mimo-v2.5           │
└──────────────┘  └──────────────────────┘
```

### 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 前端 | Streamlit | 快速构建交互式 Web UI |
| RAG 引擎 | 自研 | 检索增强生成核心 |
| 向量化 | sentence-transformers | 中文文本嵌入 (text2vec-base-chinese) |
| 向量存储 | FAISS (IndexFlatIP) | 高效余弦相似度检索 |
| LLM | MiMo V2.5 Pro | 深度推理，Token Plan API |
| 文档解析 | pdfplumber + python-docx | PDF/TXT/MD/DOCX 四格式 |
| SDK | OpenAI Python SDK | 兼容 MiMo API |

---

## 🚀 快速开始

### 1. 安装依赖

```bash
git clone https://github.com/sincere359/DocMind.git
cd DocMind
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入 MiMo Token Plan API Key
```

### 3. 启动服务

```bash
streamlit run docmind/ui.py
```

浏览器打开 `http://localhost:8501` 即可使用。

---

## 📖 使用流程

```
上传文档 → 自动解析分块 → 向量化索引 → 智能问答/摘要/提取/对比
```

1. **上传文档**：侧边栏上传 PDF/TXT/MD/DOCX 文件
2. **索引文档**：点击"索引文档"按钮，自动解析分块并建立向量索引
3. **智能问答**：基于文档内容提问，AI 精准回答并标注来源
4. **文档摘要**：一键生成详细/简短/学术风格摘要
5. **信息提取**：自动提取实体、数据、结论、风险、建议
6. **文档对比**：选择两个文档，AI 分析异同

---

## 🔬 RAG 技术细节

### 分块策略
- **滑动窗口分块**：chunk_size=500 字符，overlap=50 字符
- **句子边界切分**：优先在 `。！？\n` 处切分，避免割裂语义
- **元数据保留**：每个 chunk 记录来源文件、索引位置、字符范围

### 检索策略
- **向量化**：text2vec-base-chinese（768 维），L2 归一化后等价余弦相似度
- **FAISS IndexFlatIP**：精确内积检索，保证召回质量
- **Top-K 检索**：默认返回 Top-5 最相关片段

### 生成策略
- **上下文注入**：将检索到的 Top-K 片段作为参考文档注入 Prompt
- **来源标注**：要求 AI 回答时标注引用的片段编号
- **幻觉抑制**：系统提示约束"严格基于文档内容，不编造"
- **流式输出**：支持逐 token 流式响应，提升交互体验

### 长文档处理
- **分层摘要**：超过 8000 字符时，先分块摘要再合并
- **分段提取**：超过 6000 字符时，分段提取信息再去重合并

---

## 💰 Token 消耗估算

| 操作 | 单次 Token 消耗 | 日均使用(10次) |
|------|-----------------|---------------|
| 智能问答 | ~2000 tokens | ~20,000 |
| 文档摘要 | ~3000 tokens | ~30,000 |
| 信息提取 | ~2500 tokens | ~25,000 |
| 文档对比 | ~4000 tokens | ~40,000 |
| **日均合计** | | **~115,000 tokens** |
| **月均合计** | | **~3,450,000 tokens** |

> 随着用户增长，Token 消耗线性增长。100 活跃用户月消耗约 **3.45 亿 tokens**，适合 Pro 档位（7亿Credits/月）。

---

## 📁 项目结构

```
DocMind/
├── docmind/
│   ├── __init__.py          # 包初始化
│   ├── __main__.py          # 入口
│   ├── config.py            # 全局配置
│   ├── mimo_client.py       # MiMo API 客户端
│   ├── document_parser.py   # 文档解析 + 分块
│   ├── embeddings.py        # 文本向量化
│   ├── vector_store.py      # FAISS 向量存储
│   ├── rag_engine.py        # RAG 检索增强生成
│   ├── summarizer.py        # 文档摘要生成
│   ├── extractor.py         # 关键信息提取
│   └── ui.py                # Streamlit 前端
├── tests/
│   └── test_core.py         # 集成测试
├── data/                    # 数据目录(gitignored)
├── .env.example             # 环境变量模板
├── requirements.txt         # 依赖列表
└── README.md                # 本文件
```

---

## 🎯 目标用户

| 用户群 | 使用场景 |
|--------|---------|
| 企业知识管理 | 内部文档智能问答，替代传统搜索 |
| 学术研究者 | 论文阅读辅助，快速提取关键信息 |
| 法律从业者 | 合同条款对比，风险点识别 |
| 金融分析师 | 报告数据提取，多报告对比 |
| 学生 | 论文/教材阅读辅助，知识点提取 |

---

## 📈 后续规划

- [ ] 支持 Excel/PPT 文件格式
- [ ] 多轮对话记忆
- [ ] 文档知识图谱可视化
- [ ] API 服务化（FastAPI）
- [ ] Docker 一键部署
- [ ] 用户认证和权限管理

---

## 📄 License

MIT License

---

**Powered by [MiMo](https://xiaomimimo.com) — 小米自研大模型**
