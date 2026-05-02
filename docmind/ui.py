"""DocMind Streamlit 前端界面"""
from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

import streamlit as st

from docmind.config import Config
from docmind.document_parser import parse_document
from docmind.vector_store import VectorStore
from docmind.rag_engine import RAGEngine
from docmind.summarizer import Summarizer
from docmind.extractor import Extractor

logger = logging.getLogger(__name__)

# ── 页面配置 ──

st.set_page_config(
    page_title="DocMind - AI 文档智能助手",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 自定义样式 ──

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .source-badge {
        display: inline-block;
        background: #e8f0fe;
        color: #1a73e8;
        padding: 0.2rem 0.6rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    .chat-message-user {
        background: #f0f4ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .chat-message-assistant {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State 初始化 ──

def init_state():
    """初始化会话状态"""
    if "rag_engine" not in st.session_state:
        store = VectorStore()
        # 尝试加载已有索引
        store.load()
        st.session_state.rag_engine = RAGEngine(store)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "summaries" not in st.session_state:
        st.session_state.summaries = {}
    if "extractions" not in st.session_state:
        st.session_state.extractions = {}


init_state()


# ── 侧边栏 ──

with st.sidebar:
    st.markdown('<p class="main-header">📄 DocMind</p>', unsafe_allow_html=True)
    st.caption("AI 文档智能助手 · Powered by MiMo")

    st.divider()

    # 文档上传
    st.subheader("📤 上传文档")
    uploaded_files = st.file_uploader(
        "支持 PDF / TXT / MD / DOCX",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        help="可同时上传多个文档",
    )

    if uploaded_files:
        if st.button("🔄 索引文档", use_container_width=True, type="primary"):
            with st.spinner("正在解析和索引文档..."):
                progress = st.progress(0)
                total_chunks = 0

                for i, file in enumerate(uploaded_files):
                    # 保存上传文件到临时目录
                    suffix = Path(file.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(Config.UPLOAD_DIR)) as tmp:
                        tmp.write(file.getbuffer())
                        tmp_path = tmp.name

                    try:
                        chunks = parse_document(tmp_path)
                        st.session_state.rag_engine.store.add_chunks(chunks)
                        total_chunks += len(chunks)
                        st.session_state.uploaded_files.append(file.name)
                    except Exception as e:
                        st.error(f"解析失败 {file.name}: {e}")

                    progress.progress((i + 1) / len(uploaded_files))

                st.session_state.rag_engine.store.save()
                st.success(f"✅ 索引完成！共 {total_chunks} 个文本块")

    # 已索引文档
    st.divider()
    st.subheader("📚 已索引文档")
    engine = st.session_state.rag_engine
    if engine.store.is_empty:
        st.info("暂无文档，请先上传")
    else:
        st.metric("索引文本块", engine.store.total_chunks)
        sources = list(set(c.source for c in engine.store.chunks))
        for src in sources:
            chunk_count = sum(1 for c in engine.store.chunks if c.source == src)
            st.markdown(f"- **{src}** ({chunk_count} 块)")

    # 功能选项
    st.divider()
    st.subheader("🛠️ 工具箱")
    tool = st.radio(
        "选择功能",
        ["💬 智能问答", "📝 文档摘要", "🔍 信息提取", "📊 文档对比"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(f"模型: {Config.MIMO_MODEL_PRO} / {Config.MIMO_MODEL_FAST}")
    st.caption(f"Embedding: {Config.EMBEDDING_MODEL}")


# ── 主内容区 ──

tool_name = tool.split(" ", 1)[1]

# ── 智能问答 ──

if tool_name == "智能问答":
    st.header("💬 智能问答")

    if engine.store.is_empty:
        st.warning("请先在侧边栏上传文档并索引，才能进行问答。")
    else:
        # 显示聊天历史
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # 输入框
        if query := st.chat_input("基于文档内容提问..."):
            # 用户消息
            st.session_state.chat_history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            # AI 回复（流式）
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""

                for token in engine.ask_stream(query):
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)

            st.session_state.chat_history.append({"role": "assistant", "content": full_response})

# ── 文档摘要 ──

elif tool_name == "文档摘要":
    st.header("📝 文档摘要")

    if engine.store.is_empty:
        st.warning("请先上传文档。")
    else:
        col1, col2 = st.columns([1, 2])

        with col1:
            style = st.selectbox("摘要风格", ["详细 (detailed)", "简短 (brief)", "学术 (academic)"])
            style_key = style.split("(")[1].rstrip(")")

            sources = list(set(c.source for c in engine.store.chunks))
            selected_source = st.selectbox("选择文档", sources)

            if st.button("生成摘要", type="primary", use_container_width=True):
                with st.spinner("正在生成摘要..."):
                    chunks = [c for c in engine.store.chunks if c.source == selected_source]
                    summarizer = Summarizer()
                    result = summarizer.summarize(chunks, style=style_key)
                    st.session_state.summaries[selected_source] = result

        with col2:
            if selected_source in st.session_state.summaries:
                result = st.session_state.summaries[selected_source]

                st.subheader(result.get("title", "文档摘要"))
                st.markdown(result.get("summary", ""))

                if result.get("key_points"):
                    st.subheader("🔑 关键要点")
                    for pt in result["key_points"]:
                        st.markdown(f"- {pt}")

                if result.get("keywords"):
                    st.subheader("🏷️ 关键词")
                    st.markdown(" | ".join([f"`{k}`" for k in result["keywords"]]))

                if result.get("doc_type"):
                    st.caption(f"文档类型: {result['doc_type']}")

# ── 信息提取 ──

elif tool_name == "信息提取":
    st.header("🔍 关键信息提取")

    if engine.store.is_empty:
        st.warning("请先上传文档。")
    else:
        sources = list(set(c.source for c in engine.store.chunks))
        selected_source = st.selectbox("选择文档", sources, key="extract_source")

        if st.button("提取信息", type="primary"):
            with st.spinner("正在提取关键信息..."):
                chunks = [c for c in engine.store.chunks if c.source == selected_source]
                extractor = Extractor()
                result = extractor.extract(chunks)
                st.session_state.extractions[selected_source] = result

        if selected_source in st.session_state.extractions:
            result = st.session_state.extractions[selected_source]

            # 实体
            entities = result.get("entities", {})
            if any(entities.get(k) for k in ["people", "organizations", "locations", "dates"]):
                st.subheader("👤 实体")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown("**人物**")
                    for p in entities.get("people", [])[:5]:
                        st.markdown(f"- {p}")
                with col2:
                    st.markdown("**机构**")
                    for o in entities.get("organizations", [])[:5]:
                        st.markdown(f"- {o}")
                with col3:
                    st.markdown("**地点**")
                    for l in entities.get("locations", [])[:5]:
                        st.markdown(f"- {l}")
                with col4:
                    st.markdown("**日期**")
                    for d in entities.get("dates", [])[:5]:
                        st.markdown(f"- {d}")

            # 数据
            numbers = result.get("numbers", [])
            if numbers:
                st.subheader("📊 关键数据")
                for n in numbers[:10]:
                    st.markdown(f"- **{n.get('value', '')}** {n.get('unit', '')} — {n.get('context', '')}")

            # 结论
            conclusions = result.get("conclusions", [])
            if conclusions:
                st.subheader("💡 核心结论")
                for c in conclusions:
                    st.markdown(f"- {c}")

            # 风险
            risks = result.get("risks", [])
            if risks:
                st.subheader("⚠️ 风险与挑战")
                for r in risks:
                    st.markdown(f"- {r}")

            # 建议
            recommendations = result.get("recommendations", [])
            if recommendations:
                st.subheader("✅ 建议")
                for r in recommendations:
                    st.markdown(f"- {r}")

# ── 文档对比 ──

elif tool_name == "文档对比":
    st.header("📊 文档对比")

    if engine.store.is_empty:
        st.warning("请先上传至少两个文档。")
    else:
        sources = list(set(c.source for c in engine.store.chunks))
        if len(sources) < 2:
            st.warning("至少需要 2 个文档才能进行对比。")
        else:
            col1, col2 = st.columns(2)
            with col1:
                doc_a = st.selectbox("文档 A", sources, key="compare_a")
            with col2:
                doc_b = st.selectbox("文档 B", [s for s in sources if s != doc_a], key="compare_b")

            if st.button("开始对比", type="primary"):
                with st.spinner("正在对比分析..."):
                    chunks_a = [c for c in engine.store.chunks if c.source == doc_a]
                    chunks_b = [c for c in engine.store.chunks if c.source == doc_b]
                    extractor = Extractor()
                    result = extractor.compare_documents(chunks_a, chunks_b, doc_a, doc_b)

                    # 显示结果
                    st.subheader("🤝 共同点")
                    for cp in result.get("common_points", []):
                        st.markdown(f"- {cp}")

                    st.subheader("🔄 差异")
                    diffs = result.get("differences", [])
                    if diffs:
                        st.table(diffs)

                    st.subheader("📋 结论")
                    st.markdown(result.get("conclusion", ""))


# ── 入口 ──

def main():
    """Streamlit 入口"""
    pass  # Streamlit 自动执行模块级代码


if __name__ == "__main__":
    main()
