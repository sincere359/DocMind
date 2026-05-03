"""DocMind Streamlit 前端 — 生产级 RAG 文档智能助手"""
from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

from docmind.config import Config
from docmind.document_parser import parse_document
from docmind.vector_store import VectorStore
from docmind.rag_engine import RAGEngine
from docmind.summarizer import Summarizer
from docmind.extractor import Extractor
from docmind.knowledge_graph import KnowledgeGraphBuilder
from docmind.chat_history import ChatHistoryManager
from docmind.exporter import Exporter
from docmind.auth import UserStore, create_access_token, verify_jwt, get_user_index_dir

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════
# 页面配置
# ══════════════════════════════════════════════

st.set_page_config(
    page_title="DocMind - AI 文档智能助手",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════
# 自定义样式
# ══════════════════════════════════════════════

st.markdown("""
<style>
    .main-header { font-size: 1.8rem; font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .doc-card { background: #f8f9fa; border-radius: 8px; padding: 0.8rem;
        margin-bottom: 0.5rem; border-left: 3px solid #667eea; }
    .doc-card-title { font-weight: 600; font-size: 0.95rem; color: #1a1a2e; }
    .doc-card-meta { font-size: 0.8rem; color: #666; }
    .source-badge { display: inline-block; background: #e8f0fe; color: #1a73e8;
        padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.85rem; margin: 0.1rem; }
    .score-badge { display: inline-block; background: #e6f4ea; color: #137333;
        padding: 0.1rem 0.4rem; border-radius: 4px; font-size: 0.8rem; }
    .risk-card { background: #fef7e0; border-left: 3px solid #f9ab00;
        padding: 0.6rem; border-radius: 4px; margin-bottom: 0.3rem; }
    .rec-card { background: #e6f4ea; border-left: 3px solid #137333;
        padding: 0.6rem; border-radius: 4px; margin-bottom: 0.3rem; }
    .keyword-tag { display: inline-block; background: #fce4ec; color: #c62828;
        padding: 0.15rem 0.5rem; border-radius: 12px; font-size: 0.8rem; margin: 0.1rem; }
    .welcome-icon { font-size: 4rem; text-align: center; }
    .stat-number { font-size: 1.5rem; font-weight: 700; color: #667eea; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# Session State
# ══════════════════════════════════════════════

def init_state():
    """初始化会话状态"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    if "rag_engine" not in st.session_state and st.session_state.authenticated:
        _init_user_engine()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "summaries" not in st.session_state:
        st.session_state.summaries = {}
    if "extractions" not in st.session_state:
        st.session_state.extractions = {}
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "top_k": Config.TOP_K,
            "temperature": 0.2,
        }


def _init_user_engine():
    """根据当前用户初始化 RAG 引擎（per-user FAISS 隔离）"""
    username = st.session_state.current_user
    if username:
        index_dir = get_user_index_dir(username)
        store = VectorStore()
        store.load(dir_path=index_dir)
    else:
        store = VectorStore()
        store.load()
    st.session_state.rag_engine = RAGEngine(store)


init_state()

# ══════════════════════════════════════════════
# 登录 / 注册页面
# ══════════════════════════════════════════════

def show_login_page():
    """显示登录/注册页面"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <div style="font-size: 4rem;">📄</div>
        <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            DocMind
        </h1>
        <p style="color: #666; font-size: 1.1rem;">AI 文档智能助手 · 请登录以继续</p>
    </div>
    """, unsafe_allow_html=True)

    tab_login, tab_register = st.tabs(["🔑 登录", "📝 注册"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("用户名", key="login_user")
            password = st.text_input("密码", type="password", key="login_pass")
            submitted = st.form_submit_button("登录", use_container_width=True, type="primary")

            if submitted:
                if not username or not password:
                    st.error("请输入用户名和密码")
                else:
                    user_store = UserStore()
                    user = user_store.verify_user(username, password)
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.current_user = username
                        _init_user_engine()
                        st.success(f"欢迎回来，{username}！")
                        st.rerun()
                    else:
                        st.error("用户名或密码错误")

    with tab_register:
        with st.form("register_form"):
            new_user = st.text_input("用户名", key="reg_user")
            new_pass = st.text_input("密码", type="password", key="reg_pass")
            new_pass2 = st.text_input("确认密码", type="password", key="reg_pass2")
            submitted = st.form_submit_button("注册", use_container_width=True)

            if submitted:
                if not new_user or not new_pass:
                    st.error("请输入用户名和密码")
                elif new_pass != new_pass2:
                    st.error("两次密码不一致")
                elif len(new_user) < 2:
                    st.error("用户名至少 2 个字符")
                elif len(new_pass) < 4:
                    st.error("密码至少 4 个字符")
                else:
                    user_store = UserStore()
                    user = user_store.create_user(new_user, new_pass)
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.current_user = new_user
                        _init_user_engine()
                        st.success(f"注册成功！欢迎，{new_user}！")
                        st.rerun()
                    else:
                        st.error("用户名已存在")


# 未登录则显示登录页，登录后才显示主界面
if not st.session_state.authenticated:
    show_login_page()
    st.stop()


# ══════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════

def get_engine() -> RAGEngine:
    return st.session_state.rag_engine


def _current_user_index_dir() -> Path | None:
    """获取当前用户的专属索引目录"""
    if st.session_state.current_user:
        return get_user_index_dir(st.session_state.current_user)
    return None


def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


# ══════════════════════════════════════════════
# 侧边栏
# ══════════════════════════════════════════════

with st.sidebar:
    # ── 标题 ──
    st.markdown('<p class="main-header">📄 DocMind</p>', unsafe_allow_html=True)
    st.caption("AI 文档智能助手 · Powered by MiMo")
    st.divider()

    # ── 文档上传 ──
    st.subheader("📤 上传文档")
    uploaded_files = st.file_uploader(
        "支持 PDF / TXT / MD / DOCX / XLSX / XLS / PPTX",
        type=["pdf", "txt", "md", "docx", "xlsx", "xls", "pptx"],
        accept_multiple_files=True,
        help="可同时上传多个文档，单文件最大 50MB",
        key="file_uploader",
    )

    if uploaded_files:
        if st.button("🔄 索引文档", use_container_width=True, type="primary"):
            engine = get_engine()
            new_count = 0
            skip_count = 0
            error_count = 0

            progress = st.progress(0, text="解析和索引文档...")

            for i, file in enumerate(uploaded_files):
                # 文件大小校验
                file_size = len(file.getbuffer())
                if file_size > Config.MAX_FILE_SIZE_MB * 1024 * 1024:
                    st.warning(f"⚠️ {file.name} 超过 {Config.MAX_FILE_SIZE_MB}MB 限制，已跳过")
                    error_count += 1
                    progress.progress((i + 1) / len(uploaded_files))
                    continue

                # 去重检查
                if engine.store.has_source(file.name):
                    skip_count += 1
                    progress.progress((i + 1) / len(uploaded_files))
                    continue

                # 保存临时文件
                suffix = Path(file.name).suffix
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(Config.UPLOAD_DIR)) as tmp:
                        tmp.write(file.getbuffer())
                        tmp_path = tmp.name

                    chunks = parse_document(tmp_path)
                    engine.store.add_chunks(chunks)
                    new_count += len(chunks)
                except Exception as e:
                    st.error(f"解析失败 {file.name}: {e}")
                    error_count += 1
                finally:
                    # 清理临时文件
                    if tmp_path and Config.TEMP_CLEANUP:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass

                progress.progress((i + 1) / len(uploaded_files))

            user_dir = _current_user_index_dir()
            engine.store.save(dir_path=user_dir)

            # 结果提示
            msg_parts = []
            if new_count > 0:
                msg_parts.append(f"新增 {new_count} 个文本块")
            if skip_count > 0:
                msg_parts.append(f"跳过 {skip_count} 个已索引文档")
            if error_count > 0:
                msg_parts.append(f"{error_count} 个文件失败")
            if msg_parts:
                st.success("✅ " + "，".join(msg_parts))

    # ── 文档库 ──
    st.divider()
    st.subheader("📚 文档库")
    engine = get_engine()

    if engine.store.is_empty:
        st.info("暂无文档，请先上传")
    else:
        st.metric("索引文本块", engine.store.total_chunks)
        stats = engine.store.get_document_stats()
        for doc_stat in stats:
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(
                        f'<div class="doc-card">'
                        f'<div class="doc-card-title">📄 {doc_stat.source}</div>'
                        f'<div class="doc-card-meta">{doc_stat.chunk_count} 块 · '
                        f'{doc_stat.total_chars:,} 字</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with col2:
                    if st.button("🗑️", key=f"del_{doc_stat.source}", help=f"删除 {doc_stat.source}"):
                        if engine.store.remove_source(doc_stat.source):
                            st.rerun()

        if st.button("🗑️ 清空全部", key="clear_all", use_container_width=True):
            engine.store.clear()
            st.session_state.chat_history = []
            st.rerun()

    # ── 设置 ──
    st.divider()
    with st.expander("⚙️ 参数设置"):
        settings = st.session_state.settings
        settings["top_k"] = st.slider("检索数量 (Top-K)", 1, 10, settings["top_k"], key="slider_topk")
        settings["temperature"] = st.slider(
            "生成温度", 0.0, 1.0, settings["temperature"], step=0.1, key="slider_temp"
        )
        if st.button("重置默认", key="reset_settings"):
            st.session_state.settings = {"top_k": Config.TOP_K, "temperature": 0.2}

    # ── 底部信息 ──
    st.divider()
    st.caption(f"👤 {st.session_state.current_user}")
    if st.button("🚪 退出登录", use_container_width=True, key="logout"):
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.session_state.rag_engine = None
        st.rerun()
    st.caption(f"模型: {Config.MIMO_MODEL_PRO}")
    st.caption(f"Embedding: {Config.EMBEDDING_MODEL.split('/')[-1]}")


# ══════════════════════════════════════════════
# 主内容区 — 4 个 Tab
# ══════════════════════════════════════════════

tab_qa, tab_summary, tab_extract, tab_compare, tab_graph = st.tabs(
    ["💬 智能问答", "📝 文档摘要", "🔍 信息提取", "📊 文档对比", "🕸️ 知识图谱"]
)

# ──────────────────────────────────────────────
# Tab 1: 智能问答
# ──────────────────────────────────────────────

with tab_qa:
    engine = get_engine()

    if engine.store.is_empty:
        # 欢迎页
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <div class="welcome-icon">📄</div>
            <h2>欢迎使用 DocMind</h2>
            <p style="color: #666; font-size: 1.1rem;">上传文档后，即可开始智能问答</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # 顶部工具栏
        col_toolbar1, col_toolbar2, col_toolbar3 = st.columns([3, 1, 1])
        with col_toolbar2:
            if st.button("🗑️ 清空对话", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()
        with col_toolbar3:
            # 导出对话
            if st.session_state.chat_history:
                exporter = Exporter()
                content, filename = exporter.chat_to_markdown(st.session_state.chat_history)
                st.download_button("📥 导出", content, filename, mime="text/markdown", key="export_chat")

        # 显示聊天历史
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                # 展示来源引用
                if msg["role"] == "assistant" and msg.get("sources"):
                    with st.expander(f"📎 参考来源 ({len(msg['sources'])} 个片段)"):
                        for j, src in enumerate(msg["sources"]):
                            st.markdown(
                                f"**片段 {j+1}** <span class='source-badge'>{src['source']}</span> "
                                f"<span class='score-badge'>相关度 {src['score']:.3f}</span>",
                                unsafe_allow_html=True,
                            )
                            st.caption(src["preview"])

        # 输入框
        if query := st.chat_input("基于文档内容提问..."):
            # 用户消息
            st.session_state.chat_history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            # AI 回复（流式 + 来源）
            with st.chat_message("assistant"):
                settings = st.session_state.settings
                stream_gen, sources = engine.ask_with_sources(
                    query,
                    top_k=settings["top_k"],
                    chat_history=st.session_state.chat_history[:-1],  # 排除刚加的用户消息
                )

                response_placeholder = st.empty()
                full_response = ""

                for token in stream_gen:
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)

                # 显示来源
                if sources:
                    with st.expander(f"📎 参考来源 ({len(sources)} 个片段)"):
                        for j, src in enumerate(sources):
                            st.markdown(
                                f"**片段 {j+1}** <span class='source-badge'>{src['source']}</span> "
                                f"<span class='score-badge'>相关度 {src['score']:.3f}</span>",
                                unsafe_allow_html=True,
                            )
                            st.caption(src["preview"])

            # 保存到聊天历史（含来源）
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources,
            })

            # 持久化对话
            try:
                history_mgr = ChatHistoryManager()
                conv_id = history_mgr.save_conversation(
                    st.session_state.chat_history,
                    st.session_state.conversation_id,
                )
                st.session_state.conversation_id = conv_id
            except Exception:
                pass  # 持久化失败不影响使用


# ──────────────────────────────────────────────
# Tab 2: 文档摘要
# ──────────────────────────────────────────────

with tab_summary:
    engine = get_engine()

    if engine.store.is_empty:
        st.info("请先上传文档。")
    else:
        col_s1, col_s2 = st.columns([1, 2])

        with col_s1:
            sources = engine.store.get_sources()
            selected_source = st.selectbox("选择文档", sources, key="summary_source")

            style = st.selectbox("摘要风格", ["详细", "简短", "学术"], key="summary_style")
            style_map = {"详细": "detailed", "简短": "brief", "学术": "academic"}
            style_key = style_map[style]

            if st.button("生成摘要", type="primary", use_container_width=True, key="gen_summary"):
                with st.spinner("正在生成摘要..."):
                    try:
                        chunks = [c for c in engine.store.chunks if c.source == selected_source]
                        summarizer = Summarizer()
                        result = summarizer.summarize(chunks, style=style_key)
                        st.session_state.summaries[selected_source] = result
                    except Exception as e:
                        st.error(f"摘要生成失败: {e}")

        with col_s2:
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
                    st.markdown(" ".join([f"`{k}`" for k in result["keywords"]]))

                if result.get("doc_type"):
                    st.caption(f"文档类型: {result['doc_type']}")

                # 导出
                st.divider()
                exporter = Exporter()
                col_e1, col_e2, col_e3 = st.columns(3)
                with col_e1:
                    json_content, json_name = exporter.to_json(result)
                    st.download_button("📥 JSON", json_content, json_name, "application/json", key="dl_summary_json")
                with col_e2:
                    txt_content, txt_name = exporter.to_text(result, f"{selected_source} 摘要")
                    st.download_button("📥 TXT", txt_content, txt_name, "text/plain", key="dl_summary_txt")
                with col_e3:
                    md_content, md_name = exporter.summary_to_markdown(result, selected_source)
                    st.download_button("📥 Markdown", md_content, md_name, "text/markdown", key="dl_summary_md")


# ──────────────────────────────────────────────
# Tab 3: 信息提取
# ──────────────────────────────────────────────

with tab_extract:
    engine = get_engine()

    if engine.store.is_empty:
        st.info("请先上传文档。")
    else:
        sources = engine.store.get_sources()
        selected_source = st.selectbox("选择文档", sources, key="extract_source")

        if st.button("提取信息", type="primary", key="gen_extract"):
            with st.spinner("正在提取关键信息..."):
                try:
                    chunks = [c for c in engine.store.chunks if c.source == selected_source]
                    extractor = Extractor()
                    result = extractor.extract(chunks)
                    st.session_state.extractions[selected_source] = result
                except Exception as e:
                    st.error(f"信息提取失败: {e}")

        if selected_source in st.session_state.extractions:
            result = st.session_state.extractions[selected_source]

            # 实体
            entities = result.get("entities", {})
            if any(entities.get(k) for k in ["people", "organizations", "locations", "dates"]):
                st.subheader("👤 实体")
                e_col1, e_col2, e_col3, e_col4 = st.columns(4)
                with e_col1:
                    st.markdown("**人物**")
                    for p in entities.get("people", [])[:8]:
                        st.markdown(f"- {p}")
                with e_col2:
                    st.markdown("**机构**")
                    for o in entities.get("organizations", [])[:8]:
                        st.markdown(f"- {o}")
                with e_col3:
                    st.markdown("**地点**")
                    for l in entities.get("locations", [])[:8]:
                        st.markdown(f"- {l}")
                with e_col4:
                    st.markdown("**日期**")
                    for d in entities.get("dates", [])[:8]:
                        st.markdown(f"- {d}")

            # 数据
            numbers = result.get("numbers", [])
            if numbers:
                st.subheader("📊 关键数据")
                for n in numbers[:15]:
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
                    st.markdown(f'<div class="risk-card">⚠️ {r}</div>', unsafe_allow_html=True)

            # 建议
            recommendations = result.get("recommendations", [])
            if recommendations:
                st.subheader("✅ 建议")
                for r in recommendations:
                    st.markdown(f'<div class="rec-card">✅ {r}</div>', unsafe_allow_html=True)

            # 导出
            st.divider()
            exporter = Exporter()
            col_ex1, col_ex2 = st.columns(2)
            with col_ex1:
                json_content, json_name = exporter.to_json(result)
                st.download_button("📥 JSON", json_content, json_name, "application/json", key="dl_extract_json")
            with col_ex2:
                txt_content, txt_name = exporter.to_text(result, f"{selected_source} 信息提取")
                st.download_button("📥 TXT", txt_content, txt_name, "text/plain", key="dl_extract_txt")


# ──────────────────────────────────────────────
# Tab 4: 文档对比
# ──────────────────────────────────────────────

with tab_compare:
    engine = get_engine()

    if engine.store.is_empty:
        st.info("请先上传文档。")
    else:
        sources = engine.store.get_sources()
        if len(sources) < 2:
            st.warning("至少需要 2 个文档才能进行对比。")
        else:
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                doc_a = st.selectbox("文档 A", sources, key="compare_doc_a")
            with c_col2:
                remaining = [s for s in sources if s != doc_a]
                doc_b = st.selectbox("文档 B", remaining or sources, key="compare_doc_b")

            if st.button("开始对比", type="primary", key="gen_compare"):
                with st.spinner("正在对比分析..."):
                    try:
                        chunks_a = [c for c in engine.store.chunks if c.source == doc_a]
                        chunks_b = [c for c in engine.store.chunks if c.source == doc_b]
                        extractor = Extractor()
                        result = extractor.compare_documents(chunks_a, chunks_b, doc_a, doc_b)
                        st.session_state.compare_result = result
                        st.session_state.compare_docs = (doc_a, doc_b)
                    except Exception as e:
                        st.error(f"对比分析失败: {e}")

            if "compare_result" in st.session_state:
                result = st.session_state.compare_result

                # 共同点
                common = result.get("common_points", [])
                if common:
                    st.subheader("🤝 共同点")
                    for cp in common:
                        st.markdown(f"- {cp}")

                # 差异
                diffs = result.get("differences", [])
                if diffs:
                    st.subheader("🔄 差异")
                    st.table(diffs)

                # 结论
                conclusion = result.get("conclusion", "")
                if conclusion:
                    st.subheader("📋 结论")
                    st.markdown(conclusion)

                # 导出
                st.divider()
                exporter = Exporter()
                json_content, json_name = exporter.to_json(result)
                st.download_button("📥 导出对比结果 (JSON)", json_content, json_name, "application/json", key="dl_compare")


# ──────────────────────────────────────────────
# Tab 5: 知识图谱
# ──────────────────────────────────────────────

with tab_graph:
    engine = get_engine()

    if engine.store.is_empty:
        st.info("请先上传文档。")
    else:
        sources = engine.store.get_sources()
        selected_source = st.selectbox(
            "选择文档", ["全部文档"] + sources, key="graph_source"
        )

        max_chunks = st.slider("最大分析块数", 5, 30, 15, key="graph_max_chunks",
                               help="块数越多图谱越完整，但耗时更长")

        if st.button("🕸️ 构建知识图谱", type="primary", use_container_width=True, key="gen_graph"):
            with st.spinner("正在构建知识图谱（LLM 提取实体与关系）..."):
                try:
                    if selected_source == "全部文档":
                        chunks = engine.store.chunks
                    else:
                        chunks = [c for c in engine.store.chunks if c.source == selected_source]

                    builder = KnowledgeGraphBuilder()
                    kg = builder.build(chunks, max_chunks=max_chunks)
                    st.session_state.knowledge_graph = kg
                except Exception as e:
                    st.error(f"图谱构建失败: {e}")

        if "knowledge_graph" in st.session_state:
            kg = st.session_state.knowledge_graph

            # 统计
            col_g1, col_g2, col_g3 = st.columns(3)
            with col_g1:
                st.metric("节点数", len(kg.nodes))
            with col_g2:
                st.metric("关系数", len(kg.edges))
            with col_g3:
                categories = set(n.category for n in kg.nodes)
                st.metric("实体类别", len(categories))

            # 图例
            if categories:
                legend_html = " ".join([
                    f'<span style="background:{_category_color_fn(c)};color:white;'
                    f'padding:0.15rem 0.5rem;border-radius:4px;font-size:0.8rem;margin:0.1rem;">{c}</span>'
                    for c in sorted(categories)
                ])
                st.markdown(f"**图例**: {legend_html}", unsafe_allow_html=True)

            # 渲染图谱
            if kg.nodes:
                try:
                    from streamlit_agraph import agraph, Config as AGraphConfig, Node, Edge

                    agraph_nodes = [
                        Node(
                            id=n.id,
                            label=n.label,
                            size=n.size,
                            color=n.color if hasattr(n, "color") else _category_color_fn(n.category),
                        )
                        for n in kg.nodes
                    ]
                    agraph_edges = [
                        Edge(source=e.source, target=e.target, label=e.label)
                        for e in kg.edges
                    ]

                    agraph_config = AGraphConfig(
                        width=900,
                        height=600,
                        directed=True,
                        physics=True,
                        hierarchical=False,
                    )

                    agraph(nodes=agraph_nodes, edges=agraph_edges, config=agraph_config)

                except ImportError:
                    # 降级：表格展示
                    st.warning("未安装 streamlit-agraph，以表格形式展示图谱")

                    st.subheader("节点列表")
                    node_data = [{"ID": n.id, "名称": n.label, "类别": n.category, "连接数": n.size} for n in kg.nodes]
                    st.dataframe(node_data, use_container_width=True)

                    st.subheader("关系列表")
                    edge_data = [{"源": e.source, "目标": e.target, "关系": e.label} for e in kg.edges]
                    st.dataframe(edge_data, use_container_width=True)
            else:
                st.info("未从文档中提取到实体和关系。")

            # 导出
            st.divider()
            exporter = Exporter()
            graph_data = kg.to_agraph_data()
            json_content, json_name = exporter.to_json(graph_data)
            st.download_button("📥 导出图谱数据 (JSON)", json_content, json_name, "application/json", key="dl_graph")


def _category_color_fn(category: str) -> str:
    """知识图谱节点颜色映射"""
    colors = {
        "person": "#e74c3c",
        "org": "#3498db",
        "organization": "#3498db",
        "location": "#2ecc71",
        "concept": "#9b59b6",
        "event": "#f39c12",
        "date": "#1abc9c",
        "default": "#95a5a6",
    }
    return colors.get(category.lower(), colors["default"])


# ══════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════

def main():
    pass


if __name__ == "__main__":
    main()
