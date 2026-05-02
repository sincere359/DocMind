"""DocMind 集成测试"""
import os
import sys
import tempfile
from pathlib import Path

# 确保项目根目录在 sys.path 中
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_config():
    """测试配置加载"""
    from docmind.config import Config
    assert Config.MIMO_API_KEY, "MIMO_API_KEY 未配置"
    assert Config.MIMO_BASE_URL, "MIMO_BASE_URL 未配置"
    assert Config.CHUNK_SIZE > 0, "CHUNK_SIZE 应 > 0"
    print("✅ 配置加载正常")


def test_document_parser():
    """测试文档解析器"""
    from docmind.document_parser import DocumentParser

    parser = DocumentParser(chunk_size=200, chunk_overlap=20)

    # 测试 TXT 解析
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("这是第一段内容。数据结构是计算机科学的核心课程。栈和队列是两种基本的线性结构。\n\n")
        f.write("这是第二段内容。链表是一种动态数据结构，支持高效的插入和删除操作。二叉树则是层次化的数据组织方式。\n\n")
        f.write("这是第三段内容。排序算法包括冒泡排序、快速排序和归并排序。时间复杂度从O(n²)到O(nlogn)不等。")
        tmp_path = f.name

    try:
        chunks = parser.parse(tmp_path)
        assert len(chunks) > 0, "应至少产生一个分块"
        assert all(c.content.strip() for c in chunks), "分块内容不应为空"
        assert all(c.source for c in chunks), "分块应有来源信息"
        print(f"✅ 文档解析正常，产生 {len(chunks)} 个分块")
    finally:
        os.unlink(tmp_path)


def test_mimo_client():
    """测试 MiMo API 连通性"""
    from docmind.mimo_client import get_mimo_client

    client = get_mimo_client()
    response = client.fast_chat(
        system="你是一个测试助手。",
        user="请回复：连接成功",
    )
    assert response, "API 应返回非空响应"
    print(f"✅ MiMo API 连通，响应: {response[:50]}...")


def test_rag_engine_init():
    """测试 RAG 引擎初始化"""
    from docmind.rag_engine import RAGEngine
    from docmind.vector_store import VectorStore

    store = VectorStore()
    engine = RAGEngine(store)
    assert engine.store.is_empty, "初始状态应为空"
    print("✅ RAG 引擎初始化正常")


def test_summarizer_init():
    """测试摘要器初始化"""
    from docmind.summarizer import Summarizer

    summarizer = Summarizer()
    assert summarizer.client is not None
    print("✅ 摘要器初始化正常")


def test_extractor_init():
    """测试信息提取器初始化"""
    from docmind.extractor import Extractor

    extractor = Extractor()
    assert extractor.client is not None
    print("✅ 信息提取器初始化正常")


def test_vector_store():
    """测试向量存储（不依赖 embedding 模型）"""
    from docmind.vector_store import VectorStore
    from docmind.document_parser import Chunk

    store = VectorStore()
    assert store.is_empty, "初始应为空"
    assert store.total_chunks == 0, "初始分块数应为 0"
    print("✅ 向量存储初始化正常")


if __name__ == "__main__":
    print("=" * 50)
    print("DocMind 集成测试")
    print("=" * 50)

    tests = [
        ("配置加载", test_config),
        ("文档解析", test_document_parser),
        ("MiMo API 连通", test_mimo_client),
        ("RAG 引擎", test_rag_engine_init),
        ("摘要器", test_summarizer_init),
        ("信息提取器", test_extractor_init),
        ("向量存储", test_vector_store),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"❌ {name} 失败: {e}")
            failed += 1

    print("=" * 50)
    print(f"结果: {passed} 通过, {failed} 失败")
