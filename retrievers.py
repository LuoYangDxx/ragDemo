import logging
from typing import List, Dict, Tuple, Optional
import pymysql
from pymysql.cursors import DictCursor
from pymysql.pool import Pool
from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer, CrossEncoder
from tenacity import retry, stop_after_attempt, wait_exponential
from rank_bm25 import BM25Okapi
import jieba
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ---------- 1. Embedding Model ----------
class EmbeddingModel:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def encode(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


# ---------- 2. Reranker ----------
class Reranker:
    def __init__(self, model_name: str):
        self.cross_encoder = CrossEncoder(model_name)

    def rerank(self, query: str, passages: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        scores = self.cross_encoder.predict(pairs)
        scored = list(zip(passages, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ---------- 3. Milvus Client ----------
class MilvusClientWrapper:
    def __init__(self, host: str, port: str, user: str = "", password: str = "", secure: bool = False):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.secure = secure
        self._connect()

    def _connect(self):
        connections.connect(
            alias="default",
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            secure=self.secure
        )

    def get_collection(self, name: str) -> Collection:
        if not utility.has_collection(name):
            raise ValueError(f"Collection {name} does not exist")
        return Collection(name)


# ---------- 4. FAQ Retriever (Milvus + MySQL) ----------
class FAQRetriever:
    def __init__(self, milvus_client: MilvusClientWrapper, mysql_pool: Pool,
                 embedding_model: EmbeddingModel, collection_name: str):
        self.milvus = milvus_client
        self.collection = milvus_client.get_collection(collection_name)
        self.mysql_pool = mysql_pool
        self.embed_model = embedding_model

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        query_vec = self.embed_model.encode([query])[0]
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_vec],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["faq_id"]
        )
        if not results[0]:
            return []
        faq_ids = [hit.entity.get('faq_id') for hit in results[0]]
        with self.mysql_pool.connection() as conn:
            with conn.cursor(DictCursor) as cursor:
                placeholders = ','.join(['%s'] * len(faq_ids))
                cursor.execute(
                    f"SELECT id, question, answer, source FROM faq WHERE id IN ({placeholders})",
                    faq_ids
                )
                rows = cursor.fetchall()
        id_to_score = {hit.entity.get('faq_id'): hit.score for hit in results[0]}
        for row in rows:
            row['score'] = id_to_score.get(row['id'], 0.0)
        rows.sort(key=lambda x: x['score'], reverse=True)
        return rows


# ---------- 5. Knowledge Retriever (Milvus only) ----------
class KnowledgeRetriever:
    def __init__(self, milvus_client: MilvusClientWrapper, embedding_model: EmbeddingModel,
                 collection_name: str):
        self.milvus = milvus_client
        self.collection = milvus_client.get_collection(collection_name)
        self.embed_model = embedding_model

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_vec = self.embed_model.encode([query])[0]
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_vec],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text", "source"]
        )
        hits = []
        for hit in results[0]:
            hits.append({
                "text": hit.entity.get('text'),
                "source": hit.entity.get('source'),
                "score": hit.score
            })
        return hits

    def get_all_documents(self, limit: int = 100000) -> List[str]:
        """获取知识库中所有文档文本（用于构建 BM25 索引）"""
        results = self.collection.query(expr="id >= 0", output_fields=["text"], limit=limit)
        return [res['text'] for res in results]


# ---------- 6. BM25 Retriever (memory-based) ----------
class BM25Retriever:
    """基于内存的 BM25 检索器（rank-bm25 + jieba 分词）"""
    def __init__(self, corpus: List[str], tokenizer=None):
        self.tokenizer = tokenizer or (lambda x: list(jieba.cut(x)))
        tokenized_corpus = [self.tokenizer(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.corpus = corpus

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        tokenized_query = self.tokenizer(query)
        scores = self.bm25.get_scores(tokenized_query)
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = indexed_scores[:top_k]
        return [(self.corpus[idx], score) for idx, score in top_indices]


# ---------- 7. Retrieval Pipeline with Hybrid Search (Dense + BM25 + RRF) ----------
class RetrievalPipeline:
    def __init__(self, faq_retriever: FAQRetriever, knowledge_retriever: KnowledgeRetriever,
                 reranker: Reranker, enable_bm25: bool = False, bm25_retriever: BM25Retriever = None):
        self.faq_retriever = faq_retriever
        self.knowledge_retriever = knowledge_retriever
        self.reranker = reranker
        self.enable_bm25 = enable_bm25
        self.bm25_retriever = bm25_retriever

    @staticmethod
    def _rrf_fusion(ranked_lists: List[List[Tuple[str, float]]], k: int = 60) -> List[Tuple[str, float]]:
        """
        RRF融合多个排序列表
        ranked_lists: 每个列表为 [(doc_text, score), ...] 按分数降序
        k: RRF常数
        """
        scores = {}
        for rank_list in ranked_lists:
            for rank, (doc, _) in enumerate(rank_list):
                if doc not in scores:
                    scores[doc] = 0.0
                scores[doc] += 1.0 / (k + rank + 1)
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs

    def get_faq_answer(self, query: str) -> Tuple[Optional[str], List[str]]:
        faq_candidates = self.faq_retriever.search(query, top_k=3)
        if not faq_candidates:
            return None, []
        passages = [f"{c['question']} {c['answer']}" for c in faq_candidates]
        reranked = self.reranker.rerank(query, passages, top_k=1)
        if not reranked:
            return None, []
        best_text = reranked[0][0]
        best_candidate = next(c for c in faq_candidates if f"{c['question']} {c['answer']}" == best_text)
        return best_candidate['answer'], [best_candidate['source']]

    def get_knowledge_context(self, query: str, top_k: int = 3) -> Tuple[str, List[str]]:
        """
        旧版接口：返回拼接的上下文字符串和来源列表
        保留用于兼容旧代码
        """
        docs_with_scores = self.get_knowledge_docs_with_scores(query, top_k)
        if not docs_with_scores:
            return "", []
        context_parts = []
        sources = []
        for doc, score in docs_with_scores:
            sources.append(doc.metadata.get("source", "未知来源"))
            context_parts.append(f"[来源:{doc.metadata.get('source', '未知来源')}] {doc.page_content}")
        return "\n\n".join(context_parts), sources

    def get_knowledge_docs_with_scores(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """
        返回经过 RRF + Rerank 后的文档列表及 rerank 分数
        每个元素为 (Document, rerank_score)
        """
        # 1. 稠密向量检索
        dense_docs = self.knowledge_retriever.search(query, top_k=top_k * 2)  # 获取更多候选
        dense_list = [(d['text'], d['score']) for d in dense_docs]

        # 2. BM25 检索（如果启用）
        bm25_list = []
        if self.enable_bm25 and self.bm25_retriever:
            bm25_results = self.bm25_retriever.search(query, top_k=top_k * 2)
            bm25_list = [(text, score) for text, score in bm25_results]

        # 3. RRF 融合
        if self.enable_bm25 and bm25_list:
            fused = self._rrf_fusion([dense_list, bm25_list])
            candidate_texts = [doc for doc, _ in fused[:top_k * 2]]
        else:
            candidate_texts = [d['text'] for d in dense_docs[:top_k * 2]]

        if not candidate_texts:
            return []

        # 4. Rerank 获取相关性分数
        reranked = self.reranker.rerank(query, candidate_texts, top_k=top_k)  # [(text, score)]

        # 5. 构建文本到 source 的映射
        text_to_source = {d['text']: d['source'] for d in dense_docs}
        if self.enable_bm25 and self.bm25_retriever:
            for text, _ in bm25_list:
                if text not in text_to_source:
                    text_to_source[text] = "知识库(BM25)"

        # 6. 构建 Document 对象列表
        result = []
        for text, rerank_score in reranked:
            source = text_to_source.get(text, "未知来源")
            doc = Document(
                page_content=text,
                metadata={"source": source, "rerank_score": rerank_score}
            )
            result.append((doc, rerank_score))
        return result