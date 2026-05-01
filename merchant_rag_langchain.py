# merchant_rag_langchain.py
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import settings
from preprocess import QueryPreprocessor
from cache import L1Cache
from retrievers import (
    EmbeddingModel, Reranker, MilvusClientWrapper,
    FAQRetriever, KnowledgeRetriever, RetrievalPipeline,
    BM25Retriever
)
from tools.merchant_tools import MerchantTools
from langchain_wrapper import create_merchant_tools, MilvusRetrieverWrapper, build_rag_chain
from router import IntentRouter, Intent, ToolCall

import logging
import time
from typing import List, Tuple, Optional, Callable

logger = logging.getLogger(__name__)


# ---------- EnhancedRetriever with MMR (no duplicate RRF) ----------
class EnhancedRetriever(BaseRetriever):
    """
    Wraps RetrievalPipeline to apply:
    - RRF + Rerank (done inside pipeline)
    - MMR (Maximum Marginal Relevance) for diversity
    - Duplicate removal (by chunk_id or content)
    - Rerank score threshold filtering
    - Context overflow truncation (token‑aware)
    """
    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        final_top_k: int = 10,
        rerank_threshold: float = 0.5,
        max_context_tokens: int = 3000,
        token_counter: Callable = None,
        # MMR parameters
        mmr_lambda: float = 0.5,              # relevance-diversity trade-off (0=纯多样, 1=纯相关)
        mmr_candidate_k: int = 20,            # 从 pipeline 获取多少个候选文档
        embedding_model: EmbeddingModel = None,  # 用于计算文档间相似度
    ):
        super().__init__()
        self.retrieval_pipeline = retrieval_pipeline
        self.final_top_k = final_top_k
        self.rerank_threshold = rerank_threshold
        self.max_context_tokens = max_context_tokens
        self.token_counter = token_counter or (lambda t: len(t.split()) // 0.75)

        self.mmr_lambda = mmr_lambda
        self.mmr_candidate_k = mmr_candidate_k
        self.embedding_model = embedding_model

    def _deduplicate(self, docs: List[Document]) -> List[Document]:
        """Remove duplicates by chunk_id (or content hash if chunk_id missing)."""
        seen = set()
        unique = []
        for doc in docs:
            doc_id = doc.metadata.get("chunk_id")
            if not doc_id:
                doc_id = hash(doc.page_content)  # fallback content hash
            if doc_id not in seen:
                seen.add(doc_id)
                unique.append(doc)
        return unique

    def _truncate_context(self, docs: List[Document]) -> List[Document]:
        """Truncate documents to fit within max_context_tokens."""
        total_tokens = 0
        truncated = []
        for doc in docs:
            doc_tokens = self.token_counter(doc.page_content)
            if total_tokens + doc_tokens <= self.max_context_tokens:
                truncated.append(doc)
                total_tokens += doc_tokens
            else:
                remaining = self.max_context_tokens - total_tokens
                if remaining > 50:  # only add if meaningful
                    # rough char-token mapping (assume 1 token ≈ 4 chars)
                    truncated_text = doc.page_content[:int(remaining * 4)]
                    doc.page_content = truncated_text
                    truncated.append(doc)
                break
        return truncated

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(a * a for a in v1) ** 0.5
        norm2 = sum(b * b for b in v2) ** 0.5
        return dot / (norm1 * norm2 + 1e-8)

    def _mmr_selection(self, docs_with_scores: List[Tuple[Document, float]], k: int) -> List[Document]:
        """
        Select top-k documents using Maximum Marginal Relevance.
        docs_with_scores: list of (doc, rerank_score) sorted by rerank_score descending initially.
        Returns selected documents in the order they were chosen (which is generally relevance-then-diverse).
        """
        if not docs_with_scores or k <= 0:
            return []
        if k >= len(docs_with_scores):
            return [doc for doc, _ in docs_with_scores]

        # Prepare document texts and optional embeddings
        texts = [doc.page_content for doc, _ in docs_with_scores]
        if self.embedding_model is not None:
            # Batch encode to get vectors
            vectors = self.embedding_model.encode(texts)  # list of lists
        else:
            # Fallback: use Jaccard similarity on character sets (less accurate, but works)
            logger.warning("No embedding_model provided for MMR, falling back to character-set Jaccard")
            vectors = [set(t) for t in texts]

        # Relevance scores (already from reranker)
        relevance_scores = [score for _, score in docs_with_scores]

        selected_indices = []
        remaining_indices = list(range(len(docs_with_scores)))

        # Step 1: Add the document with highest relevance
        first_idx = 0  # because input is sorted by relevance descending
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # Step 2: Iteratively add documents maximizing MMR
        for _ in range(1, k):
            if not remaining_indices:
                break
            best_idx = -1
            best_mmr = -float("inf")
            for idx in remaining_indices:
                # compute max similarity with already selected documents
                max_sim = 0.0
                for sel_idx in selected_indices:
                    if isinstance(vectors[0], list):
                        sim = self._cosine_similarity(vectors[idx], vectors[sel_idx])
                    else:
                        # Jaccard similarity for set fallback
                        inter = len(vectors[idx] & vectors[sel_idx])
                        union = len(vectors[idx] | vectors[sel_idx])
                        sim = inter / (union + 1e-8)
                    if sim > max_sim:
                        max_sim = sim
                # MMR = λ * relevance - (1-λ) * max_similarity
                mmr = self.mmr_lambda * relevance_scores[idx] - (1 - self.mmr_lambda) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx
            if best_idx != -1:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

        # Return documents in selection order
        return [docs_with_scores[i][0] for i in selected_indices]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Main retrieval method called by LangChain.
        """
        # 1. Get candidate documents from pipeline (already includes RRF + reranking)
        candidates_with_scores = self.retrieval_pipeline.get_knowledge_docs_with_scores(
            query, top_k=self.mmr_candidate_k
        )

        # 2. Filter by rerank threshold
        filtered = [(doc, score) for doc, score in candidates_with_scores if score >= self.rerank_threshold]

        # 3. Deduplicate (based on chunk_id or content hash)
        deduped_docs = self._deduplicate([doc for doc, _ in filtered])

        # 4. Rebuild score list for deduped docs (keeping the highest score per doc if duplicates existed)
        # Map from doc identifier to (doc, score). Use page_content as fallback key.
        score_map = {}
        for doc, score in filtered:
            key = doc.metadata.get("chunk_id", doc.page_content)
            if key not in score_map or score > score_map[key][1]:
                score_map[key] = (doc, score)
        deduped_with_scores = list(score_map.values())

        # 5. Apply MMR diversity selection
        mmr_selected_docs = self._mmr_selection(deduped_with_scores, self.final_top_k)

        # 6. Token‑aware truncation
        final_docs = self._truncate_context(mmr_selected_docs)

        return final_docs

    async def _aget_relevant_documents(self, query: str):
        return self._get_relevant_documents(query)


# ---------- Main RAG Class ----------
class MerchantRAGLangChain:
    def __init__(self, tenant_id: str = "default"):
        self.tenant_id = tenant_id
        self.preprocessor = QueryPreprocessor()
        self.cache = L1Cache(
            host=settings.REDIS_HOST, port=settings.REDIS_PORT,
            db=settings.REDIS_DB, password=settings.REDIS_PASSWORD,
            ttl=settings.REDIS_CACHE_TTL, max_connections=settings.REDIS_MAX_CONNECTIONS
        )

        # ----- 1. 初始化基础组件（顺序不可调换）-----
        self.embed_model = EmbeddingModel(settings.EMBEDDING_MODEL, settings.EMBEDDING_DEVICE)
        self.reranker = Reranker(settings.RERANKER_MODEL)
        self.milvus = MilvusClientWrapper(
            host=settings.MILVUS_HOST, port=settings.MILVUS_PORT,
            user=settings.MILVUS_USER, password=settings.MILVUS_PASSWORD,
            secure=settings.MILVUS_SECURE
        )

        # MySQL 连接池（保存为实例变量以便关闭）
        from pymysql.pool import Pool
        self.mysql_pool = Pool(
            host=settings.MYSQL_HOST, port=settings.MYSQL_PORT,
            user=settings.MYSQL_USER, password=settings.MYSQL_PASSWORD,
            database=settings.MYSQL_DATABASE,
            maxsize=settings.MYSQL_POOL_SIZE, autocommit=True
        )

        # 集合名称
        faq_collection = f"{settings.FAQ_COLLECTION}_{tenant_id}"
        knowledge_collection = f"{settings.KNOWLEDGE_COLLECTION}_{tenant_id}"

        self.faq_retriever = FAQRetriever(self.milvus, self.mysql_pool, self.embed_model, faq_collection)
        self.knowledge_retriever = KnowledgeRetriever(self.milvus, self.embed_model, knowledge_collection)

        # ----- 2. BM25 检索器（依赖 knowledge_retriever）-----
        bm25_retriever = None
        if getattr(settings, 'ENABLE_BM25', False):
            logger.info("Loading knowledge documents for BM25 indexing...")
            try:
                all_docs = self.knowledge_retriever.get_all_documents()
                if all_docs:
                    bm25_retriever = BM25Retriever(all_docs)
                    logger.info(f"BM25 index built with {len(all_docs)} documents")
                else:
                    logger.warning("No documents found for BM25, disabling BM25")
            except Exception as e:
                logger.error(f"Failed to build BM25 index: {e}")

        # ----- 3. 检索流水线（唯一实例）-----
        self.retrieval_pipeline = RetrievalPipeline(
            faq_retriever=self.faq_retriever,
            knowledge_retriever=self.knowledge_retriever,
            reranker=self.reranker,
            enable_bm25=getattr(settings, 'ENABLE_BM25', False) and bm25_retriever is not None,
            bm25_retriever=bm25_retriever
        )

        # ----- 4. 商家业务工具（包含新增的 retrieve_knowledge）-----
        self.merchant_tools = MerchantTools(
            retrieval_pipeline=self.retrieval_pipeline,
            order_api_url=settings.ORDER_API_URL,
            product_api_url=settings.PRODUCT_API_URL,
            api_key=settings.MERCHANT_API_KEY
        )

        # ----- 5. LangChain LLM & 增强检索器（使用 MMR）-----
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.LLM_BASE_URL,
            temperature=settings.LLM_TEMPERATURE,
            timeout=settings.LLM_TIMEOUT,
            max_retries=settings.LLM_MAX_RETRIES
        )
        # 使用增强版检索器（含 MMR）
        self.enhanced_retriever = EnhancedRetriever(
            retrieval_pipeline=self.retrieval_pipeline,
            final_top_k=10,
            rerank_threshold=0.5,
            max_context_tokens=3000,
            mmr_lambda=0.5,                # 可调整
            mmr_candidate_k=20,            # 取前20个候选进行MMR选择
            embedding_model=self.embed_model,
        )
        self.rag_chain = build_rag_chain(self.llm, self.enhanced_retriever)

        # ----- 6. Agent 工具集（修复闭包陷阱）-----
        tools = create_merchant_tools(self.merchant_tools)  # 内部已包含 retrieve_knowledge

        # Agent Prompt – 明确告知可以使用 retrieve_knowledge 工具进行知识检索
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
你是淘宝商家客服助手。可以使用以下工具：
- query_product / query_order / query_shipping: 查询实时数据
- get_return_policy / search_faq: 获取常见问题答案
- retrieve_knowledge: 从知识库（文档）中检索相关信息，用于解答政策细节、操作指南等问题
- handle_complaint: 处理投诉（可能转人工）
如果用户问题涉及知识库内容，请优先使用 retrieve_knowledge 工具。
"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent, tools=tools, verbose=False,
            handle_parsing_errors=True, max_iterations=5
        )
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # ----- 7. 意图路由器（可选，使用 bge-small）-----
        self.intent_router = None
        if getattr(settings, 'ENABLE_INTENT_ROUTER', False):
            self.intent_router = IntentRouter(
                use_classifier=getattr(settings, 'ENABLE_INTENT_CLASSIFIER', True),
                classifier_model=getattr(settings, 'INTENT_CLASSIFIER_MODEL', "BAAI/bge-small-zh-v1.5"),
                enable_context=True
            )

    def _post_process(self, text: str) -> str:
        from preprocess import PIIMask
        return PIIMask.mask(text)

    async def _fast_path_response(self, intent_result: Intent, query: str) -> Optional[str]:
        """快速路径：仅处理 FAQ 精确匹配（不涉及 RAG）"""
        if intent_result.intent == Intent.FAQ:
            answer, sources = self.retrieval_pipeline.get_faq_answer(query)
            if answer:
                return answer
        return None

    async def process(self, raw_query: str, session_id: str = "default"):
        start = time.perf_counter()
        clean_query = self.preprocessor.clean(raw_query)

        # L1 缓存检查
        cache_key = f"{self.tenant_id}:{clean_query}"
        cached = self.cache.get(cache_key)
        if cached:
            from your_response_model import MerchantResponse  # 请根据实际导入
            return MerchantResponse(
                answer=cached, sources=[], tool_used="cache",
                cached=True, latency_ms=(time.perf_counter() - start) * 1000
            )

        # ----- 意图路由（可选）：仅用于快速路径（FAQ 精确命中）-----
        if self.intent_router:
            tool_call = self.intent_router.route(clean_query, session_id)
            # 快速路径：FAQ 精确匹配直接返回，不进入 Agent
            fast_answer = await self._fast_path_response(
                getattr(tool_call.parameters.get('entities', {}), 'intent', Intent.FALLBACK),
                clean_query
            )
            if fast_answer:
                final_answer = self._post_process(fast_answer)
                self.cache.set(cache_key, final_answer)
                latency_ms = (time.perf_counter() - start) * 1000
                from your_response_model import MerchantResponse
                return MerchantResponse(
                    answer=final_answer, sources=[], tool_used="intent_fast_path",
                    cached=False, latency_ms=latency_ms
                )
            # 未命中快速路径，继续走 Agent

        # ----- 常规 Agent 处理（支持 tool calling + RAG 作为工具）-----
        try:
            result = await self.agent_executor.ainvoke({
                "input": clean_query,
                "chat_history": self.memory.chat_memory.messages
            })
            answer = result["output"]
            # 更新对话记忆
            self.memory.chat_memory.add_user_message(clean_query)
            self.memory.chat_memory.add_ai_message(answer)
            tool_used = "langchain_agent"
            sources = []
            need_human = False
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            # 降级：直接使用 RAG 链（纯检索 + 生成）
            answer = await self.rag_chain.ainvoke(clean_query)
            tool_used = "rag_fallback"
            sources = []
            need_human = False

        final_answer = self._post_process(answer)
        self.cache.set(cache_key, final_answer)

        latency_ms = (time.perf_counter() - start) * 1000
        from your_response_model import MerchantResponse
        return MerchantResponse(
            answer=final_answer,
            sources=sources,
            tool_used=tool_used,
            cached=False,
            latency_ms=latency_ms,
            need_human=need_human
        )

    async def close(self):
        """清理资源（MySQL 连接池、Redis 等）"""
        if hasattr(self, 'mysql_pool'):
            self.mysql_pool.close()
        if hasattr(self, 'cache'):
            await self.cache.close()
        # 如有必要，可添加 Milvus 断开连接等