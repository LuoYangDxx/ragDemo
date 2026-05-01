"""
RAG 系统可观测性模块
集成 Prometheus 指标采集、结构化日志、分布式追踪
"""

import time
import psutil
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import FastAPI, Response, Request
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
import structlog
import logging

# ======================= 1. 结构化日志配置 =======================

def setup_structured_logging():
    """配置结构化日志，便于日志聚合和检索（如 ELK、Loki）"""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger()


# 全局日志记录器
logger = setup_structured_logging()


# ======================= 2. RAG 专属 Prometheus 指标 =======================

# ---------- 性能指标 ----------
# RAG 端到端延迟直方图（秒）
RAG_LATENCY = Histogram(
    'rag_latency_seconds',
    'RAG end-to-end latency in seconds',
    ['model', 'operation'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# 检索环节延迟（秒）
RETRIEVAL_LATENCY = Histogram(
    'rag_retrieval_latency_seconds',
    'Vector retrieval latency in seconds',
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
)

# LLM 生成环节延迟（秒）
LLM_LATENCY = Histogram(
    'rag_llm_latency_seconds',
    'LLM generation latency in seconds',
    ['model'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# 请求计数器
RAG_REQUESTS_TOTAL = Counter(
    'rag_requests_total',
    'Total number of RAG requests',
    ['model', 'status']  # status: success, error, timeout
)

# 当前活跃请求数（用于监控并发压力）
ACTIVE_REQUESTS = Gauge(
    'rag_active_requests',
    'Number of active RAG requests'
)

# ---------- 检索质量指标 ----------
# 检索相似度分数分布（0~1）
RETRIEVAL_SCORES = Histogram(
    'rag_retrieval_scores',
    'Distribution of retrieval similarity scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# 检索到的文档数量
RETRIEVED_DOCS_COUNT = Histogram(
    'rag_retrieved_docs_count',
    'Number of documents retrieved per query',
    buckets=[1, 2, 3, 4, 5, 10, 15, 20]
)

# 低相关性检索告警指标（相似度低于阈值）
LOW_RELEVANCE_RETRIEVALS = Counter(
    'rag_low_relevance_retrievals_total',
    'Total queries with max similarity below threshold (0.4)'
)

# ---------- LLM 运营指标 ----------
# Token 使用量（用于成本监控）
TOKENS_USED = Counter(
    'rag_tokens_total',
    'Total tokens processed by LLM',
    ['model', 'direction']  # direction: input, output
)

# 模型调用次数
MODEL_CALLS = Counter(
    'rag_model_calls_total',
    'Total LLM model calls',
    ['model']
)

# ---------- 系统资源指标 ----------
# 内存使用量（字节）
MEMORY_USAGE = Gauge(
    'rag_memory_usage_bytes',
    'Memory usage of the RAG application'
)

# CPU 使用率（百分比）
CPU_USAGE = Gauge(
    'rag_cpu_usage_percent',
    'CPU usage percentage'
)

# ---------- 可选：缓存命中率 ----------
CACHE_HITS = Counter(
    'rag_cache_hits_total',
    'Total cache hits'
)
CACHE_MISSES = Counter(
    'rag_cache_misses_total',
    'Total cache misses'
)


# ======================= 3. 辅助记录函数 =======================

def record_rag_request(
    model: str,
    status: str,
    latency_seconds: float
):
    """记录一次完整的 RAG 请求"""
    RAG_REQUESTS_TOTAL.labels(model=model, status=status).inc()
    RAG_LATENCY.labels(model=model, operation="rag").observe(latency_seconds)


def record_retrieval(
    latency_seconds: float,
    top_scores: List[float],
    docs_count: int
):
    """记录检索环节的指标"""
    RETRIEVAL_LATENCY.observe(latency_seconds)
    RETRIEVED_DOCS_COUNT.observe(docs_count)

    if top_scores:
        max_score = max(top_scores)
        RETRIEVAL_SCORES.observe(max_score)

        # 低相关性阈值告警（相似度低于 0.4 视为低质量）
        if max_score < 0.4:
            LOW_RELEVANCE_RETRIEVALS.inc()
            logger.warning("low_relevance_retrieval", max_score=max_score, docs_count=docs_count)


def record_llm_call(
    model: str,
    latency_seconds: float,
    input_tokens: int,
    output_tokens: int
):
    """记录 LLM 调用的指标"""
    LLM_LATENCY.labels(model=model).observe(latency_seconds)
    MODEL_CALLS.labels(model=model).inc()
    TOKENS_USED.labels(model=model, direction="input").inc(input_tokens)
    TOKENS_USED.labels(model=model, direction="output").inc(output_tokens)


def update_system_metrics():
    """更新系统资源指标（CPU、内存）"""
    try:
        process = psutil.Process()
        MEMORY_USAGE.set(process.memory_info().rss)
        CPU_USAGE.set(psutil.cpu_percent(interval=0.1))
    except Exception as e:
        logger.error("system_metrics_update_failed", error=str(e))


def record_cache_hit():
    """记录缓存命中"""
    CACHE_HITS.inc()


def record_cache_miss():
    """记录缓存未命中"""
    CACHE_MISSES.inc()


# ======================= 4. OpenTelemetry 追踪配置 =======================

def setup_tracing(service_name: str = "rag-demo", otlp_endpoint: str = "http://localhost:4318/v1/traces"):
    """
    配置 OpenTelemetry 分布式追踪
    可对接 Jaeger、Grafana Tempo 等后端
    """
    try:
        provider = TracerProvider()
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        logger.info("tracing_initialized", endpoint=otlp_endpoint)
        return trace.get_tracer(service_name)
    except Exception as e:
        logger.error("tracing_init_failed", error=str(e))
        # 返回一个空 tracer（不阻塞业务）
        return trace.get_tracer(service_name)


# 全局 tracer 实例（可按需初始化）
tracer = setup_tracing()


# ======================= 5. FastAPI 仪表化集成 =======================

def setup_metrics(app: FastAPI):
    """
    配置 Prometheus 指标自动采集和 FastAPI 仪表化
    会自动添加 /metrics 端点，并记录每个 HTTP 请求的指标
    """
    # 自动仪表化所有 HTTP 端点（请求数、延迟、响应大小等）
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    # 添加中间件来记录当前活跃 HTTP 请求数（可选，与 RAG 活跃请求分开）
    @app.middleware("http")
    async def record_http_active_requests(request: Request, call_next):
        from prometheus_client import Gauge
        # 可以定义一个新的 Gauge 用于 HTTP 层面，或者复用 ACTIVE_REQUESTS
        # 这里演示复用 ACTIVE_REQUESTS，但注意区分标签可能更好
        # 为简单，我们单独创建一个 HTTP 活跃请求 gauge
        if not hasattr(record_http_active_requests, "http_active"):
            record_http_active_requests.http_active = Gauge(
                'http_active_requests',
                'Number of active HTTP requests'
            )
        record_http_active_requests.http_active.inc()
        try:
            response = await call_next(request)
            return response
        finally:
            record_http_active_requests.http_active.dec()

    # 可选：手动仪表化 FastAPI（OpenTelemetry 自动追踪）
    try:
        FastAPIInstrumentor.instrument_app(app)
        HTTPXClientInstrumentor().instrument()
        logger.info("fastapi_instrumentation_enabled")
    except Exception as e:
        logger.warning("fastapi_instrumentation_failed", error=str(e))


# ======================= 6. 结构化日志辅助函数 =======================

def log_rag_query(
    query: str,
    rewritten_query: Optional[str],
    retrieved_docs: List[Dict[str, Any]],
    scores: List[float],
    latency_ms: float,
    model: str,
    generation: str
):
    """
    记录一次 RAG 查询的完整结构化日志
    便于后续在日志系统中进行检索和分析
    """
    logger.info(
        "rag_query",
        query=query[:200],  # 限制长度防止日志过大
        rewritten_query=rewritten_query,
        retrieved_docs_count=len(retrieved_docs),
        max_score=max(scores) if scores else None,
        latency_ms=latency_ms,
        model=model,
        response_length=len(generation),
        top_doc_ids=[doc.get("id", "unknown") for doc in retrieved_docs[:3]]
    )


def log_error(
    error_type: str,
    error_msg: str,
    query: Optional[str] = None,
    **extra
):
    """记录错误信息（结构化）"""
    logger.error(
        error_type,
        error=error_msg,
        query=query[:200] if query else None,
        **extra
    )


# ======================= 7. 可选：指标重置/清理（用于测试） =======================

def reset_metrics():
    """重置所有指标（主要用于单元测试）"""
    from prometheus_client import CollectorRegistry
    # 注意：这会清空所有已注册的指标，谨慎使用
    registry = CollectorRegistry()
    global RAG_LATENCY, RETRIEVAL_LATENCY, LLM_LATENCY, RAG_REQUESTS_TOTAL
    global ACTIVE_REQUESTS, RETRIEVAL_SCORES, RETRIEVED_DOCS_COUNT
    global LOW_RELEVANCE_RETRIEVALS, TOKENS_USED, MODEL_CALLS
    global MEMORY_USAGE, CPU_USAGE, CACHE_HITS, CACHE_MISSES

    RAG_LATENCY = Histogram('rag_latency_seconds', '...', ['model', 'operation'], registry=registry)
    RETRIEVAL_LATENCY = Histogram('rag_retrieval_latency_seconds', '...', registry=registry)
    LLM_LATENCY = Histogram('rag_llm_latency_seconds', '...', ['model'], registry=registry)
    RAG_REQUESTS_TOTAL = Counter('rag_requests_total', '...', ['model', 'status'], registry=registry)
    ACTIVE_REQUESTS = Gauge('rag_active_requests', '...', registry=registry)
    RETRIEVAL_SCORES = Histogram('rag_retrieval_scores', '...', registry=registry)
    RETRIEVED_DOCS_COUNT = Histogram('rag_retrieved_docs_count', '...', registry=registry)
    LOW_RELEVANCE_RETRIEVALS = Counter('rag_low_relevance_retrievals_total', '...', registry=registry)
    TOKENS_USED = Counter('rag_tokens_total', '...', ['model', 'direction'], registry=registry)
    MODEL_CALLS = Counter('rag_model_calls_total', '...', ['model'], registry=registry)
    MEMORY_USAGE = Gauge('rag_memory_usage_bytes', '...', registry=registry)
    CPU_USAGE = Gauge('rag_cpu_usage_percent', '...', registry=registry)
    CACHE_HITS = Counter('rag_cache_hits_total', '...', registry=registry)
    CACHE_MISSES = Counter('rag_cache_misses_total', '...', registry=registry)