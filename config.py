# config.py
import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_CACHE_TTL: int = 3600
    REDIS_MAX_CONNECTIONS: int = 10

    # Milvus
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: str = "19530"
    MILVUS_USER: str = ""
    MILVUS_PASSWORD: str = ""
    MILVUS_SECURE: bool = False
    FAQ_COLLECTION: str = "faq_collection"
    KNOWLEDGE_COLLECTION: str = "knowledge_collection"
    MILVUS_SEARCH_NPROBE: int = 10

    # MySQL
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = ""
    MYSQL_DATABASE: str = "rag_demo"
    MYSQL_POOL_SIZE: int = 5

    # Embedding & Reranker (用于检索底层)
    EMBEDDING_MODEL: str = "BAAI/bge-base-zh-v1.5"
    EMBEDDING_DEVICE: str = "cpu"
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"

    # LangChain LLM (OpenAI/DeepSeek 兼容)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_BASE_URL: str = "https://api.deepseek.com/v1"
    LLM_MODEL: str = "deepseek-chat"
    LLM_TEMPERATURE: float = 0.1
    LLM_TIMEOUT: int = 30
    LLM_MAX_RETRIES: int = 3

    # 外部 API (订单、商品查询)
    ORDER_API_URL: Optional[str] = None
    PRODUCT_API_URL: Optional[str] = None
    MERCHANT_API_KEY: Optional[str] = None

    # LangChain 可选追踪
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: str = "merchant-rag"

    # BM25 混合检索开关（内存版）
    ENABLE_BM25: bool = True   # 是否启用 BM25 检索
    BM25_TOKENIZER: str = "jieba"  # 分词器，支持 jieba

    class Config:
        env_file = ".env"
        case_sensitive = False

    ENABLE_INTENT_ROUTER = True          # 是否启用意图路由器
    ENABLE_INTENT_CLASSIFIER = True      # 是否使用 bge-small 分类器（False 时降级为规则路由）
    INTENT_CLASSIFIER_MODEL = "BAAI/bge-small-zh-v1.5"
    INTENT_THRESHOLD = 0.5               # 分类器置信度阈值

settings = Settings()