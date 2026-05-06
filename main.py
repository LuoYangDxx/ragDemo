# main.py 新内容
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from merchant_rag_langchain import MerchantRAGLangChain
from monitoring import setup_metrics, update_system_metrics
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----- 全局 RAG 实例 -----
rag_system = None

# ----- 请求/响应模型 -----
class AskRequest(BaseModel):
    query: str
    session_id: str = "default"
    tenant_id: str = "demo_shop"

class AskResponse(BaseModel):
    answer: str
    sources: list = []
    tool_used: str
    cached: bool
    latency_ms: float
    need_human: bool = False

# ----- 生命周期管理 -----
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_system
    # 启动时初始化 RAG 系统
    rag_system = MerchantRAGLangChain(tenant_id="demo_shop")
    logger.info("RAG system initialized")
    
    # 启动后台任务：定期采集系统资源指标
    asyncio.create_task(background_metrics_updater())
    
    yield
    
    # 关闭时清理资源
    if rag_system:
        await rag_system.close()
    logger.info("RAG system shut down")

async def background_metrics_updater():
    """每隔15秒更新 CPU/内存指标"""
    while True:
        update_system_metrics()
        await asyncio.sleep(15)

# ----- 创建 FastAPI 应用 -----
app = FastAPI(lifespan=lifespan, title="Merchant RAG with Observability")

# 集成 Prometheus 自动仪表化（可选，会添加 HTTP 请求指标）
setup_metrics(app)   # 这个函数在 monitoring.py 中定义，会暴露 /metrics

# ----- API 端点 -----
@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(req: AskRequest):
    """RAG 问答接口"""
    try:
        # 创建临时 RAG 实例（如果每个请求需要不同 tenant_id，可以动态创建）
        # 由于全局只有一个，这里直接使用
        response = await rag_system.process(req.query, req.session_id)
        return AskResponse(
            answer=response.answer,
            sources=response.sources,
            tool_used=response.tool_used,
            cached=response.cached,
            latency_ms=response.latency_ms,
            need_human=response.need_human
        )
    except Exception as e:
        logger.exception(f"Error processing query: {req.query}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Prometheus 指标端点（如果 setup_metrics 已暴露则无需重复，但保留作为保障）"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
async def health():
    return {"status": "ok"}

# ----- 可选：保留原有的测试运行方式 -----
async def run_test_queries():
    """测试已有功能，不启动 HTTP 服务"""
    system = MerchantRAGLangChain(tenant_id="demo_shop")
    queries = [
        "订单号 TB1234567890 到哪了？",
        "你们家这款手机有货吗？",
        "怎么退货？",
        "有什么优惠活动？"
    ]
    for q in queries:
        resp = await system.process(q)
        print(f"\n用户: {q}\n助手: {resp.answer}\n工具: {resp.tool_used}, 耗时: {resp.latency_ms:.2f}ms")
    await system.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        asyncio.run(run_test_queries())
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)