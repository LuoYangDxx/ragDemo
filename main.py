# main.py
import asyncio
import logging
from merchant_rag_langchain import MerchantRAGLangChain

logging.basicConfig(level=logging.INFO)

async def main():
    system = MerchantRAGLangChain(tenant_id="demo_shop")
    # 测试查询
    queries = [
        "订单号 TB1234567890 到哪了？",
        "你们家这款手机有货吗？",
        "怎么退货？",
        "有什么优惠活动？"
    ]
    for q in queries:
        response = await system.process(q)
        print(f"\n用户: {q}")
        print(f"助手: {response.answer}")
        print(f"工具: {response.tool_used}, 耗时: {response.latency_ms:.2f}ms")
    await system.close()

if __name__ == "__main__":
    asyncio.run(main())