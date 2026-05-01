# tools/merchant_tools.py
import logging
import aiohttp
from typing import Dict, Any, Tuple, List, Optional

logger = logging.getLogger(__name__)

class MerchantTools:
    def __init__(self, retrieval_pipeline, order_api_url: str = None,
                 product_api_url: str = None, api_key: str = None):
        self.retrieval = retrieval_pipeline
        self.order_api_url = order_api_url
        self.product_api_url = product_api_url
        self.api_key = api_key

    async def query_product(self, query: str, entities: Dict = None) -> Tuple[str, List[str]]:
        prod_id = (entities or {}).get('product_id')
        if prod_id and self.product_api_url:
            async with aiohttp.ClientSession() as session:
                try:
                    url = f"{self.product_api_url}/{prod_id}"
                    headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                    async with session.get(url, headers=headers) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            answer = (f"商品：{data.get('name')}\n价格：{data.get('price')}元\n"
                                      f"库存：{data.get('stock')}件\n规格：{data.get('specs')}")
                            return answer, ["商品API"]
                except Exception as e:
                    logger.error(f"Product API error: {e}")
        context, sources = self.retrieval.get_knowledge_context(query, top_k=3)
        return context, sources if context else ("未找到商品信息", [])

    async def query_order(self, query: str, entities: Dict = None) -> Tuple[str, List[str]]:
        order_id = (entities or {}).get('order_id')
        if not order_id:
            return "请提供订单号，例如：订单号 TB1234567890。", []
        if self.order_api_url:
            async with aiohttp.ClientSession() as session:
                try:
                    url = f"{self.order_api_url}/{order_id}"
                    headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                    async with session.get(url, headers=headers) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            status_map = {"pending": "待付款", "paid": "已付款", "shipped": "已发货",
                                          "delivered": "已签收", "closed": "已关闭"}
                            status = status_map.get(data.get('status'), data.get('status'))
                            answer = (f"订单号 {order_id} 状态：{status}\n"
                                      f"物流单号：{data.get('tracking_number', '无')}\n"
                                      f"预计送达：{data.get('eta', '待更新')}")
                            return answer, ["订单API"]
                        else:
                            return f"订单号 {order_id} 不存在，请核对。", []
                except Exception as e:
                    logger.error(f"Order API error: {e}")
        return "订单查询服务暂不可用，请稍后再试。", []

    async def get_return_policy(self, query: str, entities: Dict = None) -> Tuple[str, List[str]]:
        context, sources = self.retrieval.get_knowledge_context(query, top_k=2)
        if context:
            return context, sources
        return "本店支持7天无理由退换货，需保持商品完好。具体请咨询人工客服。", ["默认政策"]

    async def query_shipping(self, query: str, entities: Dict = None) -> Tuple[str, List[str]]:
        return await self.query_order(query, entities)

    async def query_promotion(self, query: str, entities: Dict = None) -> Tuple[str, List[str]]:
        context, sources = self.retrieval.get_knowledge_context(query, top_k=3)
        return context, sources if context else ("暂无进行中的优惠活动，请关注店铺首页。", [])

    async def handle_complaint(self, query: str, entities: Dict = None) -> Tuple[str, List[str]]:
        return ("非常抱歉给您带来不便，我已记录您的反馈，会尽快安排专员处理。"
                "您也可以拨打客服热线 400-123-4567。", ["投诉模板"])

    async def search_faq(self, query: str, entities: Dict = None) -> Tuple[str, List[str]]:
        answer, sources = self.retrieval.get_faq_answer(query)
        if answer:
            return answer, sources
        return "未找到相关常见问题，请尝试其他关键词或联系人工客服。", []
    
    async def retrieve_knowledge(self, query: str, entities: Dict = None) -> Tuple[str, List[str]]:
        """主动从知识库检索相关信息（RAG）"""
        context, sources = self.retrieval_pipeline.get_knowledge_context(query, top_k=3)
        if not context:
            return "未找到相关知识。", []
        return context, sources
    