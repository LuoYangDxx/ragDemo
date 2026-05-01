# router.py
import logging
import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    """商家客服意图枚举"""
    PRODUCT_INQUIRY = "product_inquiry"      # 商品咨询（价格、库存、规格）
    ORDER_STATUS = "order_status"            # 订单状态查询
    RETURN_POLICY = "return_policy"          # 退换货/售后政策
    SHIPPING_INFO = "shipping_info"          # 物流信息
    PROMOTION_QUERY = "promotion_query"      # 优惠活动/优惠券
    COMPLAINT = "complaint"                  # 投诉建议
    FAQ = "faq"                              # 常见问题（知识库）
    FALLBACK = "fallback"                    # 无法识别，走纯 LLM


@dataclass
class IntentResult:
    intent: Intent
    entities: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class ToolCall:
    tool_name: str
    parameters: Dict[str, Any]
    requires_external_api: bool = False


class EntityExtractor:
    """从用户输入中提取商家相关实体"""
    ORDER_PATTERN = re.compile(r'\b([A-Z0-9]{10,20})\b')                     # 订单号
    PRODUCT_ID_PATTERN = re.compile(r'(?:itemid|商品id)[:：]?\s*(\d{10,15})', re.I)
    COUPON_PATTERN = re.compile(r'\b([A-Z0-9]{8,16})\b')                    # 优惠券码
    AMOUNT_PATTERN = re.compile(r'(\d+(?:\.\d{1,2})?)\s*元')                # 金额

    @classmethod
    def extract_order_id(cls, text: str) -> Optional[str]:
        match = cls.ORDER_PATTERN.search(text)
        return match.group(1) if match else None

    @classmethod
    def extract_product_id(cls, text: str) -> Optional[str]:
        match = cls.PRODUCT_ID_PATTERN.search(text)
        return match.group(1) if match else None

    @classmethod
    def extract_product_name(cls, text: str) -> Optional[str]:
        patterns = [r'商品[“"](.+?)[”"]', r'“(.+?)”\s*价格', r'关于\s*(.+?)\s*的问题']
        for pat in patterns:
            match = re.search(pat, text)
            if match:
                return match.group(1).strip()
        return None

    @classmethod
    def extract_coupon(cls, text: str) -> Optional[str]:
        match = cls.COUPON_PATTERN.search(text)
        return match.group(1) if match else None


class IntentClassifier:
    """基于嵌入相似度的轻量级意图分类器（使用 BGE-small 等模型）"""
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)
        # 为每个意图预定义一组示例查询（可配置）
        self.intent_examples = {
            Intent.ORDER_STATUS: [
                "我的订单到哪里了", "订单状态", "查询订单号", "发货了吗", "物流信息"
            ],
            Intent.PRODUCT_INQUIRY: [
                "这个商品多少钱", "有货吗", "什么规格", "尺寸多大", "颜色有哪些"
            ],
            Intent.RETURN_POLICY: [
                "怎么退货", "退款流程", "换货政策", "售后", "保修期"
            ],
            Intent.SHIPPING_INFO: [
                "物流单号", "快递什么时候到", "配送时间", "运费"
            ],
            Intent.PROMOTION_QUERY: [
                "有优惠券吗", "满减活动", "打折", "促销", "积分"
            ],
            Intent.COMPLAINT: [
                "投诉", "差评", "态度不好", "质量差", "我要投诉"
            ],
            Intent.FAQ: [
                "如何联系客服", "怎么开发票", "会员等级", "退换货地址"
            ],
        }
        # 预计算每个意图示例的嵌入向量（平均值作为意图中心）
        self.intent_centroids = {}
        for intent, examples in self.intent_examples.items():
            if examples:
                embeddings = self.model.encode(examples, normalize_embeddings=True)
                centroid = np.mean(embeddings, axis=0)
                self.intent_centroids[intent] = centroid

    def predict(self, query: str, threshold: float = 0.5) -> tuple[Intent, float]:
        """返回预测的意图和置信度"""
        query_emb = self.model.encode([query], normalize_embeddings=True)[0]
        max_sim = -1.0
        best_intent = Intent.FALLBACK
        for intent, centroid in self.intent_centroids.items():
            sim = np.dot(query_emb, centroid)  # 余弦相似度
            if sim > max_sim:
                max_sim = sim
                best_intent = intent
        if max_sim < threshold:
            return Intent.FALLBACK, max_sim
        return best_intent, max_sim


class IntentRouter:
    """
    意图路由器：支持规则路由或轻量级分类器。
    - 规则模式（默认）：关键词+正则匹配，速度快，零成本。
    - 分类器模式：使用嵌入相似度模型（如 BGE-small）进行语义分类，泛化能力更强。
    """
    def __init__(self, use_classifier: bool = False, classifier_model: str = None,
                 enable_context: bool = True, max_history: int = 5):
        self.use_classifier = use_classifier
        self.enable_context = enable_context
        self.session_memory: Dict[str, List[IntentResult]] = {}
        if use_classifier:
            model_name = classifier_model or "BAAI/bge-small-zh-v1.5"
            self.classifier = IntentClassifier(model_name)
        else:
            self.classifier = None

    def _rule_based_intent(self, query: str) -> IntentResult:
        """基于关键词匹配的规则路由"""
        q_lower = query.lower()
        entities = {}

        # 提取实体
        order_id = EntityExtractor.extract_order_id(query)
        if order_id:
            entities['order_id'] = order_id
        prod_id = EntityExtractor.extract_product_id(query)
        if prod_id:
            entities['product_id'] = prod_id
        prod_name = EntityExtractor.extract_product_name(query)
        if prod_name:
            entities['product_name'] = prod_name
        coupon = EntityExtractor.extract_coupon(query)
        if coupon:
            entities['coupon_code'] = coupon

        # 意图判断
        if any(kw in q_lower for kw in ['订单', '物流', '快递', '发货', '签收', '送达']):
            if 'order_id' in entities or re.search(r'订单号|单号', query):
                return IntentResult(Intent.ORDER_STATUS, entities)
            return IntentResult(Intent.SHIPPING_INFO, entities)
        if any(kw in q_lower for kw in ['退货', '退款', '换货', '售后', '维修', '保修']):
            return IntentResult(Intent.RETURN_POLICY, entities)
        if any(kw in q_lower for kw in ['活动', '优惠', '券', '满减', '打折', '促销']):
            return IntentResult(Intent.PROMOTION_QUERY, entities)
        if any(kw in q_lower for kw in ['投诉', '差评', '不满', '生气', '态度']):
            return IntentResult(Intent.COMPLAINT, entities)
        if any(kw in q_lower for kw in ['库存', '价格', '规格', '尺寸', '颜色', '款式', '有没有货']):
            return IntentResult(Intent.PRODUCT_INQUIRY, entities)
        if any(kw in q_lower for kw in ['怎么', '如何', '什么是', '能否', '可以吗']):
            return IntentResult(Intent.FAQ, entities)
        return IntentResult(Intent.FALLBACK, entities)

    def _classifier_based_intent(self, query: str) -> IntentResult:
        intent, conf = self.classifier.predict(query)
        # 仍然尝试提取实体
        entities = {}
        order_id = EntityExtractor.extract_order_id(query)
        if order_id:
            entities['order_id'] = order_id
        prod_id = EntityExtractor.extract_product_id(query)
        if prod_id:
            entities['product_id'] = prod_id
        prod_name = EntityExtractor.extract_product_name(query)
        if prod_name:
            entities['product_name'] = prod_name
        coupon = EntityExtractor.extract_coupon(query)
        if coupon:
            entities['coupon_code'] = coupon
        return IntentResult(intent, entities, conf)

    def _apply_context(self, session_id: str, current: IntentResult) -> IntentResult:
        if not self.enable_context or session_id not in self.session_memory:
            return current
        history = self.session_memory.get(session_id, [])
        if history:
            last = history[-1]
            # 如果当前意图 fallback，继承上一轮意图
            if current.intent == Intent.FALLBACK:
                current.intent = last.intent
            # 如果订单查询缺少订单号，继承上一轮的订单号
            if current.intent == Intent.ORDER_STATUS and not current.entities.get('order_id'):
                if last.entities.get('order_id'):
                    current.entities['order_id'] = last.entities['order_id']
        # 记忆更新
        history.append(current)
        if len(history) > 5:
            history.pop(0)
        self.session_memory[session_id] = history
        return current

    def route(self, query: str, session_id: str = "default") -> ToolCall:
        if self.use_classifier and self.classifier:
            intent_result = self._classifier_based_intent(query)
        else:
            intent_result = self._rule_based_intent(query)

        intent_result = self._apply_context(session_id, intent_result)

        # 意图到工具映射
        tool_map = {
            Intent.PRODUCT_INQUIRY: "query_product",
            Intent.ORDER_STATUS: "query_order",
            Intent.RETURN_POLICY: "get_return_policy",
            Intent.SHIPPING_INFO: "query_shipping",
            Intent.PROMOTION_QUERY: "query_promotion",
            Intent.COMPLAINT: "handle_complaint",
            Intent.FAQ: "search_faq",
            Intent.FALLBACK: "llm_generate",
        }
        tool_name = tool_map[intent_result.intent]
        params = {"query": query, "entities": intent_result.entities}
        return ToolCall(tool_name=tool_name, parameters=params)