# langchain_wrapper.py
from typing import List, Dict, Any, Optional
from langchain_core.tools import StructuredTool
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from retrievers import RetrievalPipeline

class MerchantToolInput(BaseModel):
    query: str = Field(description="用户查询")
    entities: Dict[str, Any] = Field(default_factory=dict, description="提取的实体")

def create_merchant_tools(merchant_tools_instance):
    tools = []
    method_names = [
        "query_product", "query_order", "get_return_policy",
        "query_shipping", "query_promotion", "handle_complaint", 
        "search_faq",
        "retrieve_knowledge"   # ← 新增 RAG 工具
    ]
    for method_name in method_names:
        method = getattr(merchant_tools_instance, method_name)
        # 使用 partial 固定 method 参数
        async def tool_func(query: str, entities: Dict[str, Any] = None, meth=method):
            result = await meth(query, entities)
            if isinstance(result, tuple) and len(result) == 2:
                return result[0]
            return str(result)
        tool = StructuredTool.from_function(
            coroutine=tool_func,
            name=method_name,
            description=f"调用 {method_name} 工具，用于处理商家客服相关查询",
            args_schema=MerchantToolInput
        )
        tools.append(tool)
    return tools

class MilvusRetrieverWrapper(BaseRetriever):
    """将 RetrievalPipeline 包装为 LangChain Retriever"""
    def __init__(self, retrieval_pipeline: RetrievalPipeline):
        super().__init__()
        self.pipeline = retrieval_pipeline

    def _get_relevant_documents(self, query: str) -> List[Document]:
        context, sources = self.pipeline.get_knowledge_context(query, top_k=3)
        doc = Document(page_content=context, metadata={"sources": sources})
        return [doc]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)

def build_rag_chain(llm: ChatOpenAI, retriever: MilvusRetrieverWrapper):
    """使用 LCEL 构建 RAG 链"""
    template = """基于以下参考资料回答用户问题。如果参考资料不足以回答，请诚实说明。

参考资料：
{context}

用户问题：{question}
回答："""
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        if not docs:
            return "无相关参考资料。"
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain