#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基础RAG实现框架
"""

import os
import json
from typing import List, Dict, Any

# 假设使用的库，实际使用时需要安装
from langchain_community.document_loaders import TextLoader, DirectoryLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain_community.embeddings import HuggingFaceEmbeddings # 备选，但不推荐
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
# from langchain_community.llms import Ollama  # Deprecated
from langchain_ollama import OllamaLLM


class BasicRAG:
    """基础RAG类"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化RAG系统"""
        self.config = config
        self.embedding_model = None
        self.vector_store = None
        self.llm = None
        self.text_splitter = None
        
    def init_components(self):
        """初始化各个组件"""
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get("chunk_size", 1000),
            chunk_overlap=self.config.get("chunk_overlap", 200)
        )
        
        # 初始化嵌入模型
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.config.get("embedding_model", "BAAI/bge-small-zh-v1.5"),
            cache_folder=self.config.get("cache_folder", "./embedding_cache")
        )
        
        # 初始化LLM
        self.llm = OllamaLLM(
            model=self.config.get("llm_model", "qwen2:0.5b"),
            temperature=self.config.get("temperature", 0.1)
        )
        
        print("RAG组件初始化完成")
    
    def load_documents(self, data_path: str) -> List[Any]:
        """加载文档"""
        documents = []
        
        if os.path.isdir(data_path):
            # 加载目录中的所有文件
            txt_loader = DirectoryLoader(
                data_path,
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            documents.extend(txt_loader.load())
            
            # 加载Excel文件
            excel_loader = DirectoryLoader(
                data_path,
                glob="**/*.xlsx",
                loader_cls=UnstructuredExcelLoader
            )
            documents.extend(excel_loader.load())
        elif os.path.isfile(data_path):
            # 加载单个文件
            if data_path.endswith('.txt'):
                loader = TextLoader(data_path)
                documents.extend(loader.load())
            elif data_path.endswith('.xlsx'):
                loader = UnstructuredExcelLoader(data_path)
                documents.extend(loader.load())
        
        print(f"加载文档: {data_path}, 共{len(documents)}个文档")
        return documents
    
    def split_documents(self, documents: List[Any]) -> List[Any]:
        """分割文档"""
        split_docs = self.text_splitter.split_documents(documents)
        print(f"分割文档完成，从{len(documents)}个文档分割为{len(split_docs)}个片段")
        return split_docs
    
    def create_vector_store(self, documents: List[Any], store_path: str = None):
        """创建向量存储"""
        import os
        
        if store_path and os.path.exists(store_path):
            # 加载现有向量库
            self.vector_store = FAISS.load_local(
                store_path,
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            print(f"加载现有向量库: {store_path}")
        else:
            # 创建新向量库
            self.vector_store = FAISS.from_documents(
                documents,
                self.embedding_model
            )
            # 保存向量库
            if store_path:
                self.vector_store.save_local(store_path)
                print(f"向量库保存到: {store_path}")
        
        print(f"创建向量存储完成，文档数量: {len(documents)}")
    
    def retrieve(self, query: str, k: int = 3) -> List[Any]:
        """检索相关文档"""
        if not self.vector_store:
            raise ValueError("向量存储未初始化")
        
        docs = self.vector_store.similarity_search(query, k=k)
        print(f"检索查询: {query}, 找到{len(docs)}个相关文档")
        return docs
    
    def generate_answer(self, query: str, retrieved_docs: List[Any]) -> str:
        """生成回答"""
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""使用以下上下文来回答用户的问题。
        上下文：
        {context}
        问题：{question}
        请使用中文回答，并且只基于提供的上下文。如果上下文不足以回答问题，请说"根据提供的信息无法回答此问题"。
        回答："""
        )
        
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        prompt = prompt_template.format(context=context, question=query)
        
        response = self.llm.invoke(prompt)
        return response
        # print(f"生成回答: {query}")
        # return "这是一个示例回答"
    
    def rag_pipeline(self, query: str, k: int = 3) -> str:
        """完整的RAG流水线"""
        # 检索相关文档
        retrieved_docs = self.retrieve(query, k=k)
        
        # 生成回答
        answer = self.generate_answer(query, retrieved_docs)
        
        return answer
    
    def evaluate(self, test_data: List[Dict[str, str]]) -> Dict[str, float]:
        """评估RAG系统"""
        # 简单的评估示例
        # 实际评估应使用更复杂的指标，如BLEU、ROUGE、BLEURT等
        correct = 0
        total = len(test_data)
        
        for item in test_data:
            query = item["query"]
            expected = item["expected_answer"]
            
            actual = self.rag_pipeline(query)
            
            # 简单匹配检查
            if expected in actual:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        return {"accuracy": accuracy}


def main():
    """主函数"""
    # 配置参数
    config = {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "embedding_model": "BAAI/bge-small-zh-v1.5",
        "llm_model": "qwen2:0.5b",
        "temperature": 0.1
    }
    
    # 创建RAG实例
    rag = BasicRAG(config)
    
    # 初始化组件
    rag.init_components()
    
    # 加载Excel文档
    data_path = "../cigaratte_data.xlsx"  # 假设Excel文件在data目录
    documents = rag.load_documents(data_path)
    
    # 分割文档
    split_docs = rag.split_documents(documents)
    
    # 创建向量存储
    store_path = "../vector_store"
    # rag.create_vector_store(split_docs, store_path)
    
    # 示例使用
    query = "卷烟知识库的主要功能是什么？"
    answer = rag.rag_pipeline(query)
    print(f"\n查询: {query}")
    print(f"回答: {answer}")


if __name__ == "__main__":
    main()
