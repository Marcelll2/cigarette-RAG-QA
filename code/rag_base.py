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
import pandas as pd
from langchain_core.documents import Document
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
            import glob
            excel_files = glob.glob(os.path.join(data_path, "**/*.xlsx"), recursive=True)
            for excel_file in excel_files:
                # 使用pandas直接读取Excel文件
                try:
                    # 读取所有工作表
                    xls = pd.ExcelFile(excel_file)
                    for sheet_name in xls.sheet_names:
                        # 读取工作表内容
                        df = pd.read_excel(excel_file, sheet_name=sheet_name)
                        # 按行创建文档
                        for idx, row in df.iterrows():
                            # 提取关键字段到 metadata
                            metadata = {
                                "source": excel_file,
                                "sheet": sheet_name,
                                "row": idx + 2  # +2 因为Excel行号从1开始，且有表头
                            }
                            
                            # 将除了 INTRODUCTION 之外的字段添加到 metadata
                            content_parts = []
                            for col, val in row.items():
                                if pd.notna(val):
                                    col_lower = col.lower()
                                    content_parts.append(f"{col}: {val}")
                                    # 跳过 INTRODUCTION 字段
                                    if "introduction" not in col_lower:
                                        metadata[col_lower] = val
                            
                            row_content = "\n".join(content_parts)
                            if row_content.strip():
                                # 创建文档对象
                                doc = Document(
                                    page_content=row_content.strip(),
                                    metadata=metadata
                                )
                                documents.append(doc)
                except Exception as e:
                    print(f"处理Excel文件时出错 {excel_file}: {e}")
        elif os.path.isfile(data_path):
            # 加载单个文件
            if data_path.endswith('.txt'):
                loader = TextLoader(data_path)
                documents.extend(loader.load())
            elif data_path.endswith('.xlsx'):
                # 使用pandas直接读取Excel文件
                try:
                    # 读取所有工作表
                    xls = pd.ExcelFile(data_path)
                    for sheet_name in xls.sheet_names:
                        # 读取工作表内容
                        df = pd.read_excel(data_path, sheet_name=sheet_name)
                        # 按行创建文档
                        for idx, row in df.iterrows():
                            # 将行转换为字符串
                            # 提取关键字段到 metadata
                            metadata = {
                                "source": data_path,
                                "sheet": sheet_name,
                                "row": idx + 2  # +2 因为Excel行号从1开始，且有表头
                            }
                            
                            # 将除了 INTRODUCTION 之外的字段添加到 metadata
                            content_parts = []
                            for col, val in row.items():
                                if pd.notna(val):
                                    col_lower = col.lower()
                                    content_parts.append(f"{col}: {val}")
                                    # 跳过 INTRODUCTION 字段
                                    if "introduction" not in col_lower:
                                        metadata[col_lower] = val
                            
                            row_content = "\n".join(content_parts)
                            if row_content.strip():
                                # 创建文档对象
                                doc = Document(
                                    page_content=row_content.strip(),
                                    metadata=metadata
                                )
                                documents.append(doc)
                except Exception as e:
                    print(f"处理Excel文件时出错 {data_path}: {e}")
        
        print(f"加载文档: {data_path}, 共{len(documents)}个文档")
        return documents
    
    def split_documents(self, documents: List[Any]) -> List[Any]:
        """分割文档"""
        # Excel文件已经在load_documents中按行处理，这里只需要处理非Excel文档
        split_docs = []
        
        for doc in documents:
            # 检查是否是Excel文档（通过metadata中的source字段判断）
            if 'xlsx' in doc.metadata.get('source', ''):
                # Excel文档已经按行分割，直接添加
                split_docs.append(doc)
            else:
                # 对于非Excel文档，使用默认的文本分割器
                split_docs.extend(self.text_splitter.split_documents([doc]))
        
        print(f"分割文档完成，从{len(documents)}个文档分割为{len(split_docs)}个片段")
        return split_docs
    
    def save_documents(self, documents: List[Any], save_path: str):
        """保存文档到文件"""
        import json
        import os
        
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 准备文档数据
        documents_data = []
        for i, doc in enumerate(documents):
            doc_data = {
                "id": i,
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            documents_data.append(doc_data)
        
        # 保存为JSON文件
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(documents_data, f, ensure_ascii=False, indent=2)
        
        print(f"文档保存到: {save_path}, 共{len(documents_data)}个文档")
    
    def print_documents(self, documents: List[Any], max_docs: int = 5, max_content_len: int = 200):
        """整齐打印文档信息"""
        print(f"\n=== 文档信息汇总 ===")
        print(f"总文档数: {len(documents)}")
        print("=" * 50)
        
        # 限制打印的文档数量
        docs_to_print = documents[:max_docs]
        
        for i, doc in enumerate(docs_to_print):
            print(f"\n文档 {i+1}/{len(documents)}:")
            print(f"源文件: {doc.metadata.get('source', '未知')}")
            print(f"内容长度: {len(doc.page_content)} 字符")
            
            # 打印内容预览
            content_preview = doc.page_content[:max_content_len]
            if len(doc.page_content) > max_content_len:
                content_preview += "..."
            
            print("内容预览:")
            print("-" * 40)
            print(content_preview)
            print("-" * 40)
        
        if len(documents) > max_docs:
            print(f"\n... 还有 {len(documents) - max_docs} 个文档未显示 ...")
        print("\n=== 文档信息结束 ===")
    
    def create_vector_store(self, documents: List[Any], store_path: str = None):
        if not documents:
            raise ValueError("文档列表为空")
        self.vector_store = FAISS.from_documents(
            documents,
            self.embedding_model)
        # 保存向量库
        store_path_ = os.path.join(store_path, self.embedding_model.model_name.replace("/", "_"))
        self.vector_store.save_local(store_path_)
        print(f"向量库保存到: {store_path_}")
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

    def set_embedding_model(self, model_name: str = None):
        # !此处代码待验证扩展性和兼容性
        origin_embedding_model = self.embedding_model
        if model_name is None:
            raise ValueError("model_name 不能为空")
        if model_name == origin_embedding_model.model_name:
            raise ValueError(f"模型 {model_name} 与当前模型相同，无需切换")
            
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=self.config.get("cache_folder", "./embedding_models")
        )

def main():
    """主函数"""
    # 配置参数
    config = {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "embedding_model": "BAAI/bge-small-zh-v1.5",
        "llm_model": "qwen2:0.5b",
        "temperature": 0.1,
        "vector_store_path": "../vector_store/BAAI_bge-small-zh-v1.5"
    }
    
    # 创建RAG实例
    rag = BasicRAG(config)
    
    # 初始化组件
    rag.init_components()
    
    # 加载Excel文档
    data_path = "../cigaratte_data.xlsx"  # 假设Excel文件在data目录
    documents = rag.load_documents(data_path)
    # 保存文档
    # rag.save_documents(documents, ../saved_documents/documents.json)
    # rag.print_documents(documents)    
    
    # 分割文档
    split_docs = rag.split_documents(documents)
    # rag.save_documents(split_docs, "../saved_documents/split_docs.json")
    # print(f'split_docs: {split_docs}')
    
    # 创建向量存储
    if not os.path.exists(config["vector_store_path"]):
        print(f"重新创建向量存储: {config['vector_store_path']}")
        store_path = "../vector_store"
        # store_path = os.path.join(store_path, config["embedding_model"].replace("/", "_"))
        rag.create_vector_store(split_docs, store_path)
    else:
        print(f"使用现存的向量存储: {config['vector_store_path']}")
        rag.vector_store = FAISS.load_local(
            config["vector_store_path"],
            rag.embedding_model,
            allow_dangerous_deserialization=True)
    
    # 示例使用
    query = "卷烟知识库的主要功能是什么？"
    answer = rag.rag_pipeline(query)
    print(f"\n查询: {query}")
    print(f"回答: {answer}")


if __name__ == "__main__":
    main()
