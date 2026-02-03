#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG微调工具
"""

import json
import itertools
from typing import List, Dict, Any
from rag_base import BasicRAG


class RAGFinetuner:
    """RAG微调类"""
    
    def __init__(self, base_config: Dict[str, Any], data_path: str = None):
        """初始化微调器"""
        self.base_config = base_config
        self.rag = BasicRAG(base_config)
        self.rag.init_components()
        self.data_path = data_path
        self.vector_store_ready = False
        
        if data_path:
            self._prepare_vector_store()
        
    def tune_retrieval_params(self, test_queries: List[str], param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """微调检索参数"""
        print("\n=== 微调检索参数 ===")
        
        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_combinations = list(itertools.product(*param_grid.values()))
        
        best_params = None
        best_score = 0.0
        
        for i, params in enumerate(param_combinations):
            # 创建当前参数配置
            current_params = dict(zip(param_names, params))
            print(f"\n测试组合 {i+1}/{len(param_combinations)}: {current_params}")
            
            # 测试当前参数
            # 这里应该使用实际的评估指标，如相关性评分
            # 简化示例：使用固定分数
            current_score = self._evaluate_retrieval(test_queries, current_params)
            
            print(f"组合分数: {current_score}")
            
            # 更新最佳参数
            if current_score > best_score:
                best_score = current_score
                best_params = current_params
        
        print(f"\n最佳检索参数: {best_params}")
        print(f"最佳分数: {best_score}")
        
        return best_params
    
    def _prepare_vector_store(self):
        """准备向量存储"""
        if not self.data_path:
            print("警告：未提供数据路径，无法创建向量存储")
            return
            
        try:
            # 加载文档
            documents = self.rag.load_documents(self.data_path)
            
            # 分割文档
            split_docs = self.rag.split_documents(documents)
            
            # 创建向量存储
            store_path = "../vector_store"
            self.rag.create_vector_store(split_docs, store_path)
            
            self.vector_store_ready = True
            print("向量存储准备完成")
        except Exception as e:
            print(f"准备向量存储时出错: {e}")
            self.vector_store_ready = False
    
    def _evaluate_retrieval(self, test_queries: List[str], params: Dict[str, Any]) -> float:
        """评估检索效果"""
        if not self.vector_store_ready:
            print("警告：向量存储未就绪，使用模拟评估")
            return self._simulate_retrieval(test_queries, params)
        
        total_score = 0.0
        
        for query in test_queries:
            k = params.get("k", 3)
            similarity_threshold = params.get("similarity_threshold", 0.7)
            
            try:
                # 执行实际检索
                retrieved_docs = self.rag.retrieve(query, k=k)
                
                if not retrieved_docs:
                    print(f"警告：查询 '{query}' 未检索到任何文档")
                    total_score += 0.0
                    continue
                
                # 评估检索质量
                query_score = self._calculate_retrieval_score(
                    query, 
                    retrieved_docs, 
                    similarity_threshold
                )
                
                total_score += query_score
                
            except Exception as e:
                print(f"检索查询 '{query}' 时出错: {e}")
                total_score += 0.0
        
        return total_score / len(test_queries)
    
    def _calculate_retrieval_score(self, query: str, retrieved_docs: List[Any], similarity_threshold: float) -> float:
        """计算检索得分"""
        # 获取检索文档的相似度分数（如果可用）
        # FAISS similarity_search_with_relevance_scores 可以返回相似度分数
        try:
            # 使用带相似度分数的检索
            docs_with_scores = self.rag.vector_store.similarity_search_with_relevance_scores(
                query, 
                k=len(retrieved_docs)
            )
            
            total_score = 0.0
            valid_docs = 0
            
            for doc, score in docs_with_scores:
                if score >= similarity_threshold:
                    total_score += score
                    valid_docs += 1
                else:
                    # 低于阈值的文档得分减半
                    total_score += score * 0.5
                    valid_docs += 1
            
            # 归一化得分
            if valid_docs > 0:
                return min(1.0, total_score / valid_docs)
            return 0.0
            
        except Exception as e:
            print(f"计算相似度分数时出错: {e}")
            # 回退到基于文档数量的简单评分
            return min(1.0, len(retrieved_docs) / 5.0)
    
    def _simulate_retrieval(self, test_queries: List[str], params: Dict[str, Any]) -> float:
        """模拟检索效果（当向量存储不可用时）"""
        total_score = 0.0
        
        for query in test_queries:
            k = params.get("k", 3)
            similarity_threshold = params.get("similarity_threshold", 0.7)
            
            retrieved_docs = min(k, 5)
            query_score = 0.0
            
            for i in range(retrieved_docs):
                base_relevance = 1.0 - (i * 0.15)
                if base_relevance >= similarity_threshold:
                    query_score += base_relevance
                else:
                    query_score += base_relevance * 0.5
            
            normalized_score = min(1.0, query_score / retrieved_docs)
            total_score += normalized_score
        
        return total_score / len(test_queries)
    
    def optimize_prompt_template(self, test_cases: List[Dict[str, str]], prompt_candidates: List[str]) -> str:
        """优化提示模板"""
        print("\n=== 优化提示模板 ===")
        
        best_prompt = None
        best_score = 0.0
        
        for i, prompt in enumerate(prompt_candidates):
            print(f"\n测试提示模板 {i+1}/{len(prompt_candidates)}:")
            print(f"{prompt[:100]}...")
            
            # 测试当前提示模板
            current_score = self._evaluate_prompt(test_cases, prompt)
            
            print(f"模板分数: {current_score}")
            
            # 更新最佳提示模板
            if current_score > best_score:
                best_score = current_score
                best_prompt = prompt
        
        print(f"\n最佳提示模板:")
        print(f"{best_prompt}")
        print(f"最佳分数: {best_score}")
        
        return best_prompt
    
    def _evaluate_prompt(self, test_cases: List[Dict[str, str]], prompt: str) -> float:
        """评估提示模板效果"""
        # 简化的评估逻辑
        # 实际应使用更复杂的指标，如BLEU、ROUGE等
        total_score = 0.0
        
        for case in test_cases:
            query = case["query"]
            expected = case["expected_answer"]
            
            # 模拟生成回答
            # 实际应使用真实的LLM生成和评估
            answer = f"根据上下文，{expected}"  # 简化模拟
            
            # 简单的匹配评分
            if expected in answer:
                total_score += 1.0
            elif any(word in answer for word in expected.split()[:3]):
                total_score += 0.5
        
        return total_score / len(test_cases)
    
    def tune_chunking_strategy(self, documents: List[Any], chunking_params: Dict[str, List[int]]) -> Dict[str, int]:
        """微调文本分割策略"""
        print("\n=== 微调文本分割策略 ===")
        
        # 生成所有分割参数组合
        chunk_sizes = chunking_params.get("chunk_sizes", [500, 1000, 2000])
        chunk_overlaps = chunking_params.get("chunk_overlaps", [100, 200, 300])
        
        best_params = None
        best_score = 0.0
        
        for chunk_size in chunk_sizes:
            for chunk_overlap in chunk_overlaps:
                if chunk_overlap >= chunk_size:
                    continue
                    
                current_params = {
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap
                }
                
                print(f"测试分割参数: {current_params}")
                
                # 测试当前分割策略
                current_score = self._evaluate_chunking(current_params)
                
                print(f"分割分数: {current_score}")
                
                # 更新最佳参数
                if current_score > best_score:
                    best_score = current_score
                    best_params = current_params
        
        print(f"\n最佳分割参数: {best_params}")
        print(f"最佳分数: {best_score}")
        
        return best_params
    
    def _evaluate_chunking(self, params: Dict[str, int]) -> float:
        """评估分割策略效果"""
        # 简化的评估逻辑
        # 实际应基于分割后文档的检索效果和生成质量评估
        # 这里使用一个启发式评分：
        # - 适中的chunk_size更好（避免过短或过长）
        # - chunk_overlap应适中（避免过多重复）
        
        chunk_size = params["chunk_size"]
        chunk_overlap = params["chunk_overlap"]
        
        # 理想的chunk_size范围是500-1500
        size_score = 1.0 - abs(chunk_size - 1000) / 1000
        
        # 理想的chunk_overlap是chunk_size的10%-20%
        ideal_overlap = chunk_size * 0.15
        overlap_score = 1.0 - abs(chunk_overlap - ideal_overlap) / chunk_size
        
        total_score = (size_score + overlap_score) / 2
        
        return total_score
    
    def compare_embedding_models(self, model_candidates: List[str], test_data: List[Dict[str, str]]) -> str:
        """比较不同的嵌入模型"""
        print("\n=== 比较嵌入模型 ===")
        
        best_model = None
        best_score = 0.0
        
        for model in model_candidates:
            print(f"\n测试嵌入模型: {model}")
            
            # 测试当前模型
            current_score = self._evaluate_embedding_model(model, test_data)
            
            print(f"模型分数: {current_score}")
            
            # 更新最佳模型
            if current_score > best_score:
                best_score = current_score
                best_model = model
        
        print(f"\n最佳嵌入模型: {best_model}")
        print(f"最佳分数: {best_score}")
        
        return best_model
    
    def _evaluate_embedding_model(self, model: str, test_data: List[Dict[str, str]]) -> float:
        """评估嵌入模型效果"""
        # 简化的评估逻辑
        # 实际应基于模型的检索效果和计算效率评估
        # 这里使用模型名称的启发式评分
        
        # 假设更大的模型效果更好
        if "large" in model.lower():
            return 0.9
        elif "base" in model.lower() or "medium" in model.lower():
            return 0.8
        elif "small" in model.lower():
            return 0.7
        else:
            return 0.6
    
    def run_full_finetuning(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行完整的微调流程"""
        print("\n=== 开始完整的RAG微调流程 ===")
        
        # 1. 微调检索参数
        retrieval_params = self.tune_retrieval_params(
            test_data["test_queries"],
            test_data["retrieval_param_grid"]
        )
        
        # 2. 优化提示模板
        # best_prompt = self.optimize_prompt_template(
        #     test_data["test_cases"],
        #     test_data["prompt_candidates"]
        # )
        
        # 3. 微调文本分割策略
        # best_chunking = self.tune_chunking_strategy(
        #     test_data["sample_documents"],
        #     test_data["chunking_params"]
        # )
        
        # 4. 比较嵌入模型
        # best_embedding = self.compare_embedding_models(
        #     test_data["embedding_candidates"],
        #     test_data["test_cases"]
        # )
        
        # 整合最佳参数
        # best_config = {
        #     "retrieval_params": retrieval_params,
        #     "prompt_template": best_prompt,
        #     "chunking_params": best_chunking,
        #     "embedding_model": best_embedding,
        #     **self.base_config
        # }
        best_config = {}        
        
        print("\n=== 微调完成 ===")
        print(f"最佳配置: {json.dumps(best_config, ensure_ascii=False, indent=2)}")
        
        return best_config


def main():
    """主函数"""
    # 基础配置
    base_config = {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "embedding_model": "BAAI/bge-small-zh-v1.5",
        "llm_model": "qwen2:0.5b",
        "temperature": 0.1
    }
    
    # 创建微调器实例
    finetuner = RAGFinetuner(base_config)
    
    # 测试数据
    test_data = {
        "test_queries": [
            "卷烟知识库的主要功能是什么？",
            "如何使用卷烟知识库进行查询？",
            "卷烟知识库包含哪些类型的信息？"
        ],
        "test_cases": [
            {
                "query": "卷烟知识库的主要功能是什么？",
                "expected_answer": "卷烟知识库主要用于存储和检索卷烟相关的各类信息，包括产品信息、政策法规、销售数据等。"
            },
            {
                "query": "如何使用卷烟知识库进行查询？",
                "expected_answer": "用户可以通过关键词搜索、分类浏览等方式使用卷烟知识库进行查询。"
            }
        ],
        "retrieval_param_grid": {
            "k": [2, 3, 4, 5],
            "similarity_threshold": [0.7, 0.8, 0.9]
        },
        "prompt_candidates": [
            "使用以下上下文来回答用户的问题。上下文：{context} 问题：{question} 请使用中文回答，并且只基于提供的上下文。",
            "基于以下上下文，简洁地回答用户的问题。上下文：{context} 问题：{question} 回答：",
            "请根据以下上下文信息，准确回答用户的问题。上下文：{context} 问题：{question} 回答要求：1. 仅使用提供的上下文；2. 使用中文；3. 简洁明了。回答："
        ],
        "chunking_params": {
            "chunk_sizes": [500, 1000, 1500, 2000],
            "chunk_overlaps": [100, 200, 300]
        },
        "sample_documents": [],  # 实际使用时应提供样本文档
        "embedding_candidates": [
            "BAAI/bge-small-zh-v1.5",
            "BAAI/bge-base-zh-v1.5",
            "BAAI/bge-large-zh-v1.5",
            "GanymedeNil/text2vec-large-chinese"
        ]
    }
    
    # 运行完整微调流程
    best_config = finetuner.run_full_finetuning(test_data)
    
    # 保存最佳配置
    # with open("best_rag_config.json", "w", encoding="utf-8") as f:
    #     json.dump(best_config, f, ensure_ascii=False, indent=2)
    
    # print("\n最佳配置已保存到 best_rag_config.json")


if __name__ == "__main__":
    main()
