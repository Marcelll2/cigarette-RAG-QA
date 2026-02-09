#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG微调工具
"""

import json
import itertools
import os
from typing import List, Dict, Any
from rag_base import BasicRAG
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class RAGFinetuner:
    """RAG微调类"""
    
    def __init__(self, base_config: Dict[str, Any], data_path: str = None):
        """初始化微调器"""
        self.base_config = base_config
        self.rag = BasicRAG(base_config)
        self.rag.init_components()
        self.data_path = data_path
        self.vector_store_ready = False
        # 从配置中读取是否使用LLM评估的超参数
        self.use_llm_evaluation = base_config.get("use_llm_evaluation", False)
        # 存储最佳参数
        self.best_retrieval_params: Dict[str, Any] = None
        self.best_prompt: str = None
        self.best_chunking: Dict[str, int] = None
        self.best_embedding: Dict[str, Any] = None
        
        if self.data_path:
            self._prepare_vector_store()
        
    def tune_retrieval_params(self, test_queries: List[str], param_grid: Dict[str, List[Any]]) -> None:
        """微调检索参数"""
        print("\n=== 微调检索参数 ===")
        
        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_combinations = list(itertools.product(*param_grid.values()))
        
        best_params: Dict[str, Any] = None
        best_score: float = 0.0
        
        for i, params in enumerate(param_combinations):
            # 创建当前参数配置
            current_params: Dict[str, Any] = dict(zip(param_names, params))
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
        
        # 存储最佳检索参数
        self.best_retrieval_params = best_params
    
    def _prepare_vector_store(self):
        """准备向量存储"""            
            # 检查是否为现存的向量存储路径
        if os.path.isdir(self.data_path) and any(
            os.path.exists(os.path.join(self.data_path, file)) 
            for file in ["index.faiss", "index.pkl"]):
            # 使用现存的向量存储
            print(f"使用现存的向量存储: {self.data_path}")
            self.rag.vector_store = FAISS.load_local(
                self.data_path,
                self.rag.embedding_model,
                allow_dangerous_deserialization=True
            )
            self.vector_store_ready = True
            print("现存向量存储加载完成")
            return      
        # 加载文档
        documents = self.rag.load_documents(self.data_path)        
        # 分割文档
        split_docs = self.rag.split_documents(documents)        
        # 创建向量存储
        store_path = "../vector_store"
        self.rag.create_vector_store(split_docs, store_path)        
        self.vector_store_ready = True
        print("向量存储准备完成")
    
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
                retrieved_docs: List[Any] = self.rag.retrieve(query, k=k)
                
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
            docs_with_scores: List[tuple[Document, float]] = self.rag.vector_store.similarity_search_with_relevance_scores(
                query, 
                k=len(retrieved_docs)
            )
            
            total_score = 0.0
            valid_docs = 0
            
            for doc, score in docs_with_scores:
                # print(f"文档内容: \n{doc.page_content[:50]}... \n相似度分数: {score:.4f}\n")
                if score >= similarity_threshold:
                    total_score += score
                    valid_docs += 1
                else:
                    # 低于阈值的文档得分减半
                    total_score += score * 0.5
                    valid_docs += 1
            # print(f'')
            
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
    
    def optimize_prompt_template(self, test_cases: List[Dict[str, str]], prompt_candidates: List[str]) -> None:
        """优化提示模板"""
        print("\n=== 优化提示模板 ===")
        
        best_prompt: str = None
        best_score: float = 0.0
        
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
        
        # 如果没有找到最佳匹配，默认使用第一个模板
        if best_prompt is None and prompt_candidates:
            best_prompt = prompt_candidates[0]
            print("\n警告：未找到最佳匹配，使用第一个模板作为默认值")
        
        print(f"\n最佳提示模板:")
        print(f"{best_prompt}")
        print(f"最佳分数: {best_score}")
        
        # 存储最佳提示模板
        self.best_prompt = best_prompt
    
    def _evaluate_prompt(self, test_cases: List[Dict[str, str]], prompt: str) -> float:
        """评估提示模板效果"""
        # 实际使用LLM生成回答并评估
        total_score = 0.0
        
        for case in test_cases:
            query = case["query"]
            expected = case["expected_answer"]
            
            try:
                # 执行实际检索获取上下文
                retrieved_docs = self.rag.retrieve(query, k=3)
                
                if not retrieved_docs:
                    print(f"警告：查询 '{query}' 未检索到任何文档")
                    total_score += 0.0
                    continue
                
                # 使用指定的提示模板生成回答
                from langchain_core.prompts import PromptTemplate
                
                # 创建提示模板
                prompt_template = PromptTemplate(
                    input_variables=["context", "question"],
                    template=prompt
                )
                
                # 构建上下文
                context = "\n".join([doc.page_content for doc in retrieved_docs])
                
                # 生成回答
                prompt_text = prompt_template.format(context=context, question=query)
                answer = self.rag.llm.invoke(prompt_text)
                
                print(f"\n查询: {query}")
                print(f"期望回答: {expected}")
                print(f"实际回答: {answer[:100]}...")
                
                # 评估回答质量
                score = self._calculate_answer_score(answer, expected)
                total_score += score
                print(f"得分: {score}")
                
            except Exception as e:
                print(f"生成回答时出错: {e}")
                total_score += 0.0
        
        return total_score / len(test_cases)
    
    def _calculate_answer_score(self, answer: str, expected: str) -> float:
        """计算回答质量得分"""
        # 简单的匹配评分
        if expected in answer:
            return 1.0
        elif any(word in answer for word in expected.split()[:3]):
            return 0.5
            #! 该方法仅用于简单匹配，实际应用中需要更复杂的评分逻辑
            #! 而且该方法仅考虑了关键词是否在回答中出现，没有考虑关键词的顺序和上下文
            #! 同时，该方法所使用的answer据说是前三个单词是核心关键词，此处存疑
        else:
            # 计算关键词匹配率
            expected_words = set(expected.split())
            answer_words = set(answer.split())
            common_words = expected_words.intersection(answer_words)
            
            if expected_words:
                match_ratio = len(common_words) / len(expected_words)
                return min(0.5, match_ratio)
            else:
                return 0.0
    
    def tune_chunking_strategy(self, documents: List[Any] = None, chunking_params: Dict[str, List[int]] = None) -> None:
        """微调文本分割策略"""
        print("\n=== 微调文本分割策略 ===")
        
        try:
            # 如果没有提供文档，尝试获取
            if documents is None:
                documents = self._get_chunking_documents()
            
            # 如果没有提供参数，尝试获取
            if chunking_params is None:
                # 这里传入空字典，因为我们不依赖 test_data
                chunking_params = self._get_chunking_params({})
            
            # 生成所有分割参数组合
            chunk_sizes = chunking_params.get("chunk_sizes", [500, 1000, 2000])
            chunk_overlaps = chunking_params.get("chunk_overlaps", [100, 200, 300])
            
            best_params = None
            best_score = 0.0
            
            for chunk_size in chunk_sizes:
                for chunk_overlap in chunk_overlaps:
                    if chunk_overlap >= chunk_size:
                        continue
                        
                    current_params: Dict[str, int] = {
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
            
            # 如果没有找到最佳参数，使用默认值
            if best_params is None:
                print("未找到有效分割参数，使用默认值")
                best_params = {"chunk_size": 1000, "chunk_overlap": 200}
                
        except Exception as e:
            print(f"微调分割策略时出错: {e}")
            best_params = {"chunk_size": 1000, "chunk_overlap": 200}  # 使用默认值
    
    def _get_chunking_documents(self) -> List[Any]:
        """获取用于分割策略评估的文档"""
        try:
            # 尝试从数据路径加载文档
            if hasattr(self, "data_path") and self.data_path:
                try:
                    documents = self.rag.load_documents(self.data_path)
                    if documents:
                        print(f"成功加载 {len(documents)} 个文档用于分割策略评估")
                        return documents
                except Exception as e:
                    print(f"加载文档时出错: {e}")
            
            # 回退到默认测试文档
            print("使用默认文档进行分割策略评估")
            return self._get_test_documents()
        except Exception as e:
            print(f"获取分割策略评估文档时出错: {e}")
            # 返回空列表，让调用方处理
            return []
    
    def _get_chunking_params(self, test_data: Dict[str, Any]) -> Dict[str, List[int]]:
        """获取分割策略参数"""
        # 尝试从配置中获取
        try:
            if hasattr(self, "base_config"):
                # 检查是否有 finetuning_config
                if "finetuning_config" in self.base_config:
                    if "chunking_params" in self.base_config["finetuning_config"]:
                        return self.base_config["finetuning_config"]["chunking_params"]
        except Exception as e:
            print(f"获取分割参数时出错: {e}")
        
        # 使用默认参数
        print("使用默认分割参数")
        return {
            "chunk_sizes": [500, 1000, 1500, 2000],
            "chunk_overlaps": [100, 200, 300]
        }
    
    def _evaluate_chunking(self, params: Dict[str, int]) -> float:
        """评估分割策略效果"""
        # 根据配置选择评估方法
        if self.use_llm_evaluation:
            return self._evaluate_chunking_with_llm(params)
        else:
            # 原有的启发式评分逻辑
            return self._evaluate_chunking_heuristic(params)
    
    def _evaluate_chunking_heuristic(self, params: Dict[str, int]) -> float:
        """使用启发式方法评估分割策略"""
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
    
    def _evaluate_chunking_with_llm(self, params: Dict[str, int]) -> float:
        """使用LLM评估分割策略效果"""
        print("使用LLM评估分割策略...")
        
        try:
            # 准备评估数据（使用示例文档或实际文档）
            test_documents = self._get_test_documents()
            if not test_documents:
                # 如果没有测试文档，回退到启发式评估
                print("警告：无测试文档，回退到启发式评估")
                return self._evaluate_chunking_heuristic(params)
            
            # 使用当前参数执行分割
            chunk_size = params["chunk_size"]
            chunk_overlap = params["chunk_overlap"]
            
            # 临时修改RAG的分割参数进行评估
            original_chunk_size = getattr(self.rag, "chunk_size", 1000)
            original_chunk_overlap = getattr(self.rag, "chunk_overlap", 200)
            
            self.rag.chunk_size = chunk_size
            self.rag.chunk_overlap = chunk_overlap
            
            # 执行分割
            split_docs = self.rag.split_documents(test_documents)
            
            # 恢复原始参数
            self.rag.chunk_size = original_chunk_size
            self.rag.chunk_overlap = original_chunk_overlap
            
            if not split_docs:
                print("警告：分割后无文档，回退到启发式评估")
                return self._evaluate_chunking_heuristic(params)
            
            # 选择部分分割结果进行评估
            eval_docs = split_docs[:3]  # 只评估前3个分割结果以节省成本
            total_score = 0.0
            
            for i, doc in enumerate(eval_docs):
                # 构建评估提示
                eval_prompt = f"""请评估以下文本片段作为文档分割结果的质量，从以下几个方面打分（0-10分）：
1. 语义完整性：片段是否保持了完整的语义
2. 边界合理性：分割是否在合理的语义边界上
3. 信息密度：片段包含的信息量是否适中
4. 上下文连贯性：片段内部是否连贯

文本片段：
{doc.page_content}

请返回一个综合评分（0-10分），只需要数字，不需要其他说明。"""
                
                # 使用LLM生成评估
                try:
                    response = self.rag.llm.invoke(eval_prompt)
                    # 提取评分
                    score_text = response.strip()
                    # 尝试从响应中提取数字
                    import re
                    score_match = re.search(r'\d+(\.\d+)?', score_text)
                    if score_match:
                        score = float(score_match.group()) / 10.0  # 转换为0-1范围
                        score = min(1.0, max(0.0, score))  # 确保在0-1范围内
                        total_score += score
                        print(f"分割片段 {i+1} 评分: {score:.2f}")
                    else:
                        print(f"警告：无法从LLM响应中提取评分，使用默认值0.5")
                        total_score += 0.5
                except Exception as e:
                    print(f"LLM评估出错: {e}，使用默认值0.5")
                    total_score += 0.5
            
            # 计算平均得分
            final_score = total_score / len(eval_docs)
            print(f"LLM评估最终得分: {final_score:.2f}")
            
            return final_score
            
        except Exception as e:
            print(f"LLM评估分割策略时出错: {e}，回退到启发式评估")
            return self._evaluate_chunking_heuristic(params)
    
    def _get_test_documents(self) -> List[Any]:
        """获取用于评估的测试文档"""
        # 尝试从数据路径加载文档
        if self.data_path:
            try:
                return self.rag.load_documents(self.data_path)
            except Exception as e:
                print(f"加载测试文档出错: {e}")
        
        # 如果没有数据路径，使用默认测试文档
        from langchain_core.documents import Document
        default_docs = [
            Document(
                page_content="卷烟是一种复杂的消费品，由烟叶、滤嘴、卷烟纸等多种材料组成。卷烟的生产过程包括烟叶处理、配方设计、卷制包装等多个环节。不同品牌的卷烟在口味、香气、焦油含量等方面存在差异，这些差异主要由烟叶配方和生产工艺决定。",
                metadata={"source": "default"}
            ),
            Document(
                page_content="烟草行业是我国重要的产业之一，对国家财政收入有着重要贡献。近年来，随着健康意识的提高，烟草控制政策逐渐加强，卷烟销量呈现下降趋势。同时，烟草企业也在积极探索新型烟草制品，以适应市场变化。",
                metadata={"source": "default"}
            )
        ]
        return default_docs
    
    def compare_embedding_models(self, model_candidates: List[str], test_data: List[Dict[str, str]]) -> str:
        pass
    
    def _evaluate_embedding_model(self, model: str, test_data: List[Dict[str, str]]) -> float:
        pass
    
    def run_full_finetuning(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行完整的微调流程"""
        print("\n=== 开始完整的RAG微调流程 ===")
        
        # 1. 微调检索参数
        # self.tune_retrieval_params(
        #     test_data["test_queries"],
        #     test_data["retrieval_param_grid"]
        # )
        # 测试模式：使用默认值
        if self.best_retrieval_params is None:
            print("使用默认检索参数 (测试模式)")
            self.best_retrieval_params = {"k": 3, "similarity_threshold": 0.7}
        
        # 2. 优化提示模板
        # self.optimize_prompt_template(
        #     test_data["test_cases"],
        #     test_data["prompt_candidates"]
        # )
        # 测试模式：使用默认值
        if self.best_prompt is None:
            print("使用默认提示模板 (测试模式)")
            self.best_prompt = "请根据以下上下文回答问题：\n\n{context}\n\n问题：{question}\n\n回答："
        
        # 3. 微调文本分割策略
        # self.tune_chunking_strategy()
        # 测试模式：使用默认值
        if self.best_chunking is None:
            print("使用默认分割参数 (测试模式)")
            self.best_chunking = {"chunk_size": 1000, "chunk_overlap": 200}
        # print(f"最佳分割参数: {self.best_chunking}")        
        
        # 4. 比较嵌入模型
        self.compare_embedding_models(
            test_data["embedding_candidates"],
            test_data["test_cases"]
        )
        # 测试模式：使用默认值
        if self.best_embedding is None:
            print("使用默认嵌入模型 (测试模式)")
            self.best_embedding = self.base_config.get("embedding_model", "bge-base-zh")
        else:
            print(f"使用最佳嵌入模型: {self.best_embedding}")
        
        # 整合最佳参数
        best_config = {
            "retrieval_params": self.best_retrieval_params,
            "prompt_template": self.best_prompt,
            "chunking_params": self.best_chunking,
            "embedding_model": self.best_embedding,
            **self.base_config
        }
        # best_config = {}        
        
        print("\n=== 微调完成 ===")
        print(f"最佳配置: {json.dumps(best_config, ensure_ascii=False, indent=2)}")
        
        return best_config


def main():
    """主函数"""
    # 从config.json加载配置
    config_file = "config.json"
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 获取基础配置
        base_config = config.get("base_config", {})
        
        # 获取测试数据
        test_data = config.get("test_data", {})
        
        # 添加微调配置到测试数据
        finetuning_config = config.get("finetuning_config", {})
        test_data.update(finetuning_config)
        
        print("配置加载成功")
        
    except Exception as e:
        print(f"加载配置文件时出错: {e}")
        # 使用默认配置作为后备
        base_config = {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "embedding_model": "BAAI/bge-small-zh-v1.5",
            "llm_model": "qwen2:0.5b",
            "temperature": 0.1
        }
        test_data = {
            "test_queries": ["卷烟知识库的主要功能是什么？"],
            "retrieval_param_grid": {"k": [3], "similarity_threshold": [0.7]}
        }
    
    # 数据路径（用于实际检索）
    # 原始数据路径
    data_path = config.get("base_config", {}).get("data_path", "../cigaratte_data.xlsx")
    # 向量存储路径
    # vector_data_path = config.get("base_config", {}).get("vector_data_path", "../vector_store")
    
    # 创建微调器实例，传入数据路径以启用实际检索
    finetuner = RAGFinetuner(base_config, data_path=data_path)
    
    # 运行完整微调流程
    best_config = finetuner.run_full_finetuning(test_data)
    
    # 保存最佳配置
    # with open("best_rag_config.json", "w", encoding="utf-8") as f:
    #     json.dump(best_config, f, ensure_ascii=False, indent=2)
    
    # print("\n最佳配置已保存到 best_rag_config.json")


if __name__ == "__main__":
    main()
