#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG微调工具
"""

import json
import itertools
import os
import re
import argparse
import logging
from typing import List, Dict, Any
from rag_base import BasicRAG
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate


class RAGFinetuner:
    """RAG微调类"""
    def __init__(self, config: Dict[str, Any], data_path: str = None, args: argparse.Namespace = None):
        """初始化微调器"""
        self.args = args
        self.config = config
        self.rag = BasicRAG(self.config)
        self.rag.init_components()
        self.data_path = data_path
        self.vector_store_ready = False
        # 从配置中读取是否使用LLM评估的超参数
        self.use_llm_evaluation = self.config["base_config"]["use_llm_evaluation"]
        # 存储最佳参数
        self.best_retrieval_params: Dict[str, Any] = None
        self.best_prompt: str = None
        self.best_chunking: Dict[str, int] = None
        self.best_embedding: Dict[str, Any] = None
        
        self._prepare_vector_store()
        
    def tune_retrieval_params(self, test_queries: List[str], param_grid: Dict[str, List[Any]]) -> None:
        """微调检索参数"""
        print(f"\n=== 微调检索参数 ===")
        
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
            current_score = self._evaluate_retrieval(test_queries, current_params)
            print(f"组合分数: {current_score}")
            
            # 更新最佳参数
            if current_score > best_score:
                best_score = current_score
                best_params = current_params
        
        print(f"\n最佳检索参数: {best_params}")
        print(f"最佳分数: {best_score}\n\n\n==============")        
        self.best_retrieval_params = best_params
    
    def _prepare_vector_store(self):
        """准备向量存储"""
            
        # 计算完整的向量存储路径
        store_path = self.config["base_config"]["vector_store_path"]
        full_store_path = os.path.join(store_path, self.rag.embedding_model.model_name.replace("/", "_"))
        
        # 检查是否为现存的向量存储路径
        if os.path.isdir(full_store_path) and any(os.path.exists(os.path.join(full_store_path, file)) for file in ["index.faiss", "index.pkl"]):
            # 使用现存的向量存储
            print(f"使用现存的向量存储: {full_store_path}")
            self.rag.vector_store = FAISS.load_local(
                full_store_path,
                self.rag.embedding_model,
                allow_dangerous_deserialization=True
            )
            self.vector_store_ready = True
            print("现有向量存储加载完成")
        else:
            # 加载文档
            documents = self.rag.load_documents(self.config["base_config"]["data_path"])
            # 分割文档
            split_docs = self.rag.split_documents(documents)
            # 创建向量存储
            self.rag.create_vector_store(
                documents=split_docs, 
                store_path=full_store_path)
            self.vector_store_ready = True
            print("新向量创建备完成")
    
    def _evaluate_retrieval(self, test_queries: List[str], params: Dict[str, Any]) -> float:
        """评估检索效果"""
        if not self.vector_store_ready:
            print("警告：向量存储未就绪，使用模拟评估")
            return self._simulate_retrieval(test_queries, params)
        
        total_score = 0.0        
        for query in test_queries:
            k = params["k"]
            similarity_threshold = params["similarity_threshold"]                        
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
        return total_score / len(test_queries)
    
    def _calculate_retrieval_score(self, query: str, retrieved_docs: List[Any], similarity_threshold: float) -> float:
        """计算检索得分"""
        # FAISS similarity_search_with_relevance_scores 可以返回相似度分数
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
        
        # 归一化得分
        if valid_docs > 0:
            return min(1.0, total_score / valid_docs)
        return 0.0
    
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
            print(f"{prompt}...")
            
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
        print(f"最佳分数: {best_score}\n\n\n==============")        
        self.best_prompt = best_prompt
    
    def _evaluate_prompt(self, test_cases: List[Dict[str, str]], prompt: str) -> float:
        """评估提示模板效果"""
        # 实际使用LLM生成回答并评估
        total_score = 0.0
        
        for case in test_cases:
            query = case["query"]
            expected = case["expected_answer"]
            
            # 执行实际检索获取上下文
            if self.best_retrieval_params:
                best_k = self.best_retrieval_params["k"]  
            else:
                raise ValueError("best_retrieval_params 未设置")
            retrieved_docs = self.rag.retrieve(query, k=best_k)
            
            if not retrieved_docs:
                print(f"警告：查询 '{query}' 未检索到任何文档")
                total_score += 0.0
                continue
                                 
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
        
        return total_score / len(test_cases)
    
    def _calculate_answer_score(self, answer: str, expected: str) -> float:
        """计算回答质量得分"""
        # 优化后的评分逻辑：多维度评估
        
        # 1. 完全匹配（最高分）
        if expected in answer:
            return 1.0        
        # 2. 语义相似度评估
        similarity_score = self._calculate_semantic_similarity(answer, expected)        
        # 3. 关键词覆盖率评估
        keyword_score = self._calculate_keyword_coverage(answer, expected)        
        # 4. 关键信息完整性评估
        completeness_score = self._calculate_completeness(answer, expected)        
        # 综合评分（加权平均）
        final_score = (similarity_score * 0.4 + keyword_score * 0.3 + completeness_score * 0.3)        
        return final_score
    
    def _calculate_semantic_similarity(self, answer: str, expected: str) -> float:
        """计算语义相似度"""
        # 使用简单的文本相似度算法（实际应用中可使用更复杂的模型）
        from difflib import SequenceMatcher
        
        # 去除标点符号和空白字符
        import re
        answer_clean = re.sub(r'[^\w\u4e00-\u9fff]', '', answer)
        expected_clean = re.sub(r'[^\w\u4e00-\u9fff]', '', expected)
        
        similarity = SequenceMatcher(None, answer_clean, expected_clean).ratio()
        return similarity
    
    def _calculate_keyword_coverage(self, answer: str, expected: str) -> float:
        """计算关键词覆盖率"""
        # 提取关键信息词（名词、动词等）
        import jieba
        import jieba.posseg as pseg
        
        # 提取期望回答中的关键词（名词、动词）
        expected_keywords = set()
        for word, flag in pseg.cut(expected):
            if flag.startswith(('n', 'v', 'a')):  # 名词、动词、形容词
                expected_keywords.add(word)
        
        if not expected_keywords:
            return 0.0
        
        # 计算回答中覆盖的关键词比例
        matched_keywords = 0
        for keyword in expected_keywords:
            if keyword in answer:
                matched_keywords += 1
        
        return matched_keywords / len(expected_keywords)
    
    def _calculate_completeness(self, answer: str, expected: str) -> float:
        """计算关键信息完整性"""
        # 基于关键信息点的完整性评分        
        # 提取关键信息点（数字、专有名词等）
        import re
        
        # 提取数字信息
        expected_numbers = set(re.findall(r'\d+\.?\d*', expected))
        answer_numbers = set(re.findall(r'\d+\.?\d*', answer))
        
        # 提取品牌、产地等专有名词
        expected_entities = set(re.findall(r'[\u4e00-\u9fff]{2,4}(?=品牌|产地|系列)', expected))
        answer_entities = set(re.findall(r'[\u4e00-\u9fff]{2,4}(?=品牌|产地|系列)', answer))
        
        # 计算信息点覆盖率
        total_points = len(expected_numbers) + len(expected_entities)
        if total_points == 0:
            return 0.5  # 如果没有明显的信息点，给中等分数
        
        matched_points = len(expected_numbers.intersection(answer_numbers)) + \
                        len(expected_entities.intersection(answer_entities))
        
        return matched_points / total_points
    
    def tune_chunking_strategy(self, documents: List[Any] = None, chunking_params: Dict[str, List[int]] = None) -> None:
        """微调文本分割策略"""
        print("\n=== 微调文本分割策略 ===")         
        # 生成所有分割参数组合
        chunk_sizes = self.config["finetuning_config"]["chunking_params"]["chunk_sizes"]
        chunk_overlaps = self.config["finetuning_config"]["chunking_params"]["chunk_overlaps"]
        
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
            
        self.best_chunking = best_params
        print(f"\n最佳分割参数: {best_params}")
        print(f"最佳分数: {best_score}\n\n\n==============")
    
    def _get_chunking_documents(self) -> List[Any]:
        return 
    
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
        # 这里使用一个启发式评分：- 适中的chunk_size更好（避免过短或过长）- chunk_overlap应适中（避免过多重复）        
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
        # 准备评估数据（使用示例文档或实际文档）
        test_documents = self.rag.load_documents(self.data_path)
        if not test_documents:
            print("警告：无测试文档，回退到启发式评估")
            return self._evaluate_chunking_heuristic(params)
        
        # 使用当前参数执行分割
        chunk_size = params["chunk_size"]
        chunk_overlap = params["chunk_overlap"]   

        # 临时修改RAG的分割参数进行评估
        original_chunk_size = self.rag.text_splitter._chunk_size
        original_chunk_overlap = self.rag.text_splitter._chunk_overlap
        self.rag.set_text_splitter(chunk_size, chunk_overlap)        
        # 执行分割
        split_docs = self.rag.split_documents(test_documents)        
        # 恢复原始参数
        self.rag.set_text_splitter(original_chunk_size, original_chunk_overlap)     
        
        if not split_docs:
            print("警告：分割后无文档，回退到启发式评估")
            return self._evaluate_chunking_heuristic(params)
        
        total_score = 0.0        
        for i, doc in enumerate(split_docs):
            eval_prompt = f"""
            请评估以下文本片段作为文档分割结果的质量，从以下几个方面打分（0-10分）：\n1. 语义完整性：片段是否保持了完整的语义\n2. 边界合理性：分割是否在合理的语义边界上\n3. 信息密度：片段包含的信息量是否适中\n4. 上下文连贯性：片段内部是否连贯\n\n文本片段：\n{doc.page_content}\n\n请返回一个综合评分（0-10分），只需要数字，不需要其他说明。
            """            
            # 使用LLM生成评估
            response = self.rag.llm.invoke(eval_prompt)
            # 提取评分
            score_text = response.strip()
            # 尝试从响应中提取数字
            score_match = re.search(r'\d+(\.\d+)?', score_text)
            if score_match:
                score = float(score_match.group()) / 10.0  # 转换为0-1范围
                score = min(1.0, max(0.0, score))  # 确保在0-1范围内
                total_score += score
                print(f"分割片段 {i+1} 评分: {score:.2f}")
            else:
                print(f"警告：无法从LLM响应中提取评分，使用默认值0.5")
                total_score += 0.5
        
        # 计算平均得分
        final_score = total_score / len(split_docs)
        print(f"LLM评估最终得分: {final_score:.2f}")
        
        return final_score

    
    def _get_test_documents(self) -> List[Any]:
        """获取用于评估的测试文档"""
        # 尝试从数据路径加载文档
        if self.data_path:
            return self.rag.load_documents(self.data_path)
        else:
            raise ValueError("未指定数据路径，无法加载测试文档")
    
    def compare_embedding_models(self, model_candidates: List[str], test_data: List[Dict[str, str]]) -> str:
        """比较不同的嵌入模型"""
        print("\n=== 比较嵌入模型 ===")        
        if not model_candidates:
            print("警告：无嵌入模型候选，使用默认模型")
            return self.config["base_config"]["embedding_model"]
        
        best_model: str = None
        best_score: float = 0.0
        original_embedding_model: str = self.rag.embedding_model.model_name
        
        for i, model in enumerate(model_candidates):
            print(f"\n测试嵌入模型 {i+1}/{len(model_candidates)}: {model}")
                        
            # 评估当前模型
            current_score = self._evaluate_embedding_model(model, test_data)
            print(f"模型 {model} 得分: {current_score:.4f}")            
            # 更新最佳模型
            if current_score > best_score:
                best_score = current_score
                best_model = model
        
        # 如果没有找到最佳模型，使用第一个候选模型
        if not best_model:
            best_model = model_candidates[0] if model_candidates else self.config["base_config"]["embedding_model"]
            print(f"警告：未找到有效的最佳模型，使用默认候选模型 {best_model}")
        
        # 恢复原始嵌入模型
        self.rag.set_embedding_model(original_embedding_model)
        documents = self.rag.load_documents(self.data_path)
        split_documents = self.rag.split_documents(documents)
        store_path = os.path.join(self.config["base_config"]["vector_store_path"], self.rag.embedding_model.model_name.replace("/", "_"))
        self.rag.create_vector_store(
            split_documents, 
            store_path)
        print(f'已恢复原始嵌入模型: {original_embedding_model}'
                f'，并已重新构建向量存储到 {store_path}')
        
        # 保存最佳嵌入模型
        self.best_embedding = best_model        
        print(f"\n最佳嵌入模型: {best_model}")
        print(f"最佳得分: {best_score:.4f}\n\n\n==============")        
        return best_model
    
    def _evaluate_embedding_model(self, model: str, test_data: List[Dict[str, str]]) -> float:
        """评估单个嵌入模型的效果"""
        # 保存当前嵌入模型
        original_model = self.rag.embedding_model           
        self.rag.set_embedding_model(model)
        print(f'已切换到嵌入模型: {model}')
        
        # 使用最佳参数重新初始化向量存储
        if self.best_retrieval_params:
            k = self.best_retrieval_params["k"]
            similarity_threshold = self.best_retrieval_params["similarity_threshold"]
            print(f'使用最佳检索参数: k={k}, similarity_threshold={similarity_threshold}')
        else:
            k = 3
            similarity_threshold = 0.7
            print(f'使用默认（非最佳）检索参数: k={k}, similarity_threshold={similarity_threshold}')
        
        # 使用最佳文本分割参数重新构建向量存储
        if self.best_chunking:
            chunk_size = self.best_chunking["chunk_size"]
            chunk_overlap = self.best_chunking["chunk_overlap"]
            self.rag.set_text_splitter(chunk_size, chunk_overlap)
            print(f'使用最佳文本分割参数: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}')
        
        # 重新加载和分割文档
        documents = self.rag.load_documents(self.data_path)
        split_documents = self.rag.split_documents(documents)
        print(f'{model}的文档已分割为 {len(split_documents)}个片段')
        
        # 重新构建向量存储
        self.rag.create_vector_store(
            split_documents,
            store_path=self.config["base_config"]["vector_store_path"])
        # 加载新的向量存储
        self.rag.load_vector_store()
        
        # 评估模型效果
        total_score = 0.0
        for item in test_data:
            query = item["query"]
            expected = item["expected_answer"]
            
            # 检索相关文档
            retrieved_docs = self.rag.retrieve(query, k=k)
            
            if not retrieved_docs:
                print(f"警告：查询 '{query}' 未检索到任何文档")
                continue
            
            # 使用最佳提示模板生成回答
            if self.best_prompt:
                context = "\n".join([doc.page_content for doc in retrieved_docs])
                prompt = self.best_prompt.format(context=context, question=query)
                answer = self.rag.llm.invoke(prompt)
            else:
                # 使用默认生成方式
                answer = self.rag.generate_answer(query, retrieved_docs)
            
            # 评估回答质量
            score = self._calculate_answer_score(answer, expected)
            total_score += score
        
        # 计算平均得分
        avg_score = total_score / len(test_data) if test_data else 0.0    
             
        return avg_score            
    
    def run_full_finetuning(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行完整的微调流程"""
        print("\n=== 开始完整的RAG微调流程 ===")
        
        # 1. 微调检索参数
        self.tune_retrieval_params(
            test_data["test_queries"],
            self.config["finetuning_config"]["retrieval_param_grid"]
        )        
        # 2. 优化提示模板
        self.optimize_prompt_template(
            test_data["test_cases"],
            self.config["finetuning_config"]["prompt_candidates"]
        )        
        # 3. 微调文本分割策略
        self.tune_chunking_strategy(
            chunking_params=self.config["finetuning_config"]["chunking_params"])        
        # 4. 比较嵌入模型
        self.compare_embedding_models(
            self.config["finetuning_config"]["embedding_candidates"],
            self.config["test_data"]["test_cases"]
        )
        
        # 整合最佳参数
        best_config = {
            "retrieval_params": self.best_retrieval_params,
            "prompt_template": self.best_prompt,
            "chunking_params": self.best_chunking,
            "embedding_model": self.best_embedding
        }     
        print("\n=== 微调完成 ===")
        print(f"最佳配置: {json.dumps(best_config, ensure_ascii=False, indent=2)}")
        
        # 保存最佳配置
        if not self.args.save_config:
            best_config_dir = self.config["base_config"]["best_config_store_pth"]
            os.makedirs(best_config_dir, exist_ok=True)
            best_config_save_pth = os.path.join(
                best_config_dir,
                f"best_config.json"
            )
            with open(best_config_save_pth, "w", encoding="utf-8") as f:
                json.dump(best_config, f, ensure_ascii=False, indent=2)
        
        return best_config

def run_simulate_finetuning(self) -> Dict[str, Any]:
    """模拟微调流程，仅评估模型效果"""
    print("\n=== 开始模拟微调流程 ===")
    self.best_retrieval_params = {"k": 3, "similarity_threshold": 0.7}
    self.best_prompt = "请根据以下上下文回答问题：\n\n{context}\n\n问题：{question}\n\n回答："
    self.best_chunking = {"chunk_size": 1000, "chunk_overlap": 200}         
    self.best_embedding = self.config["base_config"]["embedding_model"]
    best_config = {
            "retrieval_params": self.best_retrieval_params,
            "prompt_template": self.best_prompt,
            "chunking_params": self.best_chunking,
            "embedding_model": self.best_embedding
        }
    print(f"模拟微调配置: {json.dumps(best_config, ensure_ascii=False, indent=2)}")
    return best_config

def arg_sparse():
    """命令行参数解析"""    
    parser = argparse.ArgumentParser(description="RAG微调系统参数配置")
    
    # 基本参数
    parser.add_argument("--config", type=str, default="config.json", 
                       help="配置文件路径 (默认: config.json)")
    parser.add_argument("--action", type=str, choices=["finetune", "evaluate", "test"], 
                       default="finetune", help="执行动作 (默认: finetune)")
    
    # 微调控制参数
    parser.add_argument("--skip-retrieval", action="store_true", 
                       help="跳过检索参数微调")
    parser.add_argument("--skip-prompt", action="store_true", 
                       help="跳過提示模板优化")
    parser.add_argument("--skip-chunking", action="store_true", 
                       help="跳过文本分割策略微调")
    parser.add_argument("--skip-embedding", action="store_true", 
                       help="跳过嵌入模型比较")
    
    # 性能参数
    parser.add_argument("--max-docs", type=int, default=100,
                       help="最大处理文档数 (默认: 100)")
    parser.add_argument("--eval-samples", type=int, default=10,
                       help="评估样本数量 (默认: 10)")
    parser.add_argument("--parallel", type=int, default=1,
                       help="并行处理数 (默认: 1)")
    
    # 输出控制
    parser.add_argument("--verbose", action="store_true", 
                       help="详细输出模式")
    parser.add_argument("--save-config", type=str, default=None,
                       help="最佳配置保存路径 (默认: best_rag_config.json)")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="日志级别 (默认: INFO)")
    
    # 特殊模式
    parser.add_argument("--quick-test", action="store_true",
                       help="快速测试模式（减少迭代次数）")
    parser.add_argument("--dry-run", action="store_true",
                       help="干运行模式（不实际执行，只显示计划）")
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = arg_sparse()
    """
    run:
    python rag_finetuning.py --config config.json --action finetune --verbose
    """
    
    # 设置日志级别    
    logging.basicConfig(level=getattr(logging, args.log_level))
    
    # 加载配置文件
    config_file = args.config
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"配置加载成功: {config_file}")
    except FileNotFoundError:
        print(f"错误：配置文件 {config_file} 不存在")
        return
    except json.JSONDecodeError:
        print(f"错误：配置文件 {config_file} 格式错误")
        return
    
    # 原始数据路径
    data_path = config["base_config"]["data_path"]
    
    # 根据参数调整配置
    if args.quick_test:
        print("快速测试模式：减少迭代次数")
        # 减少参数组合数量
        config["finetuning_config"]["retrieval_param_grid"]["k"] = [2, 3]
        config["finetuning_config"]["retrieval_param_grid"]["similarity_threshold"] = [0.7]
        config["finetuning_config"]["chunking_params"]["chunk_sizes"] = [500, 1000]
        config["finetuning_config"]["chunking_params"]["chunk_overlaps"] = [100]
    
    # 创建微调器实例
    finetuner = RAGFinetuner(config, data_path=data_path, args=args)
    
    # 根据action参数执行不同操作
    if args.action == "finetune":
        if args.dry_run:
            print("干运行模式：显示计划但不执行")
            print("计划执行完整微调流程")
            if args.skip_retrieval:
                print("- 跳过检索参数微调")
            if args.skip_prompt:
                print("- 跳过提示模板优化")
            if args.skip_chunking:
                print("- 跳过文本分割策略微调")
            if args.skip_embedding:
                print("- 跳过嵌入模型比较")
            return
        
        # 运行完整微调流程
        best_config = finetuner.run_full_finetuning(config["test_data"])
        
        # 保存最佳配置
        if args.save_config:
            save_path = os.path.join(config["base_config"]["best_config_store_pth"], args.save_config)
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(best_config, f, ensure_ascii=False, indent=2)
                print(f"最佳配置已保存到: {save_path}")
            except Exception as e:
                print(f"保存配置时出错: {e}")
    
    elif args.action == "evaluate":
        print("评估模式：运行系统评估")
        # 这里可以添加评估逻辑
        
    elif args.action == "test":
        print("测试模式：运行基本功能测试")
        # 这里可以添加测试逻辑
    
    else:
        print(f"未知操作: {args.action}")


if __name__ == "__main__":
    main()
