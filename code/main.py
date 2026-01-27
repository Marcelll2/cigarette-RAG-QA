#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG 卷烟知识库主程序
"""

import json
import argparse
from rag_base import BasicRAG
from rag_finetuning import RAGFinetuner


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def run_rag_pipeline(config: dict):
    """运行RAG流水线"""
    print("\n=== 运行RAG流水线 ===")
    
    # 创建RAG实例
    rag = BasicRAG(config["base_config"])
    rag.init_components()
    
    # 示例查询
    queries = [
        "卷烟知识库的主要功能是什么？",
        "如何使用卷烟知识库进行查询？"
    ]
    
    for query in queries:
        print(f"\n查询: {query}")
        answer = rag.rag_pipeline(query)
        print(f"回答: {answer}")


def run_finetuning(config: dict):
    """运行RAG微调"""
    print("\n=== 运行RAG微调 ===")
    
    # 创建微调器实例
    finetuner = RAGFinetuner(config["base_config"])
    
    # 准备微调数据
    finetuning_data = {
        "test_queries": config["test_data"]["test_queries"],
        "test_cases": config["test_data"]["test_cases"],
        "retrieval_param_grid": config["finetuning_config"]["retrieval_param_grid"],
        "prompt_candidates": config["finetuning_config"]["prompt_candidates"],
        "chunking_params": config["finetuning_config"]["chunking_params"],
        "embedding_candidates": config["finetuning_config"]["embedding_candidates"],
        "sample_documents": []  # 实际使用时应提供样本文档
    }
    
    # 运行微调
    best_config = finetuner.run_full_finetuning(finetuning_data)
    
    # 保存最佳配置
    with open("best_rag_config.json", "w", encoding="utf-8") as f:
        json.dump(best_config, f, ensure_ascii=False, indent=2)
    
    print("\n最佳配置已保存到 best_rag_config.json")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="RAG 卷烟知识库系统")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.json", 
        help="配置文件路径"
    )
    parser.add_argument(
        "--action", 
        type=str, 
        default="run", 
        choices=["run", "finetune"],
        help="执行动作: run(运行RAG) 或 finetune(微调RAG)"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 执行相应动作
    if args.action == "run":
        run_rag_pipeline(config)
    elif args.action == "finetune":
        run_finetuning(config)


if __name__ == "__main__":
    main()
