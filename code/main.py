#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG 卷烟知识库主程序
基于rag_base.py和rag_finetuning.py的完整实现
"""

import json
import argparse
import os
import sys
from typing import Dict, Any, List

from rag_base import BasicRAG, load_config as load_rag_config
from rag_finetuning import RAGFinetuner


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ 配置文件加载成功: {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ 配置文件格式错误: {e}")
        sys.exit(1)


def initialize_rag_system(config: Dict[str, Any]) -> BasicRAG:
    """初始化RAG系统"""
    print("\n=== 初始化RAG系统 ===")
    
    # 创建RAG实例
    rag = BasicRAG(config)
    
    # 初始化组件
    rag.init_components()
    
    # 检查数据路径
    data_path = config["base_config"]["data_path"]
    if not os.path.exists(data_path):
        print(f"⚠️  数据文件不存在: {data_path}")
        print("请确保数据文件路径正确，或使用 --prepare 选项准备数据")
        return rag
    
    # 加载和准备文档
    documents = rag.load_documents(data_path)
    if documents:
        print(f"✅ 成功加载 {len(documents)} 个文档")
        
        # 分割文档
        split_docs = rag.split_documents(documents)
        print(f"✅ 文档分割完成，共 {len(split_docs)} 个片段")
        
        # 保存分割后的文档
        save_path = config["base_config"]["save_docs_path"]
        os.makedirs(save_path, exist_ok=True)
        rag.save_documents(split_docs, os.path.join(save_path, "split_docs.json"))
        
        # 创建或加载向量存储
        vector_store_path = config["base_config"]["vector_store_path"]
        full_store_path = os.path.join(vector_store_path, rag.embedding_model.model_name.replace("/", "_"))
        
        if os.path.exists(full_store_path) and any(
            os.path.exists(os.path.join(full_store_path, file)) 
            for file in ["index.faiss", "index.pkl"]
        ):
            print(f"📁 使用现有向量存储: {full_store_path}")
            rag.load_vector_store(full_store_path)
        else:
            print(f"🔄 创建新向量存储: {full_store_path}")
            rag.create_vector_store(split_docs, vector_store_path)
    else:
        print("⚠️  未加载到任何文档")
    
    return rag


def run_interactive_query(rag: BasicRAG, config: Dict[str, Any]):
    """运行交互式查询"""
    print("\n=== 交互式查询模式 ===")
    print("输入查询问题（输入 'quit' 或 'exit' 退出）")
    
    retrieval_k = config["base_config"]["retrieval_k"]
    
    while True:
        try:
            query = input("\n🔍 请输入查询: ").strip()
            
            if query.lower() in ['quit', 'exit', '退出']:
                print("👋 退出交互式查询")
                break
            
            if not query:
                continue
            
            print(f"\n📝 查询: {query}")
            
            # 执行RAG流水线
            answer = rag.rag_pipeline(query, k=retrieval_k)
            print(f"💬 回答: {answer}")
            
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，退出交互式查询")
            break
        except Exception as e:
            print(f"❌ 查询过程中出错: {e}")


def run_batch_queries(rag: BasicRAG, config: Dict[str, Any]):
    """运行批量查询"""
    print("\n=== 批量查询模式 ===")
    
    # 使用测试数据中的查询
    test_queries = config["test_data"]["test_queries"]
    retrieval_k = config["base_config"]["retrieval_k"]
    
    print(f"📋 将执行 {len(test_queries)} 个测试查询")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- 查询 {i}/{len(test_queries)} ---")
        print(f"📝 查询: {query}")
        
        try:
            answer = rag.rag_pipeline(query, k=retrieval_k)
            print(f"💬 回答: {answer}")
        except Exception as e:
            print(f"❌ 查询失败: {e}")


def run_finetuning_pipeline(config: Dict[str, Any]):
    """运行RAG微调流水线"""
    print("\n=== RAG微调流水线 ===")
    
    # 创建微调器实例
    try:
        finetuner = RAGFinetuner(config, config["base_config"]["data_path"])
        print("✅ 微调器初始化成功")
    except Exception as e:
        print(f"❌ 微调器初始化失败: {e}")
        return
    
    # 准备微调数据
    finetuning_data = {
        "test_queries": config["test_data"]["test_queries"],
        "test_cases": config["test_data"]["test_cases"],
        "retrieval_param_grid": config["finetuning_config"]["retrieval_param_grid"],
        "prompt_candidates": config["finetuning_config"]["prompt_candidates"],
        "chunking_params": config["finetuning_config"]["chunking_params"],
        "embedding_candidates": config["finetuning_config"]["embedding_candidates"]
    }
    
    print("📊 微调数据准备完成")
    
    # 执行微调步骤
    try:
        # 1. 微调检索参数
        print("\n1. 微调检索参数...")
        finetuner.tune_retrieval_params(
            finetuning_data["test_queries"],
            finetuning_data["retrieval_param_grid"]
        )
        
        # 2. 优化提示模板
        print("\n2. 优化提示模板...")
        finetuner.optimize_prompt_template(
            finetuning_data["test_cases"],
            finetuning_data["prompt_candidates"]
        )
        
        # 3. 微调文本分割策略
        print("\n3. 微调文本分割策略...")
        finetuner.tune_chunking_strategy(
            chunking_params=finetuning_data["chunking_params"]
        )
        
        # 4. 比较嵌入模型
        print("\n4. 比较嵌入模型...")
        best_embedding_model = finetuner.compare_embedding_models(
            finetuning_data["embedding_candidates"],
            finetuning_data["test_cases"]
        )
        
        # 汇总最佳配置
        best_config = {
            "retrieval_params": finetuner.best_retrieval_params,
            "prompt_template": finetuner.best_prompt,
            "chunking_params": finetuner.best_chunking,
            "embedding_model": best_embedding_model
        }
        
        # 保存最佳配置
        best_config_path = config["base_config"]["best_config_store_pth"]
        os.makedirs(best_config_path, exist_ok=True)
        
        best_config_file = os.path.join(best_config_path, "best_rag_config.json")
        with open(best_config_file, "w", encoding="utf-8") as f:
            json.dump(best_config, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 微调完成！最佳配置已保存到: {best_config_file}")
        print("📋 最佳配置摘要:")
        print(f"   - 检索参数: {best_config['retrieval_params']}")
        print(f"   - 文本分割: {best_config['chunking_params']}")
        print(f"   - 嵌入模型: {best_config['embedding_model']}")
        
    except Exception as e:
        print(f"❌ 微调过程中出错: {e}")
        import traceback
        traceback.print_exc()


def evaluate_rag_system(rag: BasicRAG, config: Dict[str, Any]):
    """评估RAG系统性能"""
    print("\n=== RAG系统评估 ===")
    
    test_cases = config["test_data"]["test_cases"]
    
    if not test_cases:
        print("⚠️  无测试用例可用")
        return
    
    print(f"📊 使用 {len(test_cases)} 个测试用例进行评估")
    
    try:
        results = rag.evaluate(test_cases)
        accuracy = results.get("accuracy", 0.0)
        
        print(f"📈 评估结果:")
        print(f"   - 准确率: {accuracy:.2%}")
        print(f"   - 测试用例数: {len(test_cases)}")
        
        # 详细测试结果
        #! 这里仍需要评估，因为下面的评价“单一句子匹配”不合理且不准确不够反映模型的性能
        print("\n📋 详细测试结果:")
        for i, case in enumerate(test_cases, 1):
            print(f"\n--- 测试用例 {i} ---")
            print(f"查询: {case['query']}")
            print(f"期望回答: {case['expected_answer']}")
            
            try:
                actual_answer = rag.rag_pipeline(case['query'])
                print(f"实际回答: {actual_answer}")
                
                # 简单匹配检查
                if case['expected_answer'] in actual_answer:
                    print("✅ 匹配成功")
                else:
                    print("❌ 匹配失败")
            except Exception as e:
                print(f"❌ 测试失败: {e}")
                
    except Exception as e:
        print(f"❌ 评估过程中出错: {e}")


def main():
    """主函数"""
    """
    命令行使用示例：
    
    # ✅ 交互式查询模式 
    # python main.py --action interactive
    
    # ✅ 批量查询模式（执行所有测试查询）
    # python main.py --action batch
    
    # ✅ 批量查询模式（执行单个查询）
    # python main.py --action batch --query "双喜品牌的卷烟产品有哪些？"
    
    # ✅ RAG微调模式
    # python main.py --action finetune
    
    # ❌ 系统评估模式!
    # python main.py --action evaluate
    
    # ❌ 数据准备模式
    # python main.py --action prepare
    
    # ❌ 使用自定义配置文件
    # python main.py --config custom_config.json --action interactive
    """
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
        default="interactive", 
        choices=["interactive", "batch", "finetune", "evaluate", "prepare"],
        help="执行动作: interactive(交互查询), batch(批量查询), finetune(微调), evaluate(评估), prepare(准备数据)"
    )
    parser.add_argument(
        "--query", 
        type=str, 
        help="直接执行单个查询（仅用于batch模式）"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 根据动作执行相应操作
    if args.action == "prepare":
        # 仅准备数据，不执行查询
        print("🔄 准备RAG系统数据...")
        rag = initialize_rag_system(config)
        print("✅ 数据准备完成")
    
    elif args.action == "interactive":
        # 交互式查询模式
        rag = initialize_rag_system(config)
        run_interactive_query(rag, config)
    
    elif args.action == "batch":
        # 批量查询模式
        rag = initialize_rag_system(config)
        if args.query:
            # 执行单个查询
            print(f"\n🔍 执行单个查询: {args.query}")
            answer = rag.rag_pipeline(args.query, k=config["base_config"]["retrieval_k"])
            print(f"💬 回答: {answer}")
        else:
            # 执行批量查询
            run_batch_queries(rag, config)
    
    elif args.action == "finetune":
        # RAG微调模式
        run_finetuning_pipeline(config)
    
    elif args.action == "evaluate":
        # 评估模式
        rag = initialize_rag_system(config)
        evaluate_rag_system(rag, config)
    
    else:
        print(f"❌ 未知动作: {args.action}")
        parser.print_help()


if __name__ == "__main__":
    main()
