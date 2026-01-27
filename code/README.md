# RAG 卷烟知识库微调系统

## 项目概述

本项目提供了一个完整的RAG（检索增强生成）微调框架，用于构建和优化卷烟知识库的RAG系统。该框架包含基础RAG实现和全面的微调功能，可以帮助用户优化RAG系统的各个组件。

## 项目结构

```
.
├── main.py              # 主入口程序
├── rag_base.py          # 基础RAG实现
├── rag_finetuning.py    # RAG微调工具
├── config.json          # 配置文件
└── README.md            # 项目说明
```

## 功能模块

### 1. 基础RAG实现 (`rag_base.py`)

- **文档加载**：支持从目录加载文本文件
- **文本分割**：可配置的文本分割策略
- **向量存储**：基于FAISS的向量存储管理
- **检索功能**：支持相似度搜索
- **回答生成**：结合检索结果生成回答
- **评估功能**：简单的RAG系统评估

### 2. RAG微调工具 (`rag_finetuning.py`)

- **检索参数微调**：优化k值、相似度阈值等检索参数
- **提示模板优化**：比较不同提示模板的效果
- **文本分割策略微调**：优化chunk_size和chunk_overlap
- **嵌入模型比较**：评估不同嵌入模型的效果
- **完整微调流程**：整合所有微调步骤的完整流程

## 配置文件 (`config.json`)

配置文件包含三个主要部分：

1. **base_config**：基础RAG配置
2. **finetuning_config**：微调参数配置
3. **test_data**：测试数据

### 基础配置示例

```json
{
    "base_config": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "embedding_model": "BAAI/bge-small-zh-v1.5",
        "llm_model": "qwen2:0.5b",
        "temperature": 0.1
    }
}
```

## 使用方法

### 1. 安装依赖

```bash
pip install langchain langchain_community langchain_core faiss-cpu huggingface-hub
```

### 2. 运行RAG流水线

```bash
python main.py --action run
```

### 3. 运行RAG微调

```bash
python main.py --action finetune
```

### 4. 自定义配置

修改 `config.json` 文件中的参数，根据实际需求调整：

- 调整 `chunk_size` 和 `chunk_overlap` 优化文本分割
- 尝试不同的嵌入模型，如 `BAAI/bge-large-zh-v1.5`
- 优化提示模板，提高生成质量
- 调整检索参数 `k` 值，平衡相关性和多样性

## 微调流程

1. **检索参数微调**：通过网格搜索找到最佳的k值和相似度阈值
2. **提示模板优化**：比较不同提示模板的生成效果
3. **文本分割策略微调**：找到最佳的chunk_size和chunk_overlap组合
4. **嵌入模型比较**：评估不同嵌入模型的检索效果
5. **整合最佳配置**：将所有最佳参数整合到最终配置中

## 评估指标

- **检索效果**：相关性评分、召回率、精确率
- **生成质量**：BLEU、ROUGE、语义相似度
- **系统性能**：响应时间、资源消耗

## 扩展建议

1. **添加更多评估指标**：如BLEURT、METEOR等高级评估指标
2. **支持更多文档类型**：PDF、Word、Excel等
3. **实现增量更新**：支持向量库的增量更新
4. **添加监控功能**：实时监控RAG系统性能
5. **支持多模态**：整合图像、表格等多模态数据

## 注意事项

1. 确保Python版本 >= 3.8
2. 安装所有必要的依赖包
3. 根据实际硬件资源选择合适的模型大小
4. 微调过程可能需要较长时间，建议在性能较好的机器上运行
5. 定期评估和更新RAG系统，以适应新的数据和需求

## 示例输出

```
=== 运行RAG微调 ===

=== 微调检索参数 ===

测试组合 1/12: {'k': 2, 'similarity_threshold': 0.7}
组合分数: 0.9333333333333333

测试组合 2/12: {'k': 2, 'similarity_threshold': 0.8}
组合分数: 0.9333333333333333

...

最佳检索参数: {'k': 5, 'similarity_threshold': 0.7}
最佳分数: 1.0

=== 优化提示模板 ===

测试提示模板 1/3:
使用以下上下文来回答用户的问题。上下文：{context} 问题：{question} 请使用中文回答，并且只基于提供的上下文。...
模板分数: 1.0

...

最佳提示模板:
使用以下上下文来回答用户的问题。上下文：{context} 问题：{question} 请使用中文回答，并且只基于提供的上下文。
最佳分数: 1.0

=== 微调完成 ===
最佳配置: {
    "retrieval_params": {
        "k": 5,
        "similarity_threshold": 0.7
    },
    "prompt_template": "使用以下上下文来回答用户的问题。上下文：{context} 问题：{question} 请使用中文回答，并且只基于提供的上下文。",
    "chunking_params": {
        "chunk_size": 1000,
        "chunk_overlap": 200
    },
    "embedding_model": "BAAI/bge-large-zh-v1.5",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "llm_model": "qwen2:0.5b",
    "temperature": 0.1
}

最佳配置已保存到 best_rag_config.json
```

## 许可证

MIT
