# RAG 卷烟知识库系统

## 项目概述

本项目提供了一个完整的RAG（检索增强生成）系统，专门用于卷烟知识库的构建、查询和优化。系统包含基础RAG实现、微调功能和多种操作模式，支持从数据准备到系统评估的完整工作流。

## 项目结构

```
.
├── main.py              # 主入口程序（支持多种操作模式）
├── rag_base.py          # 基础RAG实现框架
├── rag_finetuning.py    # RAG微调工具
├── config.json          # 系统配置文件
├── best_config/         # 最佳配置存储目录
│   └── best_config.json # 微调后的最佳配置
└── README.md            # 项目说明文档
```

## 核心功能模块

### 1. 基础RAG实现 (`rag_base.py`)

**主要功能：**
- **文档加载**：支持从Excel文件加载卷烟数据，自动处理多工作表
- **智能分割**：Excel数据按行分割，文本数据使用可配置的分割策略
- **向量存储**：基于FAISS的向量存储，支持智能检测和重用现有存储
- **检索增强**：相似度搜索结合上下文生成回答
- **系统评估**：基于测试用例的准确率评估

**关键特性：**
- 自动处理Excel文件的元数据提取
- 向量存储的智能管理（检测现有存储）
- 可配置的文本分割参数
- 支持多种嵌入模型

### 2. RAG微调工具 (`rag_finetuning.py`)

**微调功能：**
- **检索参数优化**：网格搜索k值和相似度阈值
- **提示模板优化**：比较不同提示模板的生成效果
- **分割策略微调**：优化chunk_size和chunk_overlap参数
- **嵌入模型比较**：评估不同嵌入模型的检索性能
- **完整微调流程**：整合所有微调步骤的自动化流程

**评估方法：**
- 语义相似度计算
- 关键词覆盖率评估
- 关键信息完整性检查
- LLM辅助的质量评估

### 3. 主程序 (`main.py`)

**操作模式：**
- **interactive**：交互式查询模式
- **batch**：批量查询模式（支持单个查询）
- **finetune**：RAG微调模式
- **evaluate**：系统评估模式
- **prepare**：数据准备模式

## 配置文件详解 (`config.json`)

### 基础配置 (`base_config`)
```json
{
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "embedding_model": "BAAI/bge-small-zh-v1.5",
    "cache_folder": "embedding_cache",
    "llm_model": "qwen2:0.5b",
    "llm_cache": "llm_cache",
    "temperature": 0.1,
    "vector_store_path": "vector_store",
    "debug_save_pth": "debug_save",
    "use_llm_evaluation": true,
    "data_path": "../cigaratte_data.xlsx",
    "save_docs_path": "saved_documents",
    "retrieval_k": 3,
    "show_retrieved_docs": false,
    "best_config_store_pth": "best_config"
}
```

### 微调配置 (`finetuning_config`)
- **retrieval_param_grid**：检索参数网格搜索范围
- **prompt_candidates**：提示模板候选列表
- **chunking_params**：文本分割参数范围
- **embedding_candidates**：嵌入模型候选列表

### 测试数据 (`test_data`)
- **test_queries**：测试查询列表
- **test_cases**：包含查询和期望回答的测试用例

## 快速开始

### 1. 环境准备

```bash
# 安装核心依赖
pip install langchain langchain-community langchain-core faiss-cpu

# 安装嵌入模型支持
pip install langchain-huggingface

# 安装LLM支持（使用Ollama）
pip install langchain-ollama

# 安装数据处理依赖
pip install pandas openpyxl jieba
```

### 2. 数据准备

确保数据文件 `../cigaratte_data.xlsx` 存在，或修改 `config.json` 中的 `data_path` 配置。

### 3. 运行系统

#### 交互式查询模式
```bash
python main.py --action interactive
```

#### 批量查询模式
```bash
# 执行所有测试查询
python main.py --action batch

# 执行单个查询
python main.py --action batch --query "双喜品牌的卷烟产品有哪些？"
```

#### RAG微调模式
```bash
python main.py --action finetune
```

#### 系统评估模式
```bash
python main.py --action evaluate
```

#### 数据准备模式
```bash
python main.py --action prepare
```

## 详细使用说明

### 交互式查询模式

在交互式模式下，系统会：
1. 初始化RAG系统组件
2. 加载和准备文档数据
3. 创建或加载向量存储
4. 进入交互式查询循环

用户可以直接输入查询问题，系统会返回基于卷烟知识库的回答。

### RAG微调流程

微调过程包含以下步骤：

1. **检索参数微调**
   - 网格搜索k值（2,3,4,5）和相似度阈值（0.7,0.8,0.9）
   - 基于检索质量评分选择最佳参数

2. **提示模板优化**
   - 比较6种不同的提示模板
   - 基于回答质量评分选择最佳模板

3. **文本分割策略微调**
   - 测试不同的chunk_size（500,1000,1500,2000）和chunk_overlap（100,200,300）
   - 使用LLM评估分割质量

4. **嵌入模型比较**
   - 比较4种不同的中文嵌入模型
   - 基于检索效果选择最佳模型

5. **最佳配置保存**
   - 整合所有最佳参数
   - 保存到 `best_config/best_config.json`

### 系统评估

评估功能基于测试用例检查系统的：
- **准确率**：期望回答与实际回答的匹配程度
- **检索质量**：相关文档的检索效果
- **生成质量**：回答的准确性和完整性

## 配置优化建议

### 性能优化
- **chunk_size**：根据文档类型调整，卷烟数据建议500-1500
- **chunk_overlap**：设置为chunk_size的10%-20%
- **retrieval_k**：根据查询复杂度调整，一般3-5个文档

### 质量优化
- **嵌入模型**：优先选择 `BAAI/bge-large-zh-v1.5` 获得更好效果
- **提示模板**：选择约束性强的模板减少幻觉
- **温度参数**：设置为较低值（0.1-0.3）提高回答稳定性

## 故障排除

### 常见问题

1. **数据文件不存在**
   ```
   ❌ 数据文件不存在: ../cigaratte_data.xlsx
   ```
   解决方案：检查文件路径或使用 `--action prepare` 准备数据

2. **向量存储加载失败**
   解决方案：系统会自动重新创建向量存储

3. **依赖包缺失**
   解决方案：使用 `pip install` 安装缺失的包

### 日志说明

- `✅`：成功操作
- `⚠️`：警告信息
- `❌`：错误信息
- `🔄`：正在进行中的操作
- `📁`：文件/目录操作
- `📝`：查询相关
- `💬`：回答相关
- `📊`：数据/统计相关
- `📈`：评估结果

## 扩展开发

### 添加新功能
1. 在 `rag_base.py` 中添加新的RAG组件
2. 在 `rag_finetuning.py` 中添加对应的微调功能
3. 在 `main.py` 中添加新的操作模式

### 支持新数据格式
1. 在 `rag_base.py` 的 `load_documents` 方法中添加新的文档加载器
2. 实现对应的文档分割策略

### 自定义评估指标
1. 在 `rag_finetuning.py` 中添加新的评估方法
2. 集成到微调流程中

## 技术架构

```
用户输入 → 主程序(main.py) → RAG系统(rag_base.py)
                              ↓
                   微调工具(rag_finetuning.py)
                              ↓
                    最佳配置(best_config/)
```

# 待完成事项
- [ ] 完善运行/交互功能
- [ ] 增加数据库数据
  - 爬网上数据
  - 人工增添
- [ ] 完善系统评估指标
- [ ] 添加模型监控功能
- [ ] 优化批量查询模式
- [ ] 支持多语言文档处理

## 许可证

MIT


