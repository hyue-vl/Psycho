# Psycho-World

本仓库实现了一个基于 DreamCUB / WKM 思想的心理干预 POMDP 框架。核心模块如下：

- `psycho_world/data`: COKE 认知链条 ETL（向量库 + Neo4j）与对话回溯标注脚本。
- `psycho_world/knowledge`: DKB/PKB 混合检索器，可针对当前 Belief 检索相似案例。
- `psycho_world/training`: 认知编码器 `q_phi` 及世界模拟器 `p_theta` 的 LoRA 训练脚本。
- `psycho_world/envs`: 基于 Gymnasium 的 Psycho-World 环境，内置信念奖励函数。
- `psycho_world/agents`: “Dreaming” 代理，使用 MCTS 风格推演选择最优干预。
- `psycho_world/memory`: 通过 MemGPT 维护个性化 PKB。
- `train_psycho_agent.py`: 演示如何将各模块串联并运行一次交互循环。

## 快速上手

```bash
pip install -e .
python psycho_world/data/etl_coke.py              # 构建 DKB
python psycho_world/data/data_annotation.py       # 生成信念标注
python psycho_world/training/train_cognitive_encoder.py
python psycho_world/training/train_world_model.py
python train_psycho_agent.py
```

> 由于涉及 LLM/Neo4j/向量库，请预先配置对应的 API Key 与数据库服务。