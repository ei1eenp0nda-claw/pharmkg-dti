# PharmKG-DTI

## 项目概述

**PharmKG-DTI** 是一个用于药物-靶点相互作用（Drug-Target Interaction, DTI）预测的生产级知识图谱系统。该项目整合了异构图神经网络、多模态特征融合和对比学习，旨在为药物发现和重定位提供高精度的计算预测工具。

## 核心技术

- **异构图神经网络 (Heterogeneous GNN)**: 处理多种节点类型（药物、蛋白、疾病）和关系类型
- **Graph Transformer + GraphSAGE 双视角架构**: 同时捕获局部和全局网络结构
- **多模态特征融合**: 整合分子结构、蛋白序列和网络拓扑信息
- **对比学习增强**: 提高模型鲁棒性和泛化能力
- **知识图谱嵌入**: 基于 PyKEEN 的链接预测框架

## 实体类型

```
├── Drug (药物)
├── Protein/Target (靶点蛋白)
├── Disease (疾病)
├── Pathway (通路)
├── Side Effect (副作用)
└── Gene (基因)
```

## 关系类型

```
├── Drug-Target Interaction (DTI) - 核心预测目标
├── Drug-Drug Interaction (DDI)
├── Protein-Protein Interaction (PPI)
├── Drug-Disease Association
├── Protein-Disease Association
├── Drug-Side Effect
└── Protein-Pathway
```

## 项目结构

```
pharmkg-dti/
├── data/                      # 数据目录
│   ├── raw/                   # 原始数据
│   ├── processed/             # 处理后数据
│   └── embeddings/            # 预训练嵌入
├── src/                       # 源代码
│   ├── data/                  # 数据加载与预处理
│   ├── models/                # 模型实现
│   ├── training/              # 训练脚本
│   ├── evaluation/            # 评估工具
│   └── utils/                 # 工具函数
├── configs/                   # 配置文件
├── notebooks/                 # Jupyter notebooks
├── tests/                     # 单元测试
├── scripts/                   # 脚本文件
├── results/                   # 实验结果
├── checkpoints/               # 模型检查点
└── docs/                      # 文档
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 数据准备

```bash
python scripts/download_datasets.py
python scripts/preprocess_data.py
```

### 训练模型

```bash
python src/training/train.py --config configs/dhgt_dti.yaml
```

### 评估模型

```bash
python src/evaluation/evaluate.py --checkpoint checkpoints/best_model.pt
```

## 性能指标

在公开数据集上的性能（持续更新）：

| 数据集 | AUC | AUPR | Hits@10 |
|--------|-----|------|---------|
| DrugBank | TBD | TBD | TBD |
| BioKG | TBD | TBD | TBD |
| OpenBioLink | TBD | TBD | TBD |

## 引用

如果您使用了本项目，请引用：

```bibtex
@software{pharmkg_dti,
  title={PharmKG-DTI: A Production-Ready Knowledge Graph System for Drug-Target Interaction Prediction},
  author={Your Name},
  year={2026}
}
```

## 许可证

MIT License
