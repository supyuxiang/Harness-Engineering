# Harness-Engineering

本仓库是 Harness Engineering 考核项目提交代码，核心目标是构建并评估一个基于检索增强的文本分类 `Harness`。

## 项目概览

- 主要实现：`solution.py` 中的 `MyHarness`
- 关键思路：先从历史标注样本中检索相关示例，再用 LLM 在候选标签内做受约束分类
- 评估脚本：`run.py`（由作者自行实现）
- baseline 脚本：`run_baseline.py`（由作者自行实现）

## 目录说明

- `solution.py`：主实现版本（默认提交版本）
- `solution_baseline.py`：基础 baseline 版本
- `solution_actor_verifier.py` / `solution_majorityvoting.py` / `solution_final.py` / `solution_acc0d8.py`：不同实验方案
- `harness_base.py`：基类定义
- `llm_client.py`：LLM 调用与 token 计数相关接口
- `run.py`：评测主脚本
- `data/`：训练与验证数据
- `tokenizer/`：分词器配置
- `topk_sweep.py` / `topk_sweep_results.csv` / `topk_sweep_plot.png`：`top_k` 参数实验结果

## 环境准备

```bash
pip install -r requirements.txt
```

## 快速开始

建议先运行 baseline：

```bash
python run_baseline.py
```

再运行主方案评测：

```bash
python run.py
```

## `MyHarness` 核心流程

1. `update(text, label)`：接收训练样本并维护内存索引与标签顺序。
2. `retrieve(text)`：按相似度检索 top-k 示例，并进行简单去冗余。
3. `predict(text)`：在 token 预算约束下动态构造 few-shot prompt，调用 LLM 得到标签。

## 可调参数

- `tokenize_fn_name`：分词策略（如 `default`、`char3`、`word_char_mix`）
- `sim_fn_name`：相似度函数（如 `jaccard`、`overlap`）
- `top_k`：检索样本上限，影响信息量与 token 开销

## 备注

- 本项目包含多版实验代码，默认以 `solution.py` 作为主提交实现。
- 若需复现实验图表，可运行 `topk_sweep.py`。
