# MyHarness 说明文档

这个文档用于介绍 `solution.py` 中的 `MyHarness`（一个基于轻量检索 + LLM 判别的文本分类器），方便你向他人说明它的设计、流程和可配置点。

## 1. 这是什么

`MyHarness` 继承自 `Harness`，实现了一个“先检索、再让模型分类”的方案：

- 训练阶段（`update`）：持续接收 `(text, label)` 样本并建立内存索引。
- 预测阶段（`predict`）：从历史样本中检索最相关示例，拼接成 few-shot prompt，让 LLM 从候选标签中选一个。

它的目标是：在 token 预算受限的情况下，不把全部 memory 扔进 prompt，而是只保留“最相关的子集”。

---

## 2. 整体结构（你可以这样讲）

`MyHarness` 的关键组件：

- `build_tokenizer()`：构建文本分词函数。
- `build_sim_fn()`：构建样本相似度函数。
- `update()`：写入样本缓存与标签顺序。
- `retrieve()`：按相似度检索 top-k 示例，并做简单去冗余。
- `predict()`：动态拼 prompt，自动适配 token 上限后调用 LLM。
- `extract_pred_lable_from_response()`：从模型回复中稳健提取最终标签。

---

## 3. 核心流程

### 3.1 `update(text, label)`

每次喂入一个训练样本时：

1. 记录到 `self.memory`（父类逻辑）。
2. 用 `label_set + label_order` 去重并保持标签出现顺序。
3. 预计算该文本的 token 集合，保存进 `example_cache`：
   - `{"text": ..., "label": ..., "tokens": ...}`

这样在预测时就不需要重复分词所有历史样本。

### 3.2 `retrieve(text)`

给定待分类文本：

1. 对 query 分词。
2. 遍历 `example_cache`，用 `sim_fn(query_tokens, example_tokens)` 打分。
3. 仅保留分数 `> 0` 的样本。
4. 按 `(similarity, idx)` 降序排序（相似度高优先，同分时后加入样本优先）。
5. 做简单去冗余：每个 label 最多保留 3 条。
6. 取前 `top_k` 条作为 few-shot 示例。

若所有样本分数都为 0，则回退为“最近的 `top_k` 条样本”。

### 3.3 `predict(text)`

1. 组装候选标签列表（来自 `label_order`）。
2. 先 `retrieve` 拿示例，再构造 prompt。
3. 使用 while 循环逐步减少示例数 `k`，直到 `count_messages_tokens(messages)` 不超过 `max_prompt_tokens`。
4. 若 `k` 缩到 0，走无示例兜底 prompt（仅给候选标签 + 输入）。
5. 调用 `call_llm(messages)`，最后由 `extract_pred_lable_from_response` 提取标签。

---

## 4. 可配置项

`MyHarness` 初始化参数：

- `tokenize_fn_name`（默认 `default`）
  - `default`：词级 token（去停用词、去短词）
  - `char3`：字符 3-gram
  - `word_char_mix`：词级 + 字符 3-gram 并集
- `sim_fn_name`（默认 `jaccard`）
  - `jaccard`：`|A∩B| / |A∪B|`
  - `overlap`：`|A∩B| / min(|A|, |B|)`
- `top_k`（默认 `35`）
  - 控制检索示例数上限，越大通常信息越多，但更吃 token。

---

## 5. 标签提取策略（鲁棒性设计）

`extract_pred_lable_from_response()` 会按优先级尝试：

1. 回复整体是否就是某个标签。
2. 正则抽取 `Label:` / `Final label:` / `Answer:` 行（取最后一个匹配）。
3. 看最后一行是否是纯标签。
4. 用词边界匹配检查回复中是否出现标签。
5. 都失败则回退到 `labels[0]`。

这能应对模型输出“解释 + 标签”的多种格式。

---

## 6. 方案优点与边界

### 优点

- **轻量可控**：只依赖标准库 + `numpy`，结构简单。
- **token 友好**：检索子集 + 动态裁剪示例，减少超预算风险。
- **可解释**：可直接查看检索到的示例和提示词模板。
- **可扩展**：可继续加新的 tokenizer/similarity/retrieval 策略。

### 边界

- 检索是词面/字符层面，语义泛化有限。
- 每标签最多 3 条的去冗余规则是启发式，不一定全局最优。
- `top_k` 需按任务分布调参，不同数据集最优值可能差异较大。

---

## 7. 快速使用（示意）

```python
harness = MyHarness(
    call_llm=call_llm,
    count_tokens=count_tokens,
    count_messages_tokens=count_messages_tokens,
    max_prompt_tokens=max_prompt_tokens,
    tokenize_fn_name="word_char_mix",
    sim_fn_name="jaccard",
    top_k=35,
)

for text, label in train_data:
    harness.update(text, label)

pred = harness.predict("your input text")
```

---

## 8. 你对外介绍时可直接用的一句话

“`MyHarness` 是一个检索增强的 few-shot 分类器：先从历史标注样本里找最相关例子，再在 token 预算内动态组 prompt，让 LLM 只在候选标签中做受约束决策。”
