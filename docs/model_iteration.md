# 模型自适应学习与迭代方案

## 闭环目标

系统支持在交易流持续输入的同时收集人工审核反馈，并以候选模型方式完成重训、评估、发布和回滚。

## 数据流

```text
risk_results
  -> Dashboard 人工审核
  -> POST /feedback
  -> Kafka feedback_events
  -> data/processed/{dataset}/feedback/feedback_pool.jsonl
  -> python -m fraudsim.training.train_with_feedback
  -> models/{model}/candidate
  -> POST /models/promote
  -> models/{model}/latest
```

## 已实现接口

| 接口或命令 | 作用 |
|---|---|
| `POST /feedback` | 写入 Kafka 反馈 topic，并持久化到本地反馈池 |
| `python -m fraudsim.training.train_with_feedback` | 使用反馈池训练 candidate 模型 |
| `GET /models/{model}/candidate` | 查看候选模型指标和 manifest |
| `POST /models/promote` | 将 candidate 发布为 latest，并备份旧 latest 到 rollback |
| `POST /models/rollback` | 将 rollback 恢复为 latest |
| `POST /reload` | 重新加载当前模型产物 |

## 发布策略

候选模型必须满足以下条件才建议发布：

```text
1. PR-AUC 不低于当前模型
2. F1 不低于当前模型
3. false_positive_rate_at_threshold 不显著升高
4. feedback_rows 大于最小反馈样本阈值
5. API smoke test 通过
```

## 答辩表述

本系统不是直接用反馈覆盖线上模型，而是通过 candidate / latest / rollback 三段式模型生命周期降低误发布风险；人工反馈先进入样本池，训练出候选模型，经指标对比后再发布，发布前自动保留回滚版本。

