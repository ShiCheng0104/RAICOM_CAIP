# GraphSAGE 旁路实验与阈值沙盒

## GraphSAGE 旁路实验

运行：

```powershell
python -m pip install -r requirements-graph.txt
python -m fraudsim.graphsage_experiment --dataset fp_fraudsim_injected --max-edges 80000 --max-nodes 30000 --epochs 20
```

实验从统一图边中受控采样，使用节点类型、度数和邻居聚合训练两层 GraphSAGE，输出：

```text
models/graphsage_sidecar/latest/model.pt
models/graphsage_sidecar/latest/node_embeddings.parquet
models/graphsage_sidecar/latest/metrics.json
models/graphsage_sidecar/latest/history.json
```

本机最终实验：

| 指标 | 数值 |
|---|---:|
| 采样节点 | 30,000 |
| 采样边 | 25,732 |
| 节点欺诈比例 | 21.32% |
| PR-AUC | 0.6799 |
| ROC-AUC | 0.7855 |
| F1@0.5 | 0.5881 |

该实验用于验证图 embedding 对节点风险识别的价值。当前保持旁路，不直接加入线上 LightGBM，避免未经充分时序验证的 embedding 拉低主模型稳定性。

## 阈值沙盒

训练流程现在会保存：

```text
models/{model}/latest/evaluation_scores.parquet
```

已有模型可补建：

```powershell
python -m fraudsim.training.build_scorecard --dataset fp_fraudsim_injected --model lightgbm
```

Dashboard 指标页可拖动人工审核阈值和高风险拦截阈值，实时查看：

```text
放行量、审核量、拦截量、拦截精度、欺诈召回、F1、FPR、误拦量、漏过量
```

接口：

```text
GET  /threshold-sandbox
POST /threshold-sandbox
GET  /graphsage/metrics
```
