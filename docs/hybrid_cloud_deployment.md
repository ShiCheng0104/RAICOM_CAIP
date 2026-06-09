# 混合云部署与高并发低延迟验证方案

## 部署分层

```text
边缘/业务侧
  - Kafka Producer
  - Flink 实时窗口特征
  - Model API 在线推理
  - Redis 热画像

中心云/训练侧
  - 离线训练
  - 多模型评估
  - 图挖掘与团伙分析
  - 模型仓库与审计日志

同步通道
  - Kafka MirrorMaker / VPN / 专线
  - 模型产物通过对象存储或制品仓库下发
```

## 单机最低验证

在没有多设备或云服务器时，使用 Docker Compose 模拟混合云边界：

| 容器 | 对应角色 |
|---|---|
| `fraudsim-kafka` | 消息总线 |
| `fraudsim-flink-jobmanager/taskmanager` | 实时计算集群 |
| `fraudsim-model-api` | 在线推理服务 |
| `fraudsim-redis` | 热画像存储 |
| 本地 Python 脚本 | 离线训练、图挖掘、压测客户端 |

## 验证指标

```text
1. API P50 / P95 / P99 延迟
2. Kafka 输入速率
3. Flink risk_results 输出速率
4. 端到端处理耗时
5. 错误率
6. 高风险告警输出数量
```

## 命令

API 延迟：

```powershell
python scripts/benchmark_api_latency.py --url http://localhost:8000 --requests 500 --concurrency 20
```

流式链路：

```powershell
python -m fraudsim.streaming.benchmark --dataset fp_fraudsim_injected --bootstrap-servers localhost:9094 --limit 500 --timeout 180
```

微批推理：

```powershell
python -m fraudsim.streaming.batch_risk_worker --bootstrap-servers localhost:9094 --api-url http://localhost:8000 --redis-url redis://localhost:6379/0 --risk-topic risk_results_batch --batch-size 100 --linger-ms 300 --limit 500
```

## 答辩表述

当前环境用单机 Docker 模拟混合云最小部署单元；生产环境中，Kafka/Flink/Redis/Model API 可迁移到边缘或私有云，离线训练和图挖掘放在中心云，通过加密通道同步模型与反馈样本。

