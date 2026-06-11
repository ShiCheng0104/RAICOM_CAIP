# 性能测试报告

## 测试环境

```text
日期：2026-06-08
部署：本机 Docker Compose
服务：Model API + LightGBM
接口：http://localhost:8000/predict
模型：lightgbm 20260527T040026Z
特征数：67
```

## API 延迟测试

命令：

```powershell
python scripts/benchmark_api_latency.py --api-url http://localhost:8000 --records 100 --batch-size 25 --output models/validation/api_latency_v7.json
```

结果：

| 指标 | 数值 |
|---|---:|
| 单笔请求平均延迟 | 115.78 ms |
| 单笔请求 P50 | 107.81 ms |
| 单笔请求 P95 | 156.55 ms |
| 单笔请求 P99 | 240.30 ms |
| 批量请求平均批延迟 | 125.00 ms / 25 条 |
| 批量请求平均单条折算 | 5.00 ms |
| 批量单条 P95 折算 | 6.16 ms |
| 平均单条吞吐收益 | 23.16x |

## 结论

```text
1. 单笔 API 已满足演示环境低延迟在线评分。
2. 批量 /predict 对 Flink 微批场景收益明显，平均单条延迟从 115.78ms 降到 5.00ms。
3. 当前数据证明微批量推理是高并发场景下的关键优化方向。
```

## 完整流式链路验证

```text
Flink 作业状态：RUNNING
交易回放：100 条成功
risk_results：成功输出窗口特征、用户/设备/IP/商户画像、图风险特征、risk_score、decision 和 reason_codes
尾批刷新：producer 自动读取 topic 实际分区数后发送 flush marker，兼容历史 1 分区和新建多分区 topic
```
