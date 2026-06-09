# 安全与合规设计

## 当前实现

| 能力 | 说明 |
|---|---|
| 仿真数据 | 默认使用公开/仿真数据，不包含真实个人金融数据 |
| API Key | 设置 `FRAUDSIM_API_KEY` 后，`/predict`、`/feedback`、`/reload`、`/demo/run`、模型发布/回滚等敏感接口需要 `X-API-Key` |
| 集中审计 | 敏感操作同时写入本地 `audit.jsonl` 和 Kafka `audit_events`，本地日志在 Kafka 不可用时仍保底 |
| 审计日志 | 敏感操作写入 `models/audit/audit.jsonl`，可通过 `GET /audit/recent` 查看 |
| 模型发布备份 | `POST /models/promote` 会先将旧 `latest` 备份到 `rollback` |
| 前端脱敏展示 | Dashboard 对长 ID 使用标签化和横向滚动，生产环境可进一步只展示哈希前后缀 |

## 生产部署建议

```text
1. Model API 仅暴露给内网网关，外部访问走 HTTPS / WAF
2. Kafka 和 Redis 不暴露公网端口，跨云同步使用 VPN / 专线 / TLS
3. API Key 替换为 OAuth2 / mTLS / IAM 服务账号
4. 对 payer_id、device_id、ip_id 做哈希化或令牌化
5. 训练数据、反馈数据和模型产物使用对象存储加密
6. 审计日志进入集中日志平台并设置不可篡改存储策略
```

## 演示验证

启用 API Key：

```powershell
$env:FRAUDSIM_API_KEY="demo-secret"
docker compose -p fraudsim up -d --build model-api
```

调用敏感接口：

```powershell
Invoke-RestMethod http://localhost:8000/predict `
  -Method Post `
  -Headers @{"X-API-Key"="demo-secret"} `
  -ContentType "application/json" `
  -Body '{"record":{"transaction_id":"t1","amount":100}}'
```

查看审计：

```powershell
Invoke-RestMethod http://localhost:8000/audit/recent -Headers @{"X-API-Key"="demo-secret"}
```

查看集中审计 topic：

```powershell
docker exec fraudsim-kafka /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server kafka:9092 --topic audit_events --from-beginning --max-messages 10
```
