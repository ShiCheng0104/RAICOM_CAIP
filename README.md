# FP-FraudSim 流式风控系统

本项目实现了一个面向金融欺诈交易识别的最小可落地系统：离线训练多模型，在线通过 FastAPI 推理，使用 Kafka + Flink 做交易流处理，并提供 HTML Dashboard 观察模型、指标、实时结果和人工反馈。

## 1. 目录说明

```text
configs/                  配置文件
fraudsim/                 训练、特征、API、流式处理、Dashboard 代码
scripts/                  一键启动、验证脚本
tests/                    单元测试
docker-compose.yml        Kafka / Redis / Flink / Model API 编排
Dockerfile                训练与 Model API 镜像
Dockerfile.flink          PyFlink 作业镜像
requirements-fraudsim.txt 本地训练与 API 依赖
requirements-flink.txt    Flink 镜像内 Python 依赖
```

本仓库不建议提交以下目录：

```text
data/         原始数据和处理后数据，体积较大
models/       训练产物和压测报告，体积较大
RAICOM_CAIP/  赛题材料和本地答辩资料
.conda/       本地虚拟环境
```

## 2. 环境要求

推荐环境：

```text
Windows 10/11
Docker Desktop
Conda 或 Miniconda
Python 3.10
PowerShell 7 或 Windows PowerShell
```

需要 Docker Desktop 已启动，并且本地端口未被占用：

```text
8000  Model API / Dashboard
8081  Flink UI
9094  Kafka external listener
6379  Redis
```

## 3. 安装本地依赖

如果本机已经有 `.conda/ruikang` 环境，可以直接使用。否则新建环境：

```powershell
conda create -p .\.conda\ruikang python=3.10 -y
.\.conda\ruikang\python.exe -m pip install --upgrade pip
.\.conda\ruikang\python.exe -m pip install -r requirements-fraudsim.txt
```

本地 Python 主要用于：

```text
1. 离线训练模型
2. 初始化 Redis 画像
3. 回放 transaction_stream.jsonl 到 Kafka
4. 运行验证脚本和单元测试
```

## 4. 数据和模型准备

默认使用：

```text
data/processed/fp_fraudsim_injected
```

该目录应至少包含：

```text
splits/train.parquet
splits/valid.parquet
splits/test.parquet
transaction_stream.jsonl
用户、商户、设备、IP 画像表
图节点和图边特征表
```

如果本地没有模型产物，先训练 LightGBM：

```powershell
.\.conda\ruikang\python.exe -m fraudsim.training.train --dataset fp_fraudsim_injected --model lightgbm
```

训练后会生成：

```text
models/lightgbm/latest/model.pkl
models/lightgbm/latest/feature_config.json
models/lightgbm/latest/metrics.json
models/leaderboard.json
```

也可以训练其他模型：

```powershell
.\.conda\ruikang\python.exe -m fraudsim.training.train --dataset fp_fraudsim_injected --model xgboost
.\.conda\ruikang\python.exe -m fraudsim.training.train --dataset fp_fraudsim_injected --model catboost
.\.conda\ruikang\python.exe -m fraudsim.training.train --dataset fp_fraudsim_injected --model sklearn_hgb
```

## 5. 构建并启动 Docker 容器

构建并启动基础设施：

```powershell
$env:DOCKER_CONFIG = Join-Path (Get-Location) ".docker-empty"
New-Item -ItemType Directory -Force ".docker-empty" | Out-Null
docker compose -p fraudsim up -d --build kafka redis model-api flink-jobmanager flink-taskmanager flink-risk-job
```

服务地址：

```text
Model API:  http://localhost:8000/health
Dashboard:  http://localhost:8000/dashboard
Flink UI:   http://localhost:8081
Kafka:      localhost:9094
Redis:      localhost:6379
```

查看容器状态：

```powershell
docker compose -p fraudsim ps
```

## 6. 初始化画像和启动流式回放

创建 Kafka topics：

```powershell
.\.conda\ruikang\python.exe -m fraudsim.streaming.topics --bootstrap-servers localhost:9094
```

加载用户、商户、设备、IP、图统计画像到 Redis：

```powershell
.\.conda\ruikang\python.exe -m fraudsim.streaming.load_profiles --dataset fp_fraudsim_injected --redis-url redis://localhost:6379/0
```

回放交易流：

```powershell
.\.conda\ruikang\python.exe -m fraudsim.streaming.producer --dataset fp_fraudsim_injected --bootstrap-servers localhost:9094 --rate 100 --limit 1000
```

Flink 作业会消费 `transaction_events`，输出：

```text
risk_results   完整风险评分结果
alert_events   高风险 reject 告警
late_events    超过 watermark 的迟到事件
```

## 7. 启动前端平台

前端 Dashboard 由 Model API 静态托管，启动 `model-api` 后直接访问：

```text
http://localhost:8000/dashboard
```

Dashboard 支持：

```text
1. 查看 API 健康状态和当前模型
2. 查看 leaderboard 与 metrics
3. 查看 risk_results / alert_events / risk_results_batch
4. 热切换模型
5. 单笔试算
6. 提交人工反馈
```

## 8. 一键启动

推荐直接使用：

```powershell
.\scripts\start_all.ps1
```

常用参数：

```powershell
.\scripts\start_all.ps1 -Dataset fp_fraudsim_injected -ReplayLimit 1000 -ReplayRate 100
```

脚本会自动执行：

```text
1. 检查本地 Python 环境
2. 检查或训练 LightGBM 模型
3. 构建并启动 Docker 容器
4. 创建 Kafka topics
5. 初始化 Redis 画像
6. 回放交易流
7. 输出 API / Dashboard / Flink UI 地址
8. 打印基础验证结果
```

## 9. 验证方式

基础健康检查：

```powershell
.\scripts\demo_check.ps1
```

API 健康检查：

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" | ConvertTo-Json -Depth 5
```

Flink 作业状态：

```powershell
Invoke-RestMethod -Uri "http://localhost:8081/jobs" | ConvertTo-Json -Depth 5
```

查看 Kafka 风险结果：

```powershell
docker exec fraudsim-kafka /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server kafka:9092 --topic risk_results --from-beginning --max-messages 5
```

查看高风险告警：

```powershell
docker exec fraudsim-kafka /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server kafka:9092 --topic alert_events --from-beginning --max-messages 5
```

运行单元测试：

```powershell
.\.conda\ruikang\python.exe -m unittest discover -s tests
```

运行轻量流式压测：

```powershell
.\.conda\ruikang\python.exe -m fraudsim.streaming.benchmark --dataset fp_fraudsim_injected --bootstrap-servers localhost:9094 --limit 100 --timeout 120
```

运行微批量推理验证：

```powershell
.\.conda\ruikang\python.exe -m fraudsim.streaming.producer --dataset fp_fraudsim_injected --bootstrap-servers localhost:9094 --topic transaction_events_batch_demo --rate 0 --limit 500

.\.conda\ruikang\python.exe -m fraudsim.streaming.batch_risk_worker `
  --bootstrap-servers localhost:9094 `
  --api-url http://localhost:8000 `
  --redis-url redis://localhost:6379/0 `
  --input-topic transaction_events_batch_demo `
  --risk-topic risk_results_batch `
  --batch-size 100 `
  --linger-ms 300 `
  --limit 500 `
  --offset-reset earliest
```

## 10. 模型热插拔

新增模型时只需要实现统一 adapter，并在 registry 中注册。训练完成后，模型会以如下目录结构保存：

```text
models/{model_name}/latest/model.pkl
models/{model_name}/latest/feature_config.json
models/{model_name}/latest/metrics.json
```

API 支持：

```text
GET  /models
POST /models/activate
POST /reload
```

Dashboard 的模型页面可以直接查看可用模型并切换当前模型。

## 11. 关闭服务

```powershell
docker compose -p fraudsim down
```

如果要删除容器和匿名卷：

```powershell
docker compose -p fraudsim down -v
```
