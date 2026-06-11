# FP-FraudSim 流式风控系统

本项目实现了一个面向金融欺诈交易识别的最小可落地系统：离线训练多模型，在线通过 FastAPI 推理，使用 Kafka + Flink 做交易流处理，并提供 HTML Dashboard 观察模型、指标、实时结果和人工反馈。

## 当前完成状态

项目当前已经形成可演示、可训练、可验证的完整闭环：

```text
多源异构数据融合
  -> 多维特征与图画像
  -> LightGBM / 多模型离线训练
  -> Kafka + Flink 实时窗口与微批推理
  -> 风险评分、等级、理由码与团伙证据
  -> Dashboard 人工审核
  -> candidate / latest / rollback 模型迭代
```

当前已完成：

```text
1. 可调参交易仿真、跨渠道交易回放和实时检测
2. LightGBM 主模型、多模型对比和模型热插拔
3. Kafka + PyFlink + Redis + FastAPI 流式链路
4. Flink 微批推理、尾批刷新和性能验证
5. 可解释图分析团伙挖掘与 GraphSAGE 旁路实验
6. 阈值沙盒、人工反馈、候选模型发布与回滚
7. API Key、本地审计与 Kafka 集中审计
8. Docker Compose 一键启动与 Kubernetes 混合云部署清单
```

Kubernetes 当前状态：

```text
暂不实施真实 K8s 集群部署。
已完成边缘侧/中心侧部署清单、PVC、Secret、NetworkPolicy、Flink 风险作业和训练 CronJob 设计。
20 个 K8s 资源已通过 kubeconform 严格 schema 校验；真实集群部署、跨云网络和故障恢复压测留作后续工作。
```

最终答辩材料见 `答辩方案最终版.md`。

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
requirements-graph.txt    GraphSAGE 旁路实验可选依赖
Dockerfile.graph          中心云图训练镜像
docs/                     安全、混合云、模型迭代、特征解释文档
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
5. 生成可调参仿真交易流
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

可选：先运行图分析方法挖掘疑似欺诈团伙，并把团伙风险并入图画像：

```powershell
.\.conda\ruikang\python.exe -m fraudsim.graph_mining --dataset fp_fraudsim_injected --force
.\.conda\ruikang\python.exe -c "from pathlib import Path; from fraudsim.graph_features import build_graph_entity_features; build_graph_entity_features(Path('data/processed/fp_fraudsim_injected'), force=True)"
```

图挖掘会生成：

```text
data/processed/fp_fraudsim_injected/graph_mining/fraud_groups.parquet
data/processed/fp_fraudsim_injected/graph_mining/entity_graph_risk.parquet
data/processed/fp_fraudsim_injected/graph_mining/group_evidence.jsonl
data/processed/fp_fraudsim_injected/graph_mining/graph_mining_summary.json
```

回放交易流：

```powershell
.\.conda\ruikang\python.exe -m fraudsim.streaming.producer --dataset fp_fraudsim_injected --bootstrap-servers localhost:9094 --rate 100 --limit 1000
```

可选：生成一段新的可调参仿真交易流：

```powershell
.\.conda\ruikang\python.exe -m fraudsim.simulator.generate `
  --dataset fp_fraudsim_injected `
  --rows 10000 `
  --fraud-ratio 0.04 `
  --fraud-group-count 20 `
  --fraud-group-size 12
```

输出位于：

```text
data/processed/fp_fraudsim_injected/simulations/<timestamp>/
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
2. 点击 leaderboard 切换查看不同模型 metrics
3. 查看 risk_results / alert_events / risk_results_batch
4. 热切换模型
5. 表单化单笔试算，并展示理由码解释
6. 提交人工反馈
7. 查看图分析挖掘出的疑似团伙、共享设备/IP/收款方证据和召回指标
8. 拖拽、缩放团伙证据图，辅助人工研判和溯源分析
9. 通过人工反馈池训练 candidate 模型，并支持发布与回滚
10. 在指标页使用阈值沙盒评估拦截量、误报率、召回和审核工作量
11. 查看 GraphSAGE 旁路实验指标
```

总览页和实时页中的演示按钮支持任务状态展示。点击“挖掘欺诈团伙”后，页面会显示：

```text
运行中 / 成功 / 失败
任务 ID
耗时
退出码
日志尾部
```

图挖掘成功后会自动刷新“图挖掘”页。该按钮需要 `model-api` 容器对 `data/processed` 有写权限；当前 `docker-compose.yml` 已按演示需要配置为可写挂载。

如果使用一键启动脚本，脚本会在容器启动前预生成图挖掘产物，因此图挖掘页通常首次打开即可看到 Top 风险团伙；也可以在 Dashboard 内重新点击“挖掘欺诈团伙”刷新。

## 8. 一键启动

推荐直接使用：

```powershell
.\scripts\start_all.ps1
```

常用参数：

```powershell
.\scripts\start_all.ps1 -Dataset fp_fraudsim_injected -ReplayLimit 1000 -ReplayRate 100
.\scripts\start_all.ps1 -ApiKey demo-secret
```

可选跳过项：

```powershell
.\scripts\start_all.ps1 -SkipTrain
.\scripts\start_all.ps1 -SkipScorecard
.\scripts\start_all.ps1 -SkipGraphMining
.\scripts\start_all.ps1 -SkipReplay
```

脚本会自动执行：

```text
1. 检查本地 Python 环境
2. 检查或训练 LightGBM 模型
3. 补建阈值沙盒 scorecard，保证指标页可直接调阈值
4. 预生成图挖掘产物，保证图挖掘页可直接展示
5. 将 -Dataset 写入容器环境，保证 API、演示按钮、Flink 与脚本使用同一数据集
6. 构建并启动 Docker 容器
7. 创建 Kafka topics
8. 初始化 Redis 画像
9. 回放交易流
10. 输出 API / Dashboard / Flink UI 地址
11. 打印基础验证结果
```

默认情况下，一键启动后 Dashboard 的总览、实时、图挖掘、模型、指标和试算页都可直接使用。若通过 `-SkipGraphMining` 跳过预生成，图挖掘页仍可在首次访问或点击“挖掘欺诈团伙”时按需生成结果。

## 8.1 安全、反馈与模型迭代

启用 API Key 后，敏感接口需要 `X-API-Key`：

```powershell
$env:FRAUDSIM_API_KEY="demo-secret"
docker compose -p fraudsim up -d --build model-api
```

使用 `scripts/start_all.ps1 -ApiKey demo-secret` 也可以一键启用；Dashboard 左侧 API Key 输入框保存后会自动为前端请求附加 `X-API-Key`。

人工反馈会同时写入 Kafka `feedback_events` 和本地反馈池：

```text
data/processed/{dataset}/feedback/feedback_pool.jsonl
```

使用反馈池训练候选模型：

```powershell
.\.conda\ruikang\python.exe -m fraudsim.training.train_with_feedback `
  --dataset fp_fraudsim_injected `
  --model lightgbm `
  --feedback-path data\processed\fp_fraudsim_injected\feedback\feedback_pool.jsonl
```

候选模型、发布与回滚：

```powershell
Invoke-RestMethod "http://localhost:8000/models/lightgbm/candidate"
Invoke-RestMethod "http://localhost:8000/models/promote" -Method Post -ContentType "application/json" -Body '{"model_name":"lightgbm"}'
Invoke-RestMethod "http://localhost:8000/models/rollback" -Method Post -ContentType "application/json" -Body '{"model_name":"lightgbm"}'
Invoke-RestMethod "http://localhost:8000/audit/recent"
```

更多说明：

```text
docs/model_iteration.md
docs/security_design.md
docs/hybrid_cloud_deployment.md
docs/performance_report.md
docs/feature_explanation.md
docs/graphsage_and_threshold_sandbox.md
```

GraphSAGE 旁路实验：

```powershell
.\.conda\ruikang\python.exe -m pip install -r requirements-graph.txt
.\.conda\ruikang\python.exe -m fraudsim.graphsage_experiment `
  --dataset fp_fraudsim_injected `
  --max-edges 80000 `
  --max-nodes 30000 `
  --epochs 20
```

为已有模型生成阈值沙盒 scorecard：

```powershell
.\.conda\ruikang\python.exe -m fraudsim.training.build_scorecard --dataset fp_fraudsim_injected --model lightgbm
```

Kubernetes 混合云演示清单位于 `deploy/k8s/`。项目暂不实施真实 K8s 集群部署；当前可运行结构测试与离线 schema 校验：

```powershell
.\.conda\ruikang\python.exe -m unittest tests.test_k8s_manifests
docker run --rm -v "${PWD}:/work" ghcr.io/yannh/kubeconform:latest -summary -strict /work/deploy/k8s/edge.yaml /work/deploy/k8s/center.yaml
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

查看集中审计日志：

```powershell
docker exec fraudsim-kafka /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server kafka:9092 --topic audit_events --from-beginning --max-messages 10
```

运行单元测试：

```powershell
.\.conda\ruikang\python.exe -m unittest discover -s tests
```

运行轻量流式压测：

```powershell
.\.conda\ruikang\python.exe -m fraudsim.streaming.benchmark --dataset fp_fraudsim_injected --bootstrap-servers localhost:9094 --limit 100 --timeout 120
```

运行图挖掘效果验证：

```powershell
.\.conda\ruikang\python.exe -m fraudsim.graph_mining --dataset fp_fraudsim_injected --force
Get-Content data\processed\fp_fraudsim_injected\graph_mining\graph_mining_summary.json
```

图挖掘 API 验证：

```powershell
Invoke-RestMethod "http://localhost:8000/graph/mining/summary" | ConvertTo-Json -Depth 8
Invoke-RestMethod "http://localhost:8000/graph/mining/groups?limit=3" | ConvertTo-Json -Depth 8
Invoke-RestMethod "http://localhost:8000/graph/mining/groups/GM_000316" | ConvertTo-Json -Depth 10
```

团伙详情接口会返回 `risk_level`、`explanation_codes`、`explanation_text`、`evidence` 和前端可直接绘制的 `subgraph`。

演示任务状态接口：

```powershell
Invoke-RestMethod "http://localhost:8000/demo/runs/<run_id>" | ConvertTo-Json -Depth 8
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
