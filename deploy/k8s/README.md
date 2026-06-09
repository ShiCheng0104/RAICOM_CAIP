# Kubernetes 混合云演示清单

`edge.yaml` 模拟边缘/私有云实时侧，部署 Kafka、Redis、双副本 Model API 和 Flink。

`center.yaml` 模拟中心云训练侧，使用 CronJob 运行反馈候选模型训练和 GraphSAGE 旁路实验。

部署前需要：

1. 使用 `Dockerfile`、`Dockerfile.flink`、`Dockerfile.graph` 构建并推送 `fraudsim/model-api:latest`、`fraudsim/flink:latest`、`fraudsim/graph-training:latest`，或替换为实际镜像仓库地址。
2. 将数据、模型和反馈池挂载到对象存储 CSI、NFS 或 PVC。
3. 替换 `fraudsim-api-secret`，生产环境使用 Secret Manager。
4. 跨云 Kafka 同步使用 MirrorMaker 2，并启用 TLS/SASL 或专线/VPN。

静态检查：

```powershell
python -m unittest tests.test_k8s_manifests
docker run --rm -v "${PWD}:/work" ghcr.io/yannh/kubeconform:latest -summary -strict /work/deploy/k8s/edge.yaml /work/deploy/k8s/center.yaml
```

启用 Docker Desktop Kubernetes 后可部署：

```powershell
kubectl apply -f deploy/k8s/edge.yaml
kubectl apply -f deploy/k8s/center.yaml
kubectl get pods -n fraudsim-edge
kubectl get cronjobs -n fraudsim-center
```
