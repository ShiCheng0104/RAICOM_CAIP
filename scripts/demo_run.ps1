param(
  [string]$Dataset = "fp_fraudsim_injected",
  [int]$ReplayLimit = 1000,
  [int]$ReplayRate = 100,
  [string]$ProjectName = "fraudsim"
)

$ErrorActionPreference = "Stop"
$env:DOCKER_CONFIG = Join-Path (Get-Location) ".docker-empty"
New-Item -ItemType Directory -Force ".docker-empty" | Out-Null

Write-Host "[demo] starting kafka/redis/model-api/flink"
docker compose -p $ProjectName up -d --build kafka redis model-api flink-jobmanager flink-taskmanager flink-risk-job

Write-Host "[demo] creating topics"
.conda\ruikang\python.exe -m fraudsim.streaming.topics --bootstrap-servers localhost:9094

Write-Host "[demo] loading profiles"
.conda\ruikang\python.exe -m fraudsim.streaming.load_profiles --dataset $Dataset --redis-url redis://localhost:6379/0

Write-Host "[demo] replaying transactions"
.conda\ruikang\python.exe -m fraudsim.streaming.producer --dataset $Dataset --bootstrap-servers localhost:9094 --rate $ReplayRate --limit $ReplayLimit

Write-Host "[demo] quick status"
.\scripts\demo_check.ps1 -ProjectName $ProjectName

Write-Host "[demo] sample risk output command:"
Write-Host "docker exec fraudsim-kafka /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server kafka:9092 --topic risk_results --from-beginning --max-messages 5"
