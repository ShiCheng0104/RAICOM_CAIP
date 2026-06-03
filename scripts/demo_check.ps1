param(
  [string]$ProjectName = "fraudsim",
  [string]$KafkaBootstrap = "localhost:9094"
)

$ErrorActionPreference = "Stop"
$env:DOCKER_CONFIG = Join-Path (Get-Location) ".docker-empty"

Write-Host "[demo-check] docker compose status"
docker compose -p $ProjectName ps

Write-Host "[demo-check] model api health"
try {
  Invoke-RestMethod -Uri "http://localhost:8000/health" | ConvertTo-Json -Depth 5
} catch {
  Write-Warning "model api is not healthy: $($_.Exception.Message)"
}

Write-Host "[demo-check] flink jobs"
try {
  Invoke-RestMethod -Uri "http://localhost:8081/jobs" | ConvertTo-Json -Depth 5
} catch {
  Write-Warning "flink ui is not reachable: $($_.Exception.Message)"
}

Write-Host "[demo-check] kafka topics"
docker exec fraudsim-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server kafka:9092 --list

Write-Host "[demo-check] model artifacts"
Get-ChildItem models/lightgbm/latest | Select-Object Name,Length,LastWriteTime
