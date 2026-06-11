param(
  [string]$ProjectName = "fraudsim",
  [string]$KafkaBootstrap = "localhost:9094",
  [string]$ApiKey = ""
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

Write-Host "[demo-check] dashboard capabilities"
try {
  $dashboard = Invoke-WebRequest -Uri "http://localhost:8000/dashboard" -UseBasicParsing
  $threshold = Invoke-RestMethod -Uri "http://localhost:8000/threshold-sandbox"
  $graphSage = Invoke-RestMethod -Uri "http://localhost:8000/graphsage/metrics"
  $groups = Invoke-RestMethod -Uri "http://localhost:8000/graph/mining/groups?limit=1"
  $models = Invoke-RestMethod -Uri "http://localhost:8000/models"
  [pscustomobject]@{
    dashboard_http = $dashboard.StatusCode
    dashboard_current_assets = ($dashboard.Content -match "阈值沙盒")
    threshold_sandbox = $threshold.available
    graphsage_metrics = $graphSage.available
    graph_groups = $groups.groups.Count
    model_count = $models.models.Count
  } | ConvertTo-Json -Depth 5
} catch {
  Write-Warning "dashboard capability check failed: $($_.Exception.Message)"
}

Write-Host "[demo-check] flink jobs"
try {
  Invoke-RestMethod -Uri "http://localhost:8081/jobs" | ConvertTo-Json -Depth 5
} catch {
  Write-Warning "flink ui is not reachable: $($_.Exception.Message)"
}

Write-Host "[demo-check] kafka topics"
docker exec fraudsim-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server kafka:9092 --list

Write-Host "[demo-check] recent risk output"
try {
  Invoke-RestMethod -Uri "http://localhost:8000/topics/risk_results/recent?limit=1&timeout_ms=1000" | ConvertTo-Json -Depth 5
} catch {
  Write-Warning "risk_results is unavailable: $($_.Exception.Message)"
}

if ($ApiKey) {
  Write-Host "[demo-check] protected API access"
  try {
    Invoke-RestMethod -Uri "http://localhost:8000/audit/recent?limit=3" -Headers @{"X-API-Key" = $ApiKey} | ConvertTo-Json -Depth 5
  } catch {
    Write-Warning "protected API check failed: $($_.Exception.Message)"
  }
}

Write-Host "[demo-check] model artifacts"
Get-ChildItem models/lightgbm/latest | Select-Object Name,Length,LastWriteTime
