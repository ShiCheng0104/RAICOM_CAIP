param(
  [string]$Dataset = "fp_fraudsim_injected",
  [int]$ReplayLimit = 1000,
  [int]$ReplayRate = 100,
  [string]$ProjectName = "fraudsim",
  [string]$PythonExe = ".\.conda\ruikang\python.exe",
  [string]$ApiKey = "",
  [switch]$SkipDependencyInstall,
  [switch]$SkipTrain,
  [switch]$SkipScorecard,
  [switch]$SkipGraphSage,
  [switch]$SkipGraphMining,
  [switch]$SkipReplay
)

$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -Scope Global -ErrorAction SilentlyContinue) {
  $global:PSNativeCommandUseErrorActionPreference = $false
}

function Write-Step {
  param([string]$Message)
  Write-Host ""
  Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-Checked {
  param(
    [Parameter(Mandatory = $true)]
    [string]$FilePath,
    [Parameter(Mandatory = $true)]
    [string[]]$Arguments
  )

  & $FilePath @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
  }
}

function Wait-Http {
  param(
    [string]$Url,
    [int]$TimeoutSeconds = 120
  )

  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    try {
      Invoke-RestMethod -Uri $Url -TimeoutSec 5 | Out-Null
      return
    } catch {
      Start-Sleep -Seconds 3
    }
  }

  throw "Timed out waiting for $Url"
}

function Wait-FlinkJob {
  param([int]$TimeoutSeconds = 180)

  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    try {
      $jobs = Invoke-RestMethod -Uri "http://localhost:8081/jobs" -TimeoutSec 5
      if ($jobs.jobs | Where-Object { $_.status -eq "RUNNING" }) {
        return
      }
    } catch {
      # Flink may still be starting.
    }
    Start-Sleep -Seconds 3
  }

  docker compose -p $ProjectName logs --tail 120 flink-risk-job flink-jobmanager flink-taskmanager
  throw "Timed out waiting for a RUNNING Flink job."
}

function Wait-RiskOutput {
  param([int]$TimeoutSeconds = 120)

  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    try {
      $result = Invoke-RestMethod -Uri "http://localhost:8000/topics/risk_results/recent?limit=1&timeout_ms=1000" -TimeoutSec 10
      if ($result.messages.Count -gt 0) {
        return
      }
    } catch {
      # The topic may not contain output yet.
    }
    Start-Sleep -Seconds 3
  }

  throw "Timed out waiting for risk_results output."
}

function Assert-DashboardCapabilities {
  $dashboard = Invoke-WebRequest -Uri "http://localhost:8000/dashboard" -UseBasicParsing -TimeoutSec 15
  if ($dashboard.StatusCode -ne 200 -or $dashboard.Content -notmatch "阈值沙盒") {
    throw "Dashboard assets are unavailable or stale."
  }

  if (-not $SkipScorecard) {
    $threshold = Invoke-RestMethod -Uri "http://localhost:8000/threshold-sandbox" -TimeoutSec 15
    if (-not $threshold.available) {
      throw "Threshold sandbox scorecard is unavailable."
    }
  }

  if (-not $SkipGraphSage) {
    $graphSage = Invoke-RestMethod -Uri "http://localhost:8000/graphsage/metrics" -TimeoutSec 15
    if (-not $graphSage.available) {
      throw "GraphSAGE metrics are unavailable."
    }
  }

  if (-not $SkipGraphMining) {
    $graph = Invoke-RestMethod -Uri "http://localhost:8000/graph/mining/groups?limit=1" -TimeoutSec 30
    if ($graph.groups.Count -lt 1) {
      throw "Graph mining groups are unavailable."
    }
  }

  Invoke-RestMethod -Uri "http://localhost:8000/models" -TimeoutSec 15 | Out-Null
  Invoke-RestMethod -Uri "http://localhost:8000/metrics" -TimeoutSec 15 | Out-Null
  Invoke-RestMethod -Uri "http://localhost:8000/leaderboard" -TimeoutSec 15 | Out-Null
  Invoke-RestMethod -Uri "http://localhost:8000/demo/actions" -TimeoutSec 15 | Out-Null
}

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$env:DOCKER_CONFIG = Join-Path (Get-Location) ".docker-empty"
New-Item -ItemType Directory -Force ".docker-empty" | Out-Null

if (-not (Test-Path $PythonExe)) {
  throw "Python environment not found: $PythonExe. Create it with: conda create -p .\.conda\ruikang python=3.10 -y"
}

Write-Step "Checking local Python dependencies"
$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
  & $PythonExe -c "import pandas, pyarrow, sklearn, lightgbm, fastapi, confluent_kafka, redis, requests, yaml, joblib" *> $null
  $dependencyExitCode = $LASTEXITCODE
} finally {
  $ErrorActionPreference = $previousErrorActionPreference
}
if ($dependencyExitCode -ne 0) {
  if ($SkipDependencyInstall) {
    throw "Required Python packages are missing and SkipDependencyInstall is set."
  }
  Write-Host "[start-all] Required packages are missing, installing requirements-fraudsim.txt..." -ForegroundColor Yellow
  Invoke-Checked -FilePath $PythonExe -Arguments @("-m", "pip", "install", "--upgrade", "pip")
  Invoke-Checked -FilePath $PythonExe -Arguments @("-m", "pip", "install", "-r", "requirements-fraudsim.txt")
}

Write-Step "Checking Docker daemon"
$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
  docker info *> $null
  $dockerExitCode = $LASTEXITCODE
} finally {
  $ErrorActionPreference = $previousErrorActionPreference
}
if ($dockerExitCode -ne 0) {
  throw "Docker is not reachable. Start Docker Desktop first, then rerun this script."
}

Write-Step "Checking dataset"
$DatasetRoot = Join-Path "data\processed" $Dataset
if (-not (Test-Path $DatasetRoot)) {
  throw "Dataset not found: $DatasetRoot"
}

Write-Step "Checking model artifact"
$ModelPath = "models\lightgbm\latest\model.pkl"
if ((-not $SkipTrain) -and (-not (Test-Path $ModelPath))) {
  Write-Host "[start-all] LightGBM artifact missing, training now..."
  Invoke-Checked -FilePath $PythonExe -Arguments @("-m", "fraudsim.training.train", "--dataset", $Dataset, "--model", "lightgbm")
} elseif (Test-Path $ModelPath) {
  Write-Host "[start-all] Found $ModelPath"
} else {
  Write-Warning "[start-all] SkipTrain is set and model artifact is missing. model-api may fail to load a model."
}

if ((-not $SkipScorecard) -and (Test-Path $ModelPath)) {
  $ScorecardPath = "models\lightgbm\latest\evaluation_scores.parquet"
  if (-not (Test-Path $ScorecardPath)) {
    Write-Step "Building threshold sandbox scorecard"
    Invoke-Checked -FilePath $PythonExe -Arguments @("-m", "fraudsim.training.build_scorecard", "--dataset", $Dataset, "--model", "lightgbm")
  } else {
    Write-Host "[start-all] Found $ScorecardPath"
  }
} elseif ($SkipScorecard) {
  Write-Warning "[start-all] SkipScorecard is set. The Dashboard threshold sandbox may be unavailable."
}

if (-not $SkipGraphSage) {
  $GraphSageMetrics = "models\graphsage_sidecar\latest\metrics.json"
  if (-not (Test-Path $GraphSageMetrics)) {
    Write-Step "Building GraphSAGE sidecar metrics for the Dashboard"
    $GraphImage = "$ProjectName-graph-training:latest"
    Invoke-Checked -FilePath "docker" -Arguments @("build", "-t", $GraphImage, "-f", "Dockerfile.graph", ".")
    $DataMount = (Resolve-Path "data\processed").Path
    $ModelsMount = (Resolve-Path "models").Path
    Invoke-Checked -FilePath "docker" -Arguments @(
      "run", "--rm",
      "--mount", "type=bind,source=$DataMount,target=/app/data/processed,readonly",
      "--mount", "type=bind,source=$ModelsMount,target=/app/models",
      $GraphImage, "python", "-m", "fraudsim.graphsage_experiment", "--dataset", $Dataset
    )
  } else {
    Write-Host "[start-all] Found $GraphSageMetrics"
  }
} else {
  Write-Warning "[start-all] SkipGraphSage is set. The Dashboard GraphSAGE panel may be unavailable."
}

if (-not $SkipGraphMining) {
  Write-Step "Preparing graph mining artifacts"
  Invoke-Checked -FilePath $PythonExe -Arguments @("-m", "fraudsim.graph_mining", "--dataset", $Dataset)
} else {
  Write-Warning "[start-all] SkipGraphMining is set. The Dashboard graph mining page can still generate artifacts on demand."
}

Write-Step "Building and starting Docker services"
$env:FRAUDSIM_DATASET = $Dataset
$env:FRAUDSIM_API_KEY = $ApiKey

Write-Step "Building and starting Kafka, Redis, and Model API"
Invoke-Checked -FilePath "docker" -Arguments @("compose", "-p", $ProjectName, "up", "-d", "--build", "--force-recreate", "kafka", "redis", "model-api")

Write-Step "Waiting for Model API"
Wait-Http -Url "http://localhost:8000/health" -TimeoutSeconds 180

Write-Step "Creating Kafka topics"
Invoke-Checked -FilePath $PythonExe -Arguments @("-m", "fraudsim.streaming.topics", "--bootstrap-servers", "localhost:9094")

Write-Step "Loading Redis profiles"
Invoke-Checked -FilePath $PythonExe -Arguments @("-m", "fraudsim.streaming.load_profiles", "--dataset", $Dataset, "--redis-url", "redis://localhost:6379/0")

Write-Step "Building and starting Flink streaming job"
Invoke-Checked -FilePath "docker" -Arguments @("compose", "-p", $ProjectName, "up", "-d", "--build", "--force-recreate", "flink-jobmanager", "flink-taskmanager", "flink-risk-job")

Write-Step "Waiting for Flink job"
Wait-FlinkJob -TimeoutSeconds 240

if (-not $SkipReplay) {
  Write-Step "Replaying transaction stream"
  Invoke-Checked -FilePath $PythonExe -Arguments @(
    "-m", "fraudsim.streaming.producer",
    "--dataset", $Dataset,
    "--bootstrap-servers", "localhost:9094",
    "--rate", [string]$ReplayRate,
    "--limit", [string]$ReplayLimit
  )
  Write-Step "Waiting for real-time risk output"
  Wait-RiskOutput -TimeoutSeconds 180
}

Write-Step "Checking all Dashboard capabilities"
Assert-DashboardCapabilities

Write-Step "Validation summary"
& ".\scripts\demo_check.ps1" -ProjectName $ProjectName -ApiKey $ApiKey

Write-Host ""
Write-Host "Ready." -ForegroundColor Green
Write-Host "Model API : http://localhost:8000/health"
Write-Host "Dashboard : http://localhost:8000/dashboard"
Write-Host "Flink UI  : http://localhost:8081"
Write-Host ""
Write-Host "Sample risk output:"
Write-Host "docker exec fraudsim-kafka /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server kafka:9092 --topic risk_results --from-beginning --max-messages 5"
