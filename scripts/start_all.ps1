param(
  [string]$Dataset = "fp_fraudsim_injected",
  [int]$ReplayLimit = 1000,
  [int]$ReplayRate = 100,
  [string]$ProjectName = "fraudsim",
  [string]$PythonExe = ".\.conda\ruikang\python.exe",
  [switch]$SkipTrain,
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
    [Parameter(ValueFromRemainingArguments = $true)]
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

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$env:DOCKER_CONFIG = Join-Path (Get-Location) ".docker-empty"
New-Item -ItemType Directory -Force ".docker-empty" | Out-Null

if (-not (Test-Path $PythonExe)) {
  throw "Python environment not found: $PythonExe. Create it with: conda create -p .\.conda\ruikang python=3.10 -y"
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
  Invoke-Checked $PythonExe -m fraudsim.training.train --dataset $Dataset --model lightgbm
} elseif (Test-Path $ModelPath) {
  Write-Host "[start-all] Found $ModelPath"
} else {
  Write-Warning "[start-all] SkipTrain is set and model artifact is missing. model-api may fail to load a model."
}

Write-Step "Building and starting Docker services"
Invoke-Checked docker compose -p $ProjectName up -d --build kafka redis model-api flink-jobmanager flink-taskmanager flink-risk-job

Write-Step "Waiting for Model API"
Wait-Http -Url "http://localhost:8000/health" -TimeoutSeconds 180

Write-Step "Creating Kafka topics"
Invoke-Checked $PythonExe -m fraudsim.streaming.topics --bootstrap-servers localhost:9094

Write-Step "Loading Redis profiles"
Invoke-Checked $PythonExe -m fraudsim.streaming.load_profiles --dataset $Dataset --redis-url redis://localhost:6379/0

if (-not $SkipReplay) {
  Write-Step "Replaying transaction stream"
  Invoke-Checked $PythonExe -m fraudsim.streaming.producer --dataset $Dataset --bootstrap-servers localhost:9094 --rate $ReplayRate --limit $ReplayLimit
}

Write-Step "Validation summary"
& ".\scripts\demo_check.ps1" -ProjectName $ProjectName

Write-Host ""
Write-Host "Ready." -ForegroundColor Green
Write-Host "Model API : http://localhost:8000/health"
Write-Host "Dashboard : http://localhost:8000/dashboard"
Write-Host "Flink UI  : http://localhost:8081"
Write-Host ""
Write-Host "Sample risk output:"
Write-Host "docker exec fraudsim-kafka /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server kafka:9092 --topic risk_results --from-beginning --max-messages 5"
