# 下载赛题一所需的全部公开数据集
# 使用方法: 在 PowerShell 中直接运行本脚本
$ErrorActionPreference = "Continue"
$env:KAGGLE_API_TOKEN = "KGAT_c9b704b249e4faa15b45a5465a718b24"

$KAGGLE = "D:\Anaconda3\Scripts\kaggle.exe"
$ROOT = Split-Path -Parent $PSScriptRoot
$DATA = Join-Path $ROOT "data"

function Ensure-Dir($p) { if (-not (Test-Path $p)) { New-Item -ItemType Directory -Path $p -Force | Out-Null } }

Ensure-Dir $DATA

function Download-Dataset($slug, $folder) {
    $target = Join-Path $DATA $folder
    Ensure-Dir $target
    Write-Host "==== Downloading dataset: $slug -> $target ====" -ForegroundColor Cyan
    & $KAGGLE datasets download -d $slug -p $target --unzip
    if ($LASTEXITCODE -ne 0) { Write-Host "[WARN] Failed: $slug" -ForegroundColor Yellow }
}

function Download-Competition($slug, $folder) {
    $target = Join-Path $DATA $folder
    Ensure-Dir $target
    Write-Host "==== Downloading competition: $slug -> $target ====" -ForegroundColor Cyan
    & $KAGGLE competitions download -c $slug -p $target
    if ($LASTEXITCODE -ne 0) { Write-Host "[WARN] Failed: $slug (need accept rules)" -ForegroundColor Yellow; return }
    Get-ChildItem $target -Filter *.zip | ForEach-Object {
        Write-Host "Unzipping $($_.Name)"
        Expand-Archive -Path $_.FullName -DestinationPath $target -Force
        Remove-Item $_.FullName
    }
}

# 1. PaySim - 移动支付仿真
Download-Dataset "ealaxi/paysim1" "paysim"

# 2. BankSim - 银行支付仿真
Download-Dataset "ealaxi/banksim1" "banksim"

# 3. IEEE-CIS Fraud Detection (竞赛数据)
Download-Competition "ieee-fraud-detection" "ieee_cis"

# 4. Credit Card Fraud Detection
Download-Dataset "mlg-ulb/creditcardfraud" "creditcard"

# 5. Elliptic Bitcoin Dataset
Download-Dataset "ellipticco/elliptic-data-set" "elliptic"

# 6. DGraph-Fin
Download-Dataset "gahoiambuj/dgraphfin" "dgraphfin"

# 7. SAML-D 反洗钱合成监测数据
Download-Dataset "berkanoztas/synthetic-transaction-monitoring-dataset-aml" "saml_d"

# 8. AMLSim (GitHub) - 仅克隆代码与示例配置
$amlsimDir = Join-Path $DATA "amlsim"
if (-not (Test-Path (Join-Path $amlsimDir ".git"))) {
    Write-Host "==== Cloning AMLSim from GitHub ====" -ForegroundColor Cyan
    git clone --depth 1 https://github.com/IBM/AMLSim.git $amlsimDir
} else {
    Write-Host "AMLSim already cloned." -ForegroundColor Green
}

Write-Host "`n=== ALL DOWNLOADS FINISHED ===" -ForegroundColor Green
Get-ChildItem $DATA | Format-Table Name, Mode, LastWriteTime
