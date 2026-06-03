"""
赛题一·路线三 数据集批量下载脚本
通过 kaggle Python API 串行下载所有公开数据集到 data/ 子目录
进度日志写到 data/_download.log
"""
import os
import sys
import time
import shutil
import zipfile
import subprocess
from pathlib import Path

os.environ["KAGGLE_API_TOKEN"] = "KGAT_c9b704b249e4faa15b45a5465a718b24"

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)
LOG = DATA / "_download.log"


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def is_complete(folder: Path, marker_files=None) -> bool:
    if not folder.exists():
        return False
    if marker_files:
        return all((folder / m).exists() for m in marker_files)
    # 至少有一个数据文件且没残留 zip
    has_data = any(p.suffix in {".csv", ".parquet",
                   ".npz", ".json"} for p in folder.rglob("*"))
    has_zip = any(folder.glob("*.zip"))
    return has_data and not has_zip


def kaggle_dataset(slug: str, folder_name: str, marker=None):
    """从 kaggle 官方 datasets 下载并解压"""
    target = DATA / folder_name
    target.mkdir(parents=True, exist_ok=True)
    if is_complete(target, marker):
        log(f"[skip] {slug} 已下载: {target}")
        return True
    log(f"[start] {slug} -> {target}")
    # 清理残留
    for z in target.glob("*.zip"):
        z.unlink()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    try:
        api.dataset_download_files(
            slug, path=str(target), unzip=True, quiet=False)
        log(f"[done] {slug}")
        return True
    except Exception as e:
        log(f"[fail] {slug}: {e}")
        return False


def kaggle_competition(slug: str, folder_name: str, marker=None):
    """从 kaggle competitions 下载并解压"""
    target = DATA / folder_name
    target.mkdir(parents=True, exist_ok=True)
    if is_complete(target, marker):
        log(f"[skip] {slug} 已下载: {target}")
        return True
    log(f"[start] competition {slug} -> {target}")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    try:
        api.competition_download_files(slug, path=str(target), quiet=False)
    except Exception as e:
        log(f"[fail] competition {slug}: {e}")
        log("       提示: 可能需要先在 Kaggle 网站接受比赛规则")
        return False
    # 解压所有 zip
    for z in target.glob("*.zip"):
        log(f"[unzip] {z.name}")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(target)
        z.unlink()
    log(f"[done] competition {slug}")
    return True


def clone_amlsim():
    target = DATA / "amlsim"
    if (target / ".git").exists() or (target / "README.md").exists():
        log("[skip] AMLSim 已克隆")
        return True
    log("[start] git clone IBM/AMLSim")
    target.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1",
                "https://github.com/IBM/AMLSim.git", str(target)],
            check=True,
        )
        log("[done] AMLSim cloned")
        return True
    except Exception as e:
        log(f"[fail] AMLSim: {e}")
        return False


TASKS = [
    ("dataset", "ealaxi/paysim1", "paysim", None),
    ("dataset", "ealaxi/banksim1", "banksim", None),
    ("competition", "ieee-fraud-detection",
     "ieee_cis", ["train_transaction.csv"]),
    ("dataset", "mlg-ulb/creditcardfraud", "creditcard", ["creditcard.csv"]),
    ("dataset", "ellipticco/elliptic-data-set", "elliptic", None),
    ("dataset", "gahoiambuj/dgraphfin", "dgraphfin", None),
    ("dataset", "berkanoztas/synthetic-transaction-monitoring-dataset-aml", "saml_d", None),
]


def main():
    log("=" * 60)
    log("开始下载赛题一全部数据集")
    log("=" * 60)
    failed = []
    for kind, slug, folder, marker in TASKS:
        ok = (
            kaggle_competition(slug, folder, marker)
            if kind == "competition"
            else kaggle_dataset(slug, folder, marker)
        )
        if not ok:
            failed.append(slug)
    if not clone_amlsim():
        failed.append("AMLSim")
    log("=" * 60)
    if failed:
        log(f"以下数据集失败: {failed}")
        sys.exit(1)
    log("全部数据集下载完成 ✓")


if __name__ == "__main__":
    main()
