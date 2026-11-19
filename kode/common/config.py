"""Konfigurasi path tunggal untuk seluruh task.

Semua kode mengambil lokasi data dan output dari sini. Default-nya:
  DATA_DIR = <root repo>/data        (bisa ditimpa --data_dir atau env UTS_DATA_DIR)
  OUT_DIR  = <root repo>/outputs     (bisa ditimpa --out_dir atau env UTS_OUT_DIR)
Dengan begitu kode yang sama jalan di lokal, Colab (/content/data) dan Kaggle
(/kaggle/input/...) tanpa diubah.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("UTS_DATA_DIR", REPO_ROOT / "data"))
OUT_DIR = Path(os.environ.get("UTS_OUT_DIR", REPO_ROOT / "outputs"))

# Nama berkas sesuai data/kamus_data.md
DATA_FILES = {
    "task1_train": "klasifikasi_train.npz",
    "task1_train_meta": "klasifikasi_train_meta.csv",
    "task1_test": "klasifikasi_test.npz",
    "task1_test_meta": "klasifikasi_test_meta.csv",
    "task2_train": "rul_train.csv.gz",
    "task2_test": "rul_test.csv.gz",
    "task3_data": "clustering.npz",
    "task3_meta": "clustering_meta.csv",
}


def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--data_dir", type=str, default=str(DATA_DIR),
                   help="Folder berisi berkas dataset (default: <repo>/data)")
    p.add_argument("--out_dir", type=str, default=str(OUT_DIR),
                   help="Folder output: checkpoint, grafik, CSV (default: <repo>/outputs)")
    p.add_argument("--pred_dir", type=str, default=str(REPO_ROOT),
                   help="Folder tujuan CSV prediksi final (default: root repo)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug", action="store_true",
                   help="Subset kecil data + epoch sedikit, untuk uji cepat di lokal")
    p.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="Lanjutkan dari checkpoint terakhir jika ada (default: ya)")
    return p


def resolve_paths(args, task: str):
    """Ubah string path jadi Path, cek data ada, dan buat folder output task."""
    args.data_dir = Path(args.data_dir)
    args.out_dir = Path(args.out_dir)
    args.pred_dir = Path(args.pred_dir)
    args.task_out = args.out_dir / task
    if args.debug:
        args.task_out = args.task_out / "debug"   # hasil debug tidak menimpa hasil penuh
    args.ckpt_dir = args.task_out / "checkpoints"
    for d in (args.task_out, args.ckpt_dir, args.pred_dir):
        d.mkdir(parents=True, exist_ok=True)
    if not args.data_dir.exists():
        raise FileNotFoundError(
            f"data_dir tidak ditemukan: {args.data_dir}. Lihat README.md bagian 'Data'.")
    return args


def data_path(args, key: str) -> Path:
    p = Path(args.data_dir) / DATA_FILES[key]
    if not p.exists():
        raise FileNotFoundError(f"Berkas {p} tidak ada. Cek --data_dir.")
    return p
