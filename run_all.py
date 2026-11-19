"""Satu perintah untuk mereproduksi semua angka di laporan.

  python run_all.py                       # training penuh (GPU disarankan)
  python run_all.py --debug               # subset kecil, uji cepat di lokal
  python run_all.py --tasks 2 3 --data_dir /content/data --out_dir /content/drive/MyDrive/uts_out

Menjalankan task1, task2, task3 berurutan; CSV prediksi ditulis ke root repo
(atau --pred_dir), lalu divalidasi formatnya.
"""
import argparse
import sys
import traceback

from kode.common import add_common_args
from kode.common.utils import env_info, save_json


def main():
    p = add_common_args(argparse.ArgumentParser(description="UTS DL - jalankan semua task"))
    p.add_argument("--tasks", nargs="+", type=int, default=[1, 2, 3], choices=[1, 2, 3])
    args = p.parse_args()

    from pathlib import Path
    save_json(env_info(), Path(args.out_dir) / "env_info.json")

    from kode.task1 import main as t1
    from kode.task2 import main as t2
    from kode.task3 import main as t3
    modules = {1: t1, 2: t2, 3: t3}

    failed = []
    for t in args.tasks:
        print(f"\n===== TASK {t} =====")
        try:
            # namespace baru per task agar resolve_paths satu task tidak memengaruhi task lain
            modules[t].run(argparse.Namespace(**vars(args)))
        except Exception:
            traceback.print_exc()
            failed.append(t)

    print("\n===== VALIDASI CSV =====")
    from kode import validasi_csv
    rc = validasi_csv.main(["--data_dir", args.data_dir, "--pred_dir", args.pred_dir])
    if failed:
        print(f"Task gagal: {failed}")
    sys.exit(1 if failed or rc else 0)


if __name__ == "__main__":
    main()
