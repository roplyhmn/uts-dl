"""Task 1 - Klasifikasi cacat produk (gambar + sensor). PLACEHOLDER untuk Alief.

Kontrak yang dipakai run_all.py:
  - run(args) menerima argumen dari kode.common.add_common_args
  - menulis <args.pred_dir>/<PRED_FILE> dan mengembalikan path-nya
  - grafik/tabel ke args.task_out (= <out_dir>/task1)
"""
import argparse

from kode.common import add_common_args, resolve_paths, set_seed

PRED_FILE = "prediksi_klasifikasi.csv"   # header wajib: id_unit,kelas


def run(args):
    args = resolve_paths(args, "task1")
    set_seed(args.seed)
    # TODO(Alief): EDA, training, prediksi test, tulis CSV ke args.pred_dir / PRED_FILE
    print("[task1] belum diimplementasi (placeholder) - dilewati")
    return None


def build_parser():
    return add_common_args(argparse.ArgumentParser(description="Task 1"))


if __name__ == "__main__":
    run(build_parser().parse_args())
