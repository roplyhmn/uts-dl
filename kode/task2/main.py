"""Task 2 - Prediksi Remaining Useful Life (RUL), cap 150.

Kerangka. Akan diisi bertahap:
  a. eda.py         - pengaruh setel_1..3, sensor konstan, NaN, spike, normalisasi
  b/c. data.py      - label RUL (cap 150), sliding window per mesin tanpa bocor masa depan
  d. models.py      - LSTM/GRU, 1D-CNN/TCN (+ baseline non-DL di baseline.py)
  e. evaluate.py    - split per mesin, RMSE, skor asimetris, per rentang RUL
  f. uncertainty.py - MC dropout / ensemble, prediction interval
Output: <pred_dir>/prediksi_rul.csv  (id_unit,rul ; rul di 0..150)
"""
import argparse

from kode.common import add_common_args, resolve_paths, set_seed

PRED_FILE = "prediksi_rul.csv"
RUL_CAP = 150


def asymmetric_score(y_pred, y_true):
    """d = pred - aktual; telat memprediksi (d >= 0, overestimate) dihukum lebih berat."""
    import numpy as np
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def run(args):
    args = resolve_paths(args, "task2")
    set_seed(args.seed)
    print("[task2] kerangka siap; pipeline belum diimplementasi - dilewati")
    return None


def build_parser():
    return add_common_args(argparse.ArgumentParser(description="Task 2 - RUL"))


if __name__ == "__main__":
    run(build_parser().parse_args())
