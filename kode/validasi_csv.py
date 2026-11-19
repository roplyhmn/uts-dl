"""Validasi format CSV prediksi sebelum submit.

Pemakaian:
  python -m kode.validasi_csv                 # cek semua CSV yang ada di root repo
  python -m kode.validasi_csv --data_dir /content/data --pred_dir .
Keluar dengan kode 1 jika ada kesalahan.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kode.common.config import DATA_DIR, DATA_FILES, REPO_ROOT


def _ids_task2(data_dir):
    return set(pd.read_csv(Path(data_dir) / DATA_FILES["task2_test"], usecols=["id_unit"]).id_unit.unique())


def _ids_task3(data_dir):
    return set(pd.read_csv(Path(data_dir) / DATA_FILES["task3_meta"], usecols=["id_sampel"]).id_sampel)


def _ids_task1(data_dir):
    return set(pd.read_csv(Path(data_dir) / DATA_FILES["task1_test_meta"], usecols=["id_unit"]).id_unit)


KELAS_TASK1 = {"normal", "gores", "noda", "retak", "gelembung", "las_dingin", "kontaminasi"}

SPECS = {
    "prediksi_klasifikasi.csv": dict(cols=["id_unit", "kelas"], id_col="id_unit", ids=_ids_task1,
                                     val_col="kelas", categories=KELAS_TASK1),
    "prediksi_rul.csv": dict(cols=["id_unit", "rul"], id_col="id_unit", ids=_ids_task2,
                             val_col="rul", vmin=0, vmax=150, integer=False),
    "cluster.csv": dict(cols=["id_sampel", "cluster"], id_col="id_sampel", ids=_ids_task3,
                        val_col="cluster", vmin=None, vmax=None, integer=True),
}


def validate(path: Path, spec: dict, data_dir) -> list[str]:
    err = []
    df = pd.read_csv(path)
    if list(df.columns) != spec["cols"]:
        err.append(f"header harus {spec['cols']}, didapat {list(df.columns)}")
        return err
    exp = spec["ids"](data_dir)
    got = df[spec["id_col"]]
    if len(df) != len(exp):
        err.append(f"jumlah baris {len(df)}, seharusnya {len(exp)}")
    if got.duplicated().any():
        err.append(f"{got.duplicated().sum()} id duplikat")
    # samakan tipe untuk perbandingan himpunan id
    got_set = set(got.astype(type(next(iter(exp)))) if len(exp) else got)
    if missing := exp - got_set:
        err.append(f"{len(missing)} id hilang, contoh {sorted(missing)[:5]}")
    if extra := got_set - exp:
        err.append(f"{len(extra)} id tidak dikenal, contoh {sorted(extra)[:5]}")
    if "categories" in spec:   # label kategorikal (Task 1)
        col = df[spec["val_col"]]
        if col.isna().any():
            err.append(f"{col.isna().sum()} nilai {spec['val_col']} kosong")
        if bad := set(col.dropna()) - spec["categories"]:
            err.append(f"kelas tidak dikenal: {sorted(bad)[:5]}")
        return err
    v = pd.to_numeric(df[spec["val_col"]], errors="coerce")
    if v.isna().any():
        err.append(f"{v.isna().sum()} nilai {spec['val_col']} kosong / bukan angka")
    if spec["vmin"] is not None and (v < spec["vmin"]).any():
        err.append(f"ada {spec['val_col']} < {spec['vmin']}")
    if spec["vmax"] is not None and (v > spec["vmax"]).any():
        err.append(f"ada {spec['val_col']} > {spec['vmax']}")
    if spec["integer"] and not np.all(np.mod(v.dropna(), 1) == 0):
        err.append(f"{spec['val_col']} harus integer")
    return err


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=str(DATA_DIR))
    p.add_argument("--pred_dir", default=str(REPO_ROOT))
    p.add_argument("files", nargs="*", help="nama CSV tertentu (default: semua di SPECS)")
    a = p.parse_args(argv)
    names = a.files or list(SPECS)
    ok = True
    for name in names:
        path = Path(a.pred_dir) / name
        if not path.exists():
            print(f"[--] {name}: belum ada")
            continue
        errs = validate(path, SPECS[name], a.data_dir)
        ok &= not errs
        print(f"[{'OK' if not errs else 'GAGAL'}] {name}" + "".join(f"\n     - {e}" for e in errs))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
