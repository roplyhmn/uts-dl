# UTS Deep Learning – Kelompok Rofly & Alief

| Task | Topik | Penanggung jawab | Output |
| --- | --- | --- | --- |
| 1 | Klasifikasi cacat produk (gambar + sensor) | Alief | `prediksi_klasifikasi.csv` |
| 2 | Regresi RUL time-series | Rofly | `prediksi_rul.csv` |
| 3 | Clustering multimodal material | Rofly | `cluster.csv` |

Reproduksi semua angka di laporan: `python run_all.py` (detail di [kode/README.md](kode/README.md)).

## Data

Dataset diambil dari **link resmi dosen** (`data_uts_dl.zip`); tidak ada data eksternal yang dipakai.
Dataset **tidak di-commit** (`data/` ada di `.gitignore`).

Ekstrak isi zip ke folder `data/` di root repo sehingga strukturnya:

```
uts-dl/
└── data/
    ├── kamus_data.md
    ├── klasifikasi_train.npz       klasifikasi_train_meta.csv     # Task 1
    ├── klasifikasi_test.npz        klasifikasi_test_meta.csv
    ├── rul_train.csv.gz            rul_test.csv.gz                # Task 2
    └── clustering.npz              clustering_meta.csv            # Task 3
```

Lokasi lain bisa dipakai lewat `--data_dir` (mis. `/content/data` di Colab,
`/kaggle/input/<nama-dataset>` di Kaggle) atau env `UTS_DATA_DIR`.

## Struktur repo

```
kode/common/     seed, path config (DATA_DIR/OUT_DIR), checkpoint & utils
kode/task1/      Task 1 (Alief)
kode/task2/      Task 2 – RUL
kode/task3/      Task 3 – clustering
kode/validasi_csv.py   cek format CSV sebelum submit
notebooks/colab_train.ipynb   training di Colab (Drive, resume)
outputs/task2/, outputs/task3/   grafik & tabel untuk laporan
run_all.py       menjalankan task 1→2→3 dan menulis CSV prediksi ke root repo
```
