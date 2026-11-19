# Task 1 – Klasifikasi cacat produk (dikerjakan Alief)

Placeholder. Isi `main.py` dengan fungsi `run(args)` sesuai kontrak di docstring-nya, supaya
`python run_all.py` bisa menjalankan ketiga task secara berurutan.

- Data: `klasifikasi_train.npz`, `klasifikasi_test.npz`, `klasifikasi_*_meta.csv` (lihat `kode/common/config.py`).
- Pakai `kode.common` untuk seed, path (`--data_dir`, `--out_dir`), checkpoint/resume.
- Grafik dan tabel ke `outputs/task1/`; CSV prediksi ke root repo.
- Output wajib: `prediksi_klasifikasi.csv`, header `id_unit,kelas`, satu baris per unit test. Formatnya sudah dicek oleh `kode/validasi_csv.py`.
