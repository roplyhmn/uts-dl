# Setup & menjalankan

## Lokal (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt    # torch CPU cukup untuk mode --debug
python run_all.py --debug          # uji cepat, subset kecil
```

## Colab / Kaggle (training penuh)

Buka `notebooks/colab_train.ipynb` → Runtime GPU → jalankan semua sel. Notebook me-mount Drive,
clone repo, install requirements, ekstrak `data_uts_dl.zip` ke `/content/data`, lalu menjalankan
training dengan `--out_dir` di Drive. Kaggle: `--data_dir /kaggle/input/<dataset> --out_dir /kaggle/working/out`.

## Argumen umum (semua task)

| Argumen | Default | Fungsi |
| --- | --- | --- |
| `--data_dir` | `<repo>/data` | lokasi dataset |
| `--out_dir` | `<repo>/outputs` | checkpoint, grafik, tabel (`<out_dir>/taskN/`) |
| `--pred_dir` | root repo | lokasi CSV prediksi final |
| `--seed` | 42 | seed global |
| `--debug` | off | subset kecil + epoch sedikit; hasil ke `<out_dir>/taskN/debug/` |
| `--resume/--no-resume` | resume | lanjutkan dari checkpoint terakhir |
| `--tasks` (run_all) | 1 2 3 | task yang dijalankan |

Per task juga bisa dijalankan sendiri: `python -m kode.task2.main --debug`.

## Validasi CSV sebelum submit

```powershell
python -m kode.validasi_csv
```
Mengecek header, jumlah baris = jumlah unit/sampel test, id lengkap & unik, rentang nilai
(kelas Task 1 valid, RUL 0–150, cluster integer). `run_all.py` menjalankannya otomatis di akhir.
