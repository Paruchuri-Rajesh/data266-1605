# START HERE — Windows GPU lab PC

Copy the whole `hw2_5` folder onto the lab machine (USB, OneDrive, git —
whatever your lab allows).

You already have a seed. **SID4 = 1605, SEED = 1605.** Do not change it.
This assignment does **not** need a second HP_ID model.

Do this on **one** RTX 4090 or RTX 5090 PC (the assignment says "either ...
or", not both). If you can book time on the second card too, the scripts and
`06_compile_report.py` will happily add it as a second column, but it is not
required to get full credit.

## 0. Before you sit down

- Book the machine. Screenshot/save the reservation into `reservations\`.
- In Windows: **Settings → System → Power → Screen and sleep → Sleep = Never** while you run.
- Do not close the Command Prompt window during the 20-minute thermal part.

## 1. Open a CUDA Python

On the lab PC, use **their** GPU Python (Anaconda Prompt, "PyTorch CUDA", or
whatever the lab sheet says). Check:

```bat
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You want `True` and `NVIDIA GeForce RTX 4090` or `RTX 5090`.
If that prints `False`, you are in the wrong Python. Do not pip-install a CPU torch.

Then:

```bat
cd path\to\hw2_5
python -m pip install numpy matplotlib pandas
python scripts\check_env.py
```

`check_env.py` must print `ok` and the GPU name.

## 2. Smoke test first (2–4 min)

Double-click **`run_quick.bat`** in this folder,
or in Command Prompt:

```bat
python scripts\run_all.py --quick --skip-thermal
```

If this dies, fix Python / CUDA before you burn the reservation on a 20-minute job.

## 3. Full run (~45–60 min, includes 20 min thermal)

Double-click **`run_full.bat`**, or:

```bat
python scripts\run_all.py
```

Leave the window open. After it finishes, fill `reservations\GPU_HOURS.md`
**on that PC** (login/logout, hours reserved vs used, UUID from the printout).

## 4. Copy off this machine

Take the whole `results\` folder (it will have a subfolder named like
`NVIDIAGeForceRTX4090_abcd1234`). Also copy `figures\` and `RUN_LOG.txt`.

## After the run (your laptop is fine)

With the `results\<gpu>_<uuid>` folder in place:

```bat
python scripts\06_compile_report.py
```

That writes Table HW2.5.1 into `METRICS.md` and the plots into `figures\`.

Then finish `AI_USE.md` (paste a real GPU failure), fill numbers in
`HW2.5_Report.md`, export PDF, commit in `data266-1605`, tag **`hw2-5`**.

## Time budget per card

| Step | Time |
|------|------|
| Part A nvidia-smi | ~1 min |
| Part B GEMM | ~5–10 min |
| Part C roofline | ~2–3 min |
| Part D attention + OOM search | ~10–20 min |
| Part E thermal | **20 min** |
| **Total** | **~45–60 min** |

## If something is missing

- `nvidia-smi` not found: the NVIDIA driver is not on PATH. `check_env` / Part A look in `C:\Windows\System32\nvidia-smi.exe`.
- Out of memory in Part D is expected at long sequences. The script records it.
- FP8/FP4 failing is a valid finding. Leave the error in the log.
