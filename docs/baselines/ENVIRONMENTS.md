# Baseline environments

Use a dedicated environment so baseline-only compatibility packages do not replace the project's approved runtime. Chemprop 1.6.1 requires NumPy 1.x because its import path uses an API removed in NumPy 2.x.

Local Windows verification can use an already activated compatible interpreter as a system-package base:

```powershell
python -m venv --system-site-packages .venv-baselines
& '.venv-baselines\Scripts\python.exe' -m pip install -r requirements-baselines.txt
```

Linux server setup uses the same isolation model while retaining the server's working PyTorch/CUDA stack:

```bash
cd /home/shangzeli/RGCER
python3 -m venv --system-site-packages .venv-baselines
source .venv-baselines/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-baselines.txt
```

Chemprop and GROVER are loaded from exact official source commits passed on the command line, not from an untracked workstation path. The TOXACol source checkout is required by the READY preflight for source and license identity; runtime uses the reviewed transcription. Official sources and weights stay outside Git.
