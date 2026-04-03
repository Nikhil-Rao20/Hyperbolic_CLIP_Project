# CPU to GPU Handoff Checklist

Use this when preparing on CPU and executing on GPU.

## 1) Commit and push from CPU machine

```bash
git status
git add -A
git commit -m "Baseline fairness fixes and GPU handoff readiness"
git push
```

## 2) Pull on GPU machine

```bash
git pull
```

## 3) Fresh environment on GPU machine

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## 4) Baseline repo/bootstrap setup

```bash
python scripts/setup_official_baselines.py
```

## 5) Sequential runs

### WinCLIP (fairness-fixed)
```bash
python scripts/run_winclip_official.py --device cuda --threshold-percentile 95
```

### SimpleNet (fairness-checked)
```bash
python scripts/run_simplenet_baseline.py --device cuda --seed 42
```

## 6) Output checks

### WinCLIP
- experiments/WinClip_Official/winclip_official/final_8run_summary_per_fold.csv
- experiments/WinClip_Official/winclip_official/final_8run_summary.csv
- experiments/WinClip_Official/winclip_official/final_8run_summary_stats.csv

### SimpleNet
- experiments/SimpleNet_Official/simplenet_official/final_8run_summary_per_fold.csv
- experiments/SimpleNet_Official/simplenet_official/final_8run_summary.csv
- experiments/SimpleNet_Official/simplenet_official/final_8run_summary_stats.csv

## Notes

- WinCLIP now calibrates thresholds from fold validation splits (no test-set calibration leakage).
- SimpleNet reloads fold best checkpoint before final calibration/evaluation.
- Both are aligned to the same manifest protocol structure.
