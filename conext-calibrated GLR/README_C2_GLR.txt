
Quick Start (C²-GLR)
====================

Files created:
- c2_glr.py        : Library implementing Context-Calibrated GLR + Conformal + CPD
- run_c2_glr.py    : Runner script to process your CSV, inject attacks (optional), and save outputs
- README_C2_GLR.txt: This file

Your dataset (from this chat):
- /mnt/data/combined_minute_ALL.csv

Run (no injection):
-------------------
python run_c2_glr.py --csv /mnt/data/combined_minute_ALL.csv

Run (with injected additive & deductive pulses for evaluation):
--------------------------------------------------------------
python run_c2_glr.py --csv /mnt/data/combined_minute_ALL.csv --inject --add-delta 0.5 --ded-delta 0.5 --seed 42

Outputs (written next to your CSV):
- combined_minute_ALL_detections.csv   : per-minute outputs (predictions, residuals, p-values, GLR stats, alarms)
- combined_minute_ALL_attacked.csv     : attacked series (if --inject)
- combined_minute_ALL_labels.csv       : ground-truth labels (if --inject)
- figures/*.png                        : diagnostic charts

Notes:
- The detector trains a contextual quantile forecaster on the first 14 days (configurable via --train-days).
- Conformal calibration is stratified by (occupied, hour, appliance_any) and uses sliding buffers.
- A simple CUSUM CPD resets calibration after behavior shifts (e.g., travel).
- State-gated GLR-CUSUM runs only when occupied AND (evening hours or appliance on).

Tune thresholds in DetectorParams within c2_glr.py if you need stricter/looser alarms.
