"""Mean and sample standard deviation of test metrics over seeds 42, 123, 456 and 123456.

Example:
  python experiments/summarize_seeds.py --runs experiments/results/multi_seed/runs --out experiments/results/multi_seed/summary.json
"""

import argparse
import json
import statistics
from pathlib import Path

SYSTEMS = {"CodeBERT base": "A_codebert", "CodeBERT+BASK": "D_codebert_bridge",
           "UniXcoder base": "B_unixcoder", "UniXcoder+BASK": "E_unixcoder_bridge"}
SEEDS = (42, 123, 456, 123456)
METRICS = ("VD-S_FNR", "AUC", "F1", "P-C")

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True, help="folder with <system>_seed<seed>/results.json")
ap.add_argument("--out", required=True)
a = ap.parse_args()
summary = {}
for label, system in SYSTEMS.items():
    tests = [json.load(open(Path(a.runs) / f"{system}_seed{s}" / "results.json"))["test"] for s in SEEDS]
    summary[label] = {m: [statistics.fmean(t[m] for t in tests), statistics.stdev(t[m] for t in tests)]
                      for m in METRICS}
json.dump(summary, open(a.out, "w"), indent=2)
for label, v in summary.items():
    print(f"{label:<16}" + "  ".join(f"{m} {v[m][0]:.3f}+/-{v[m][1]:.3f}" for m in METRICS))
