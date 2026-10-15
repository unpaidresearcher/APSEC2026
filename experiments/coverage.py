"""
Detection performance as a function of KG coverage.

  coverage   (CPU) number of real (non-[PAD]) KG slots, 0..32, for every PrimeVul function
  predict    (GPU) per-function test probabilities of each checkpoint
  buckets    (CPU) metrics per coverage bucket, with and without the bridge

Example:
  python experiments/coverage.py coverage --out experiments/results/coverage
  python experiments/coverage.py predict  --out experiments/results/coverage --configs codebert_full_base,codebert_full_bridge
  python experiments/coverage.py buckets  --out experiments/results/coverage
"""

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from common import CONFIGS, SPLITS, GRAPHS, CKPT, METRIC_KEYS, log, build, predict

# Every backbone with and without the bridge, at its main adaptation mode.
PAIRED = [f"{bb}_{v}" for bb in ("codebert_full", "unixcoder_full", "starcoder_lora", "starcoder2_lora",
                                 "qwen_lora", "deepseek_lora") for v in ("base", "bridge")]
BUCKETS = [("0", 0, 0), ("1-7", 1, 7), ("8-15", 8, 15), ("16-31", 16, 31), ("32", 32, 32)]


def bucket_of(n):
    for name, lo, hi in BUCKETS:
        if lo <= n <= hi:
            return name


def cmd_coverage(a):
    from primevul_phase1 import PerProjectRetriever
    ret = PerProjectRetriever(str(GRAPHS), fixed_max_k=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for split, path in SPLITS.items():
        i = 0
        for line in open(path, encoding="utf-8"):          # same rows and filters as PrimeVulDataset
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            func = (row.get("func") or "").strip()
            if not func:
                continue
            project = (row.get("project") or "unknown").strip()
            proj_key = re.sub(r'[^a-zA-Z0-9_\-]', '_', project)
            has_graph = ret._load_project(proj_key)
            n_real = int((ret.get_caqs(func, project).abs().sum(-1) > 0).sum())
            rows_out.append({"split": split, "row": i, "project": proj_key,
                             "target": int(row.get("target", 0)), "commit_id": row.get("commit_id", ""),
                             "has_project_graph": int(has_graph), "n_real": n_real,
                             "bucket": bucket_of(n_real)})
            i += 1
        log(f"{split}: {i} functions")
    with open(out / "coverage_per_function.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)

    def summ(rs):
        n = len(rs)
        no_graph = sum(1 for r in rs if not r["has_project_graph"])
        return {"n": n, "no_project_graph": no_graph, "no_project_graph_pct": 100 * no_graph / max(n, 1),
                "mean_real_slots": statistics.fmean(r["n_real"] for r in rs) if rs else 0,
                "median_real_slots": statistics.median(r["n_real"] for r in rs) if rs else 0,
                "buckets": {b: sum(1 for r in rs if r["bucket"] == b) for b, _, _ in BUCKETS}}

    summary = {"all": summ(rows_out)}
    for split in SPLITS:
        for lab, name in ((None, "all"), (1, "vuln"), (0, "benign")):
            summary[f"{split}_{name}"] = summ([r for r in rows_out if r["split"] == split
                                               and (lab is None or r["target"] == lab)])
    per_proj = defaultdict(list)
    for r in rows_out:
        per_proj[r["project"]].append(r)
    summary["per_project"] = {p: {"n": len(rs), "has_graph": rs[0]["has_project_graph"],
                                  "mean_real_slots": statistics.fmean(r["n_real"] for r in rs),
                                  "n_vuln": sum(r["target"] for r in rs)}
                              for p, rs in sorted(per_proj.items(), key=lambda x: -len(x[1]))}
    summary["projects_total"] = len(per_proj)
    summary["projects_without_graph"] = sum(1 for v in summary["per_project"].values() if not v["has_graph"])
    json.dump(summary, open(out / "coverage_summary.json", "w"), indent=2)


def cmd_predict(a):
    import torch
    import primevul_phase1 as P
    out = Path(a.out); (out / "predictions").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    retriever = P.PerProjectRetriever(str(GRAPHS), fixed_max_k=True)
    status_path = out / "predict_status.json"
    status = json.load(open(status_path)) if status_path.exists() else {}
    for name in (a.configs.split(",") if a.configs else PAIRED):
        log(f"predict {name}")
        model, tok, use_graph, info = build(name, device)
        ck = CKPT / info["checkpoint"]
        model.load_state_dict(torch.load(ck / "best.pt", map_location=device, weights_only=True), strict=True)
        model.eval()
        labels, probs, ds, _, _ = predict(model, tok, use_graph, "test", device, a.eval_bs, retriever)
        pub = json.load(open(ck / "results.json"))
        thr = pub.get("best_threshold", 0.5)
        m = P.compute_metrics(labels, probs, threshold=thr, dataset=ds)
        with open(out / "predictions" / f"{name}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["row", "target", "prob"])
            w.writerows((i, l, p) for i, (l, p) in enumerate(zip(labels, probs)))
        status[name] = {"checkpoint": info["checkpoint"], "threshold": thr, "eval_batch_size": a.eval_bs,
                        "recomputed": {k: m[k] for k in METRIC_KEYS},
                        "published": {k: pub["test"][k] for k in METRIC_KEYS},
                        "matches_published": all(abs(m[k] - pub["test"][k]) < 1e-3 * max(1, abs(pub["test"][k]))
                                                 for k in METRIC_KEYS)}
        del model, ds
        torch.cuda.empty_cache()
        json.dump(status, open(status_path, "w"), indent=2)


def cmd_buckets(a):
    import numpy as np
    from sklearn.metrics import roc_auc_score, roc_curve, f1_score
    out = Path(a.out)
    cov = [r for r in csv.DictReader(open(out / "coverage_per_function.csv")) if r["split"] == "test"]
    status = json.load(open(out / "predict_status.json"))
    res = {"bucket_definition": "number of real (non-[PAD]) KG slots out of 32",
           "pair_rule": "a vulnerable/patched pair is assigned to the bucket of its vulnerable function",
           "vds_global_rule": "FNR inside the bucket at the threshold where FPR=0.5% on the whole test set",
           "configs": {}}
    for pf in sorted((out / "predictions").glob("*.csv")):
        name = pf.stem
        pr = list(csv.DictReader(open(pf)))
        assert len(pr) == len(cov), f"{name}: {len(pr)} predictions vs {len(cov)} coverage rows"
        y = np.array([int(r["target"]) for r in pr]); p = np.array([float(r["prob"]) for r in pr])
        assert all(int(c["target"]) == t for c, t in zip(cov, y)), f"{name}: label order mismatch"
        thr = status[name]["threshold"]
        fpr, _, th = roc_curve(y, p)
        glob_thr = th[fpr <= 0.005][-1]                   # loosest threshold with FPR <= 0.5%
        groups = defaultdict(list)                        # pairs as in PrimeVulDataset
        for i, c in enumerate(cov):
            if c["commit_id"]:
                groups[c["commit_id"]].append(i)
        pairs = []
        for idxs in groups.values():
            v = [i for i in idxs if y[i] == 1]; b = [i for i in idxs if y[i] == 0]
            if v and b:
                pairs.append((v[0], b[0]))
        per = {}
        for bname, _, _ in BUCKETS + [("no_project_graph", None, None)]:
            if bname == "no_project_graph":
                idx = np.array([i for i, c in enumerate(cov) if c["has_project_graph"] == "0"], dtype=int)
                bp = [(v, b) for v, b in pairs if cov[v]["has_project_graph"] == "0"]
            else:
                idx = np.array([i for i, c in enumerate(cov) if c["bucket"] == bname], dtype=int)
                bp = [(v, b) for v, b in pairs if cov[v]["bucket"] == bname]
            yb, pb = y[idx], p[idx]
            nv, nb = int(yb.sum()), int(len(yb) - yb.sum())
            d = {"n": int(len(yb)), "n_vuln": nv, "n_benign": nb, "n_pairs": len(bp)}
            if nv and nb:
                bf, bt, _ = roc_curve(yb, pb)
                d["AUC"] = float(roc_auc_score(yb, pb))
                d["VD-S_FNR_within_bucket"] = float((1 - bt[bf <= 0.005].max()) * 100)
            if nv:
                d["FNR_at_global_0.5pct_FPR"] = float(100 * np.mean(pb[yb == 1] < glob_thr))
                d["F1"] = float(f1_score(yb, (pb >= thr).astype(int), zero_division=0))
            if bp:
                d["P-C"] = 100 * sum(1 for v, b in bp if p[v] >= thr and p[b] < thr) / len(bp)
            per[bname] = d
        res["configs"][name] = {"threshold": thr, "global_0.5pct_fpr_threshold": float(glob_thr), "buckets": per}
    json.dump(res, open(out / "buckets.json", "w"), indent=2)
    log("buckets done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["coverage", "predict", "buckets"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", default="", help="comma-separated names from common.CONFIGS (default: PAIRED)")
    ap.add_argument("--eval_bs", type=int, default=128, help="inference batch size")
    a = ap.parse_args()
    globals()[f"cmd_{a.step}"](a)
