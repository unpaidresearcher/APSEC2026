"""
Measure the cost of BASK. Each step writes one JSON file to <out>/.

  kg_build   build the per-project graphs (src/build_primevul_graphs.py): wall/CPU time, peak RAM
  transh     train the TransH embeddings (src/train_primevul_transh.py): wall/CPU time, peak RAM
  storage    on-disk size of graphs, embeddings and index
  retrieval  per-function KG retrieval + structural-feature extraction time (CPU, one thread)
  train      seconds per training step and peak GPU memory, with and without the bridge
  infer      inference time per test function and peak GPU memory, with and without the bridge;
             also recomputes the test metrics of each checkpoint and compares them with its results.json

Example:
  python experiments/cost.py train --out experiments/results/cost --configs codebert_full_base,codebert_full_bridge
"""

import argparse
import json
import re
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

from common import CONFIGS, SPLITS, GRAPHS, CKPT, SRC, METRIC_KEYS, log, build, predict


def write_result(out, step, payload, merge=False):
    p = Path(out) / f"{step}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    if merge and p.exists():
        old = json.load(open(p))
        old.update(payload)
        payload = old
    json.dump(payload, open(p, "w"), indent=2)
    log(f"wrote {p}")


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))]


def summarize_ms(xs):
    return {"n": len(xs), "mean_ms": statistics.fmean(xs), "median_ms": pct(xs, 50),
            "p95_ms": pct(xs, 95), "p99_ms": pct(xs, 99), "max_ms": max(xs),
            "total_s": sum(xs) / 1000}


def run_timed(cmd):
    """Run a script as a child process and return its wall time, CPU time and peak RSS."""
    log("RUN", " ".join(map(str, cmd)))
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.perf_counter()
    proc = subprocess.run(list(map(str, cmd)), cwd=SRC)
    wall = time.perf_counter() - t0
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {"returncode": proc.returncode, "wall_s": wall,
            "user_cpu_s": r1.ru_utime - r0.ru_utime, "sys_cpu_s": r1.ru_stime - r0.ru_stime,
            "peak_rss_gb": r1.ru_maxrss / 1024 / 1024}


def cmd_kg_build(a):
    res = run_timed([sys.executable, SRC / "build_primevul_graphs.py",
                     "--train", SPLITS["train"], "--valid", SPLITS["valid"],
                     "--test", SPLITS["test"], "--out", a.graphs, "--min_funcs", 10])
    stats = json.load(open(Path(a.graphs) / "graph_stats.json"))
    res.update({"n_projects": len(stats),
                "n_nodes": sum(s["n_nodes"] for s in stats.values()),
                "n_edges": sum(s["n_edges"] for s in stats.values()),
                "n_funcs": sum(s["n_funcs"] for s in stats.values())})
    write_result(a.out, "kg_build", res)


def cmd_transh(a):
    res = run_timed([sys.executable, SRC / "train_primevul_transh.py",
                     "--graphs", a.graphs, "--out", a.graphs,
                     "--epochs", 50, "--dim", 128, "--device", "cuda"])
    g = Path(a.graphs)
    stats = json.load(open(g / "transh_stats.json")) if (g / "transh_stats.json").exists() else []
    res.update({"n_projects_trained": len(stats),
                "n_entities": sum(s["n_entities"] for s in stats),
                "n_triples": sum(s["n_triples"] for s in stats)})
    write_result(a.out, "transh", res)


def cmd_storage(a):
    g = Path(a.graphs)

    def size(pattern):
        fs = list(g.glob(pattern))
        return {"files": len(fs), "mb": sum(f.stat().st_size for f in fs) / 1e6}

    res = {"graph_gpickle": size("*/graph.gpickle"),
           "entity_embeddings_pt": size("*/entity_embeddings.pt"),
           "entity_to_idx_json": size("*/entity_to_idx.json"),
           "global_index_json_mb": (g / "global_index.json").stat().st_size / 1e6
                                   if (g / "global_index.json").exists() else None}
    res["total_mb"] = sum(f.stat().st_size for f in g.rglob("*") if f.is_file()) / 1e6
    res["needed_at_inference_mb"] = res["entity_embeddings_pt"]["mb"] + res["entity_to_idx_json"]["mb"]
    write_result(a.out, f"storage_{a.tag}" if a.tag else "storage", res)


def cmd_retrieval(a):
    import torch
    from primevul_phase1 import PerProjectRetriever, extract_structural
    torch.set_num_threads(1)
    ret = PerProjectRetriever(a.graphs, fixed_max_k=True)
    load_ms, kg_ms, feat_ms, n_real = [], [], [], []
    no_project = 0
    for split, path in SPLITS.items():
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            func = (row.get("func") or "").strip()
            if not func:
                continue
            proj = re.sub(r'[^a-zA-Z0-9_\-]', '_', (row.get("project") or "unknown").strip())
            if proj not in ret._cache:            # one-off project load, timed separately
                t = time.perf_counter()
                if ret._load_project(proj):
                    load_ms.append((time.perf_counter() - t) * 1000)
                else:
                    no_project += 1
            t = time.perf_counter()
            c = ret.get_caqs(func, row.get("project") or "unknown")
            kg_ms.append((time.perf_counter() - t) * 1000)
            n_real.append(int((c.abs().sum(-1) > 0).sum()))
            t = time.perf_counter()
            extract_structural(func)
            feat_ms.append((time.perf_counter() - t) * 1000)
        log(f"{split} done")
    res = {"threads": 1,
           "project_load": summarize_ms(load_ms),
           "kg_retrieval": summarize_ms(kg_ms),
           "structural_features": summarize_ms(feat_ms),
           "retrieval_plus_features_mean_ms": statistics.fmean(kg_ms) + statistics.fmean(feat_ms),
           "functions_without_project_graph": no_project,
           "n_real_slots": {"mean": statistics.fmean(n_real),
                            "zero": sum(1 for x in n_real if x == 0),
                            "full_32": sum(1 for x in n_real if x == 32)}}
    write_result(a.out, f"retrieval_{a.tag}" if a.tag else "retrieval", res)


def gpu_mem(torch):
    return {"peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30}


def cmd_train(a):
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup
    import primevul_phase1 as P
    import primevul_phase1_auc as A
    device = torch.device("cuda")
    retriever = P.PerProjectRetriever(str(a.graphs), fixed_max_k=True)
    results = {}
    for name in (a.configs.split(",") if a.configs else list(CONFIGS)):
        log(f"train {name}")
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model, tok, use_graph, info = build(name, device)
        bs = info["batch_size"]
        ds = P.PrimeVulDataset(str(SPLITS["train"]), tok, retriever if use_graph else None, use_graph)
        sampler = A.BalancedSampler(ds, bs, A.MIN_VULN_PER_BATCH)
        pad_id = tok.pad_token_id or 0
        ldr = DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=0, drop_last=True,
                         collate_fn=lambda b: P.collate(b, pad_id))
        crit = A.CombinedAUCLoss(alpha=A.ALPHA, tau=A.TAU, hard_neg_k=A.HARD_NEG_K)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=info["lr"], weight_decay=0.01)
        sch = get_linear_schedule_with_warmup(opt, A.WARMUP_STEPS, P.EPOCHS * len(ldr))
        model.train()
        it = iter(ldr)
        data_s, step_s = [], []
        for step in range(a.warmup + a.steps):
            t0 = time.perf_counter()
            batch = next(it)                    # tokenisation + retrieval + features (CPU)
            t1 = time.perf_counter()
            kw = {}
            if use_graph:
                kw = {"C_AQ": batch["C_AQ"].to(device), "graph_mask": batch["graph_mask"].to(device),
                      "struct_feats": batch["struct_feats"].to(device)}
            loss = crit(model(batch["input_ids"].to(device), batch["attention_mask"].to(device), **kw),
                        batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), P.MAX_GRAD_NORM)
            opt.step(); sch.step(); opt.zero_grad()
            torch.cuda.synchronize()
            if step >= a.warmup:
                data_s.append(t1 - t0); step_s.append(time.perf_counter() - t0)
        mean_step = statistics.fmean(step_s)
        info.update({"warmup_steps": a.warmup, "timed_steps": a.steps, "steps_per_epoch": len(ldr),
                     "step_s_mean": mean_step, "step_s_median": statistics.median(step_s),
                     "data_s_mean": statistics.fmean(data_s),
                     "gpu_s_mean": mean_step - statistics.fmean(data_s),
                     "est_epoch_h": mean_step * len(ldr) / 3600,
                     "est_10_epoch_train_h": mean_step * len(ldr) * P.EPOCHS / 3600,
                     **gpu_mem(torch)})
        log(f"  {mean_step:.3f} s/step, peak {info['peak_allocated_gb']:.1f} GB")
        results[name] = info
        del model, opt, sch, ldr, ds, it
        torch.cuda.empty_cache()
        write_result(a.out, "train", results, merge=True)


def cmd_infer(a):
    import torch
    import primevul_phase1 as P
    device = torch.device("cuda")
    retriever = P.PerProjectRetriever(str(a.graphs), fixed_max_k=True)
    results = {}
    for name in (a.configs.split(",") if a.configs else list(CONFIGS)):
        log(f"infer {name}")
        torch.cuda.empty_cache()
        model, tok, use_graph, info = build(name, device)
        ck = CKPT / info["checkpoint"]
        model.load_state_dict(torch.load(ck / "best.pt", map_location=device, weights_only=True), strict=True)
        model.eval()
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t_all = time.perf_counter()
        labels, probs, ds, data_s, gpu_s = predict(model, tok, use_graph, "test", device,
                                                   P.EVAL_BATCH_SIZE, retriever)
        total = time.perf_counter() - t_all
        pub = json.load(open(ck / "results.json"))
        thr = pub.get("best_threshold", 0.5)
        m = P.compute_metrics(labels, probs, threshold=thr, dataset=ds)
        info.update({"n_test": len(labels), "eval_batch_size": P.EVAL_BATCH_SIZE,
                     "total_s": total, "data_s": data_s, "gpu_s": gpu_s,
                     "ms_per_function": 1000 * total / len(labels),
                     "gpu_ms_per_function": 1000 * gpu_s / len(labels), **gpu_mem(torch),
                     "recomputed": {k: m[k] for k in METRIC_KEYS},
                     "published": {k: pub["test"][k] for k in METRIC_KEYS},
                     "threshold_used": thr})
        info["matches_published"] = all(abs(info["recomputed"][k] - info["published"][k])
                                        < 1e-3 * max(1, abs(info["published"][k])) for k in METRIC_KEYS)
        log(f"  {info['ms_per_function']:.2f} ms/function, match={info['matches_published']}")
        results[name] = info
        del model, ds
        torch.cuda.empty_cache()
        write_result(a.out, "infer", results, merge=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["kg_build", "transh", "storage", "retrieval", "train", "infer"])
    ap.add_argument("--out", required=True, help="output directory for the JSON results")
    ap.add_argument("--graphs", default=str(GRAPHS), help="graph directory (kg_build/transh write here)")
    ap.add_argument("--tag", default="", help="suffix for the storage/retrieval output file")
    ap.add_argument("--configs", default="", help="comma-separated names from common.CONFIGS (default: all)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=200)
    a = ap.parse_args()
    globals()[f"cmd_{a.step}"](a)
