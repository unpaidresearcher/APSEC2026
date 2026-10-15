"""
Pair-level analysis of vulnerable/patched test pairs (CPU only).

For every backbone, compares the predictions with and without the bridge (from coverage.py predict)
on each vulnerable/patched pair, at each model's own validation-calibrated threshold.
Pair outcomes follow PrimeVul: correct / both vulnerable / both benign / reversed.

Pairs are formed as in PrimeVulDataset (first vulnerable and first benign function of a commit).
A pair is "same-function" when both members define the same function name. Example pairs
(code diff, retrieved KG entities, changed structural features, scores) are drawn from the
same-function pairs.

Example:
  python experiments/pair_analysis.py --coverage experiments/results/coverage --out experiments/results/pair_analysis
"""

import argparse
import csv
import difflib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from common import SPLITS, GRAPHS

BACKBONES = ["codebert_full", "unixcoder_full", "starcoder_lora", "starcoder2_lora", "qwen_lora", "deepseek_lora"]


def load_test_rows():
    rows = []
    for line in open(SPLITS["test"], encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if (r.get("func") or "").strip():
            rows.append(r)
    return rows


def pairs_of(rows):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        if r.get("commit_id"):
            groups[r["commit_id"]].append(i)
    out = []
    for idxs in groups.values():
        v = [i for i in idxs if int(rows[i].get("target", 0)) == 1]
        b = [i for i in idxs if int(rows[i].get("target", 0)) == 0]
        if v and b:
            out.append((v[0], b[0]))
    return out


def func_name(src):
    skip = {"if", "for", "while", "switch", "return", "sizeof", "typeof", "static", "inline",
            "void", "int", "char", "unsigned", "struct"}
    m = re.search(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', src)
    return m.group(1) if m and m.group(1) not in skip else "?"


def same_function(rows, v, b):
    return func_name(rows[v]["func"]) == func_name(rows[b]["func"])


def outcome(pv, pb, thr):
    v, b = pv >= thr, pb >= thr
    return "correct" if v and not b else "both_vuln" if v and b else "both_benign" if not v and not b else "reversed"


def retrieved_names(ret, func, project, max_k=32):
    """Entity names in the order PerProjectRetriever.get_caqs retrieves them."""
    project = re.sub(r'[^a-zA-Z0-9_\-]', '_', project)
    if not ret._load_project(project):
        return []
    e2i, _ = ret._cache[project]
    fname = ret._func_name(func)
    names, seen = [], set()

    def add(e):
        if e in e2i and e not in seen:
            seen.add(e); names.append(e)

    add(f"{project}::Function::{fname}")
    for var in ret._extract_local_vars(func):
        if len(names) >= max_k: break
        add(f"{project}::Variable::{fname}::{var}")
    apis = ret._extract_called_apis(func)
    for api in apis:
        if len(names) >= max_k: break
        for pre in (f"{project}::ExternalAPI::{api}", f"GLOBAL::ExternalAPI::{api}", f"{project}::Function::{api}"):
            if pre in e2i and pre not in seen:
                seen.add(pre); names.append(pre); break
    tm = re.findall(r'\b(?:struct\s+([a-zA-Z_]\w*)|([a-zA-Z_]\w*_t)\b)', func)
    types = {a or b for a, b in tm if a or b}
    for t in types:
        if len(names) >= max_k: break
        add(f"{project}::Type::{t}")
    if len(names) < max_k and types:
        for k in e2i:
            if len(names) >= max_k: break
            if k in seen or not k.startswith(f"{project}::Function::"): continue
            f = k.split("::")[-1]
            if any(a[:6] in f or f[:6] in a for a in apis):
                seen.add(k); names.append(k)
    return [n.split("::", 1)[1] if n.startswith(project + "::") else n for n in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", required=True, help="output directory of coverage.py (predictions + coverage)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_loc", type=int, default=40)
    ap.add_argument("--max_diff", type=int, default=8)
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cdir = Path(a.coverage)

    import primevul_phase1 as P
    feat_names = [n for n, _ in P.STRUCTURAL_FEATURES]
    ret = P.PerProjectRetriever(str(GRAPHS), fixed_max_k=True)
    rows = load_test_rows()
    pairs = pairs_of(rows)
    cov = [c for c in csv.DictReader(open(cdir / "coverage_per_function.csv")) if c["split"] == "test"]
    status = json.load(open(cdir / "predict_status.json"))
    assert len(cov) == len(rows)

    def preds(name):
        p = cdir / "predictions" / f"{name}.csv"
        if not p.exists() or name not in status:
            return None, None
        pr = [float(r["prob"]) for r in csv.DictReader(open(p))]
        assert len(pr) == len(rows)
        return pr, status[name]["threshold"]

    summary, report = {"n_pairs": len(pairs), "backbones": {}}, []
    report.append("# Vulnerable/patched pairs with and without the bridge\n")
    report.append(f"{len(pairs)} test pairs (first vulnerable and first benign function of each commit). "
                  "Same-function pairs are those where both members define the same function; the examples "
                  "below are drawn from them. Outcomes are at each model's validation-calibrated threshold. "
                  "'fixes' = correct with the bridge but not without it; 'breaks' = the reverse.\n")
    for bb in BACKBONES:
        base, thr_b = preds(f"{bb}_base")
        brg, thr_g = preds(f"{bb}_bridge")
        if base is None or brg is None:
            summary["backbones"][bb] = "predictions missing"
            continue
        trans = Counter()
        fixes, breaks, both_fail = [], [], []
        for v, b in pairs:
            ob, og = outcome(base[v], base[b], thr_b), outcome(brg[v], brg[b], thr_g)
            trans[(ob, og)] += 1
            if og == "correct" and ob != "correct": fixes.append((v, b))
            elif ob == "correct" and og != "correct": breaks.append((v, b))
            elif ob != "correct" and og != "correct": both_fail.append((v, b))
        same = [(v, b) for v, b in pairs if same_function(rows, v, b)]
        s_base = sum(outcome(base[v], base[b], thr_b) == "correct" for v, b in same)
        s_brg = sum(outcome(brg[v], brg[b], thr_g) == "correct" for v, b in same)
        s_fix = [p for p in fixes if same_function(rows, *p)]
        s_brk = [p for p in breaks if same_function(rows, *p)]
        s_fail = [p for p in both_fail if same_function(rows, *p)]
        n_base = sum(c for (ob, og), c in trans.items() if ob == "correct")
        n_brg = sum(c for (ob, og), c in trans.items() if og == "correct")
        summary["backbones"][bb] = {"thr_base": thr_b, "thr_bridge": thr_g,
                                    "pairs_correct_base": n_base, "pairs_correct_bridge": n_brg,
                                    "bridge_fixes": len(fixes), "bridge_breaks": len(breaks),
                                    "both_correct": n_base - len(breaks), "both_fail": len(both_fail),
                                    "transitions": {f"{k[0]}->{k[1]}": c for k, c in sorted(trans.items())},
                                    "same_function_pairs": len(same), "same_function_correct_base": s_base,
                                    "same_function_correct_bridge": s_brg, "same_function_fixes": len(s_fix),
                                    "same_function_breaks": len(s_brk), "same_function_both_fail": len(s_fail)}
        report.append(f"\n## {bb}: same-function pairs {len(same)}: correct without bridge {s_base}, "
                      f"with bridge {s_brg}; fixes {len(s_fix)}, breaks {len(s_brk)}, both fail {len(s_fail)}\n")
        report.append(f"\nAll {len(pairs)} pairs: P-C {100*n_base/len(pairs):.2f}% without vs {100*n_brg/len(pairs):.2f}% "
                      f"with the bridge | fixes {len(fixes)}, breaks {len(breaks)}, both fail {len(both_fail)}\n")

        def describe(kind, v, b):
            fv, fb = rows[v]["func"], rows[b]["func"]
            diff = list(difflib.unified_diff(fv.splitlines(), fb.splitlines(), "vulnerable", "patched", lineterm="", n=1))
            changed = sum(1 for l in diff if l[:1] in "+-" and not l.startswith(("+++", "---")))
            sv, sb = P.extract_structural(fv).tolist(), P.extract_structural(fb).tolist()
            kv, kb = retrieved_names(ret, fv, rows[v]["project"]), retrieved_names(ret, fb, rows[b]["project"])
            return {"kind": kind, "backbone": bb, "project": rows[v]["project"], "cwe": rows[v].get("cwe"),
                    "cve": rows[v].get("cve"), "commit": rows[v]["commit_id"], "loc": fv.count("\n") + 1,
                    "changed_lines": changed,
                    "scores": {"base_vuln": round(base[v], 4), "base_patched": round(base[b], 4), "thr_base": thr_b,
                               "bask_vuln": round(brg[v], 4), "bask_patched": round(brg[b], 4), "thr_bask": thr_g},
                    "outcome_base": outcome(base[v], base[b], thr_b), "outcome_bask": outcome(brg[v], brg[b], thr_g),
                    "kg_slots_vuln": int(cov[v]["n_real"]), "kg_slots_patched": int(cov[b]["n_real"]),
                    "kg_same_entities": kv == kb, "kg_entities_vuln": kv[:12],
                    "kg_only_in_vuln": [e for e in kv if e not in kb][:8],
                    "kg_only_in_patched": [e for e in kb if e not in kv][:8],
                    "features_changed": {n: (round(x, 2), round(y, 2)) for n, x, y in zip(feat_names, sv, sb)
                                         if abs(x - y) > 1e-9},
                    "diff": diff, "margin": round((brg[v] - thr_g) + (thr_g - brg[b]), 4)}

        cands = {}
        for kind, lst in (("fixes", s_fix), ("breaks", s_brk), ("both_fail", s_fail)):
            ds = [describe(kind, v, b) for v, b in lst]
            short = [d for d in ds if d["loc"] <= a.max_loc and 0 < d["changed_lines"] <= a.max_diff]
            pool = short if short else [d for d in ds if d["changed_lines"] > 0]   # else the most compact pairs
            pool.sort(key=lambda d: (d["changed_lines"], d["loc"], -bool(d["cwe"]), -d["margin"]))
            cands[kind] = pool[:a.top]
            summary["backbones"][bb][f"short_{kind}"] = len(short)
        json.dump(cands, open(out / f"examples_{bb}.json", "w"), indent=2)
        for kind in ("fixes", "breaks", "both_fail"):
            report.append(f"\n### {kind} ({summary['backbones'][bb][f'short_{kind}']} short; top {len(cands[kind])})\n")
            for d in cands[kind]:
                s = d["scores"]
                report.append(f"- **{d['project']}** {d['cve'] or ''} {d['cwe'] or ''} commit {d['commit'][:10]} | "
                              f"{d['loc']} lines, {d['changed_lines']} changed | without bridge "
                              f"{s['base_vuln']:.3f}/{s['base_patched']:.3f} (thr {s['thr_base']}), with bridge "
                              f"{s['bask_vuln']:.3f}/{s['bask_patched']:.3f} (thr {s['thr_bask']}) | "
                              f"KG slots {d['kg_slots_vuln']}/{d['kg_slots_patched']}, same entities: "
                              f"{d['kg_same_entities']} | features changed: {d['features_changed']}\n")
                report.append("```diff\n" + "\n".join(d["diff"][2:40]) + "\n```\n")
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    open(out / "examples.md", "w").write("\n".join(report))


if __name__ == "__main__":
    main()
