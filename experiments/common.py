"""Shared paths, model configurations and model construction for the scripts in experiments/."""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

DATA = ROOT / "data" / "primevul"
SPLITS = {s: DATA / f"primevul_{s}.jsonl" for s in ("train", "valid", "test")}
GRAPHS = ROOT / "data" / "primevul_graphs"
CKPT = ROOT / "checkpoints"

# name -> (system, backbone_mode, batch_size, lr, checkpoint dir under checkpoints/)
# Batch sizes are the ones each checkpoint was trained with.
CONFIGS = {
    "codebert_full_base":     ("A_codebert",          "full",   64, 2e-5, "encoders_base/A_codebert"),
    "codebert_full_bridge":   ("D_codebert_bridge",   "full",   64, 2e-5, "encoders_bask/D_codebert_bridge"),
    "codebert_lora_bridge":   ("D_codebert_bridge",   "lora",   64, 2e-4, "encoders_bask_lora/D_codebert_bridge"),
    "codebert_frozen_bridge": ("D_codebert_bridge",   "frozen", 64, 2e-4, "encoders_bask_frozen/D_codebert_bridge"),
    "unixcoder_full_base":    ("B_unixcoder",         "full",   64, 2e-5, "encoders_base/B_unixcoder"),
    "unixcoder_full_bridge":  ("E_unixcoder_bridge",  "full",   64, 2e-5, "encoders_bask/E_unixcoder_bridge"),
    "unixcoder_lora_bridge":  ("E_unixcoder_bridge",  "lora",   64, 2e-4, "encoders_bask_lora/E_unixcoder_bridge"),
    "unixcoder_frozen_bridge":("E_unixcoder_bridge",  "frozen", 64, 2e-4, "encoders_bask_frozen/E_unixcoder_bridge"),
    "starcoder_lora_base":    ("G0_starcoder_base",   "lora",   64, 2e-4, "llm_starcoder/G0_starcoder_base"),
    "starcoder_lora_bridge":  ("G_starcoder_bridge",  "lora",   64, 2e-4, "llm_starcoder/G_starcoder_bridge"),
    "starcoder2_lora_base":   ("J0_starcoder2_base",  "lora",   32, 2e-4, "llm_starcoder2/J0_starcoder2_base"),
    "starcoder2_lora_bridge": ("J_starcoder2_bridge", "lora",   32, 2e-4, "llm_starcoder2/J_starcoder2_bridge"),
    "qwen_lora_base":         ("I0_qwen_base",        "lora",   32, 2e-4, "llm_qwen/I0_qwen_base"),
    "qwen_lora_bridge":       ("I_qwen_bridge",       "lora",   32, 2e-4, "llm_qwen/I_qwen_bridge"),
    "deepseek_lora_base":     ("H0_deepseek_base",    "lora",   16, 2e-4, "llm_deepseek/H0_deepseek_base"),
    "deepseek_lora_bridge":   ("H_deepseek_bridge",   "lora",   16, 2e-4, "llm_deepseek/H_deepseek_bridge"),
}

METRIC_KEYS = ("VD-S_FNR", "AUC", "F1", "P-C")


def log(*a):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), *a, flush=True)


def build(name, device):
    """Build a model exactly as primevul_phase1_auc.train_system_auc does for this configuration."""
    from transformers import AutoTokenizer
    import primevul_phase1 as P
    import primevul_phase1_auc as A
    system, mode, bs, lr, ck = CONFIGS[name]
    backbone_key, use_graph = A.SYSTEMS_CONFIG[system]
    backbone_name, d_model = P.BACKBONE_CONFIGS[backbone_key]
    is_decoder = backbone_key in P.DECODER_BACKBONES
    P.set_seed(P.SEED)
    tok = AutoTokenizer.from_pretrained(backbone_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = P.VulDetector(backbone_name, d_model, use_graph=use_graph,
                          fixed_len_pad=use_graph, use_structural=use_graph,
                          backbone_mode=mode, structural_only=False,
                          is_decoder=is_decoder).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = model.count_trainable()
    n_bridge = sum(p.numel() for n, p in model.named_parameters()
                   if n.startswith(("graph_ca2.", "struct_proj.")))
    n_cls = sum(p.numel() for p in model.classifier.parameters())
    info = {"system": system, "backbone": backbone_key, "backbone_id": backbone_name,
            "backbone_mode": mode, "use_bridge": use_graph, "batch_size": bs, "lr": lr,
            "checkpoint": ck, "params_total": n_total, "params_trainable": n_train,
            "params_trainable_pct": 100 * n_train / n_total,
            "params_bridge": n_bridge, "params_classifier": n_cls,
            "bridge_pct_of_backbone": 100 * n_bridge / (n_total - n_bridge - n_cls)}
    return model, tok, use_graph, info


def predict(model, tok, use_graph, split, device, batch_size, retriever):
    """Return (labels, probs, dataset, data_seconds, gpu_seconds) for one split."""
    import torch
    from torch.utils.data import DataLoader
    import primevul_phase1 as P
    ds = P.PrimeVulDataset(str(SPLITS[split]), tok, retriever if use_graph else None, use_graph)
    pad_id = tok.pad_token_id or 0
    ldr = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                     collate_fn=lambda b: P.collate(b, pad_id))
    labels, probs = [], []
    data_s = gpu_s = 0.0
    it = iter(ldr)
    with torch.no_grad():
        while True:
            t0 = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                break
            t1 = time.perf_counter()
            kw = {}
            if use_graph:
                kw = {"C_AQ": batch["C_AQ"].to(device), "graph_mask": batch["graph_mask"].to(device),
                      "struct_feats": batch["struct_feats"].to(device)}
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), **kw)
            probs.extend(torch.softmax(logits.float(), -1)[:, 1].cpu().tolist())
            labels.extend(batch["labels"].tolist())
            torch.cuda.synchronize()
            data_s += t1 - t0
            gpu_s += time.perf_counter() - t1
    return labels, probs, ds, data_s, gpu_s
