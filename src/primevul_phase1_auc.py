"""
primevul_phase1_auc.py
======================
Phase 1 with AUC-optimising pairwise ranking loss instead of cross-entropy.

Motivation:
  VD-S at 0.5% FPR measures the top of the ROC curve - a ranking problem.
  Cross-entropy optimises classification, not ranking, which is why Bridge
  improves P-C (paired discrimination) but degrades VD-S (global ranking).
  A pairwise loss directly optimises ranking and should fix VD-S without
  losing the P-C improvement.

Loss design:
  Combined objective:
    L = (1 - alpha) * CE(logits, labels)             # classification term
      + alpha       * PairwiseHinge(scores_pos, scores_neg, hard_negs)

  PairwiseHinge: for each (vuln, benign) pair in batch:
    loss_ij = log(1 + exp(-(score_vuln_i - score_benign_j) / tau))
    
  Hard negative mining: for each vulnerable function in the batch,
    use the top-K benign functions by predicted score as negatives.
    This focuses the loss on the hardest examples - the benign functions
    that the model currently scores closest to vulnerable.

Hyperparameters:
  alpha       = 0.7    weight on ranking term (vs classification)
  tau         = 0.5    temperature for sigmoid scaling
  hard_neg_k  = 8      number of hard negatives per positive

Run (from src/):
    python primevul_phase1_auc.py \\
        --train  ../data/primevul/primevul_train.jsonl \\
        --valid  ../data/primevul/primevul_valid.jsonl \\
        --test   ../data/primevul/primevul_test.jsonl \\
        --graphs ../data/primevul_graphs \\
        --out    ../checkpoints/main \\
        --systems A_codebert,D_codebert_bridge \\
        --fixed_len_pad --use_structural --device cuda
"""

import json
import re
import sys
import random
import argparse
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

sys.path.insert(0, ".")

# Reuse all components from primevul_phase1
from primevul_phase1 import (
    BLOCK_SIZE, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, LEARNING_RATE,
    WARMUP_STEPS, MAX_GRAD_NORM, EPOCHS, SEED,
    D_G, MAX_K, LORA_RANK, LORA_ALPHA, BACKBONE_CONFIGS, DECODER_BACKBONES,
    set_seed, LoRALinear, GraphCA2, VulDetector,
    PerProjectRetriever, PrimeVulDataset, collate,
    compute_metrics, find_best_threshold, eval_loss,
    evaluate_full, roc_auc_score_safe,
    N_STRUCT, set_active_feature_group, n_active_features,
)

# AUC-loss specific hyperparameters
ALPHA          = 0.7      # weight on ranking term
TAU            = 0.5      # temperature for ranking sigmoid
HARD_NEG_K     = 8        # hard negatives per positive in batch
MIN_VULN_PER_BATCH = 2    # ensure at least this many vulnerable in each batch

# Pair-aware loss: number of commit-paired (vuln, benign) examples placed
# at fixed positions [0, 2*N_PAIR_PER_BATCH) of each batch when --pair_aware.
N_PAIR_PER_BATCH = 2
PAIR_MARGIN      = 0.5    # margin in the hinge: max(0, margin - (s_v - s_b))
LAMBDA_PAIR      = 0.3    # weight on pair loss in the combined objective


# ---------------------------------------------------------------------------
# AUC / Pairwise Ranking Loss
# ---------------------------------------------------------------------------

class PairwiseRankingLoss(nn.Module):
    """
    AUC-optimising pairwise hinge loss with hard negative mining.

    For each vulnerable function in the batch, find the top-K benign
    functions with the highest current scores (the hardest negatives)
    and apply a smooth hinge loss to push the vulnerable score above them.

    This directly optimises the ROC curve at the top end (low FPR),
    which is what VD-S measures.
    """

    def __init__(self, tau: float = TAU, hard_neg_k: int = HARD_NEG_K):
        super().__init__()
        self.tau         = tau
        self.hard_neg_k  = hard_neg_k

    def forward(self, scores: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        scores : (B,) - predicted vulnerability score (logit for class 1)
        labels : (B,) - 0 or 1
        """
        vuln_mask   = labels == 1
        benign_mask = labels == 0

        n_vuln   = vuln_mask.sum().item()
        n_benign = benign_mask.sum().item()

        if n_vuln == 0 or n_benign == 0:
            # No pair possible in this batch - return zero loss
            return scores.sum() * 0.0

        vuln_scores   = scores[vuln_mask]    # (n_vuln,)
        benign_scores = scores[benign_mask]  # (n_benign,)

        # Hard negative mining: top-K benign scores (most confused)
        k = min(self.hard_neg_k, n_benign)
        hard_neg_scores, _ = torch.topk(benign_scores, k=k, dim=0)
        # Shape: (k,)

        # Compute pairwise differences: (n_vuln, k)
        diff = vuln_scores.unsqueeze(1) - hard_neg_scores.unsqueeze(0)

        # Smooth hinge: log(1 + exp(-diff/tau))
        # This is equivalent to soft-margin pairwise loss
        loss = F.softplus(-diff / self.tau).mean()

        return loss


class CombinedAUCLoss(nn.Module):
    """
    Combined loss: alpha * pairwise_ranking + (1 - alpha) * cross_entropy
    """

    def __init__(self, alpha: float = ALPHA, tau: float = TAU,
                 hard_neg_k: int = HARD_NEG_K):
        super().__init__()
        self.alpha = alpha
        self.ranking_loss = PairwiseRankingLoss(tau, hard_neg_k)
        self.ce_loss      = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        # CE on logits
        ce = self.ce_loss(logits, labels)

        # Ranking on positive class score
        scores = logits[:, 1] - logits[:, 0]  # vulnerable - benign logit
        ranking = self.ranking_loss(scores, labels)

        return self.alpha * ranking + (1 - self.alpha) * ce


# ---------------------------------------------------------------------------
# Pair-aware loss
#
# A retrieval analysis showed that 15% of paired (vuln, benign) test examples
# share Jaccard >= 0.9 retrieval, and even outside that tail the meaningful
# tokens are shared - so the bridge has limited pair-level signal. This loss
# adds direct supervision: for each (vuln, benign) commit pair in the batch,
# require score(vuln) > score(benign) + margin under a hinge.
#
# The accompanying PairBalancedSampler places paired examples at consecutive
# fixed positions [0, 2*N_PAIRS) so the loss can locate them.
# ---------------------------------------------------------------------------

class PairBalancedSampler(torch.utils.data.Sampler):
    """
    Each batch has:
      * 2*N_PAIR_PER_BATCH leading slots: pairs from same commit_id,
        interleaved as (v0, b0, v1, b1, ...).
      * Remaining slots: random benign (with MIN_VULN_PER_BATCH already
        guaranteed by the pair slots).
    Pair positions are NOT shuffled, so the loss can index into them.
    """

    def __init__(self, dataset, batch_size: int = TRAIN_BATCH_SIZE,
                 n_pairs_per_batch: int = N_PAIR_PER_BATCH):
        self.dataset    = dataset
        self.batch_size = batch_size
        self.n_pairs    = n_pairs_per_batch

        # dataset.pairs is List[(vuln_idx, benign_idx)] keyed by commit_id
        if not dataset.pairs:
            raise ValueError(
                "PairBalancedSampler requires dataset.pairs (commit_id-keyed). "
                "Got an empty list - check that the training jsonl has commit_ids."
            )
        self.pairs = list(dataset.pairs)
        self.benign_indices = [i for i, r in enumerate(dataset.rows)
                               if r["target"] == 0]

        n_batches = len(dataset) // batch_size
        self.n_batches = n_batches
        print(f"  PairBalancedSampler: {len(self.pairs):,} commit pairs, "
              f"{len(self.benign_indices):,} benign, "
              f"{n_batches:,} batches/epoch")
        print(f"  Each batch: {n_pairs_per_batch} commit-paired (v,b) at "
              f"positions [0, {2*n_pairs_per_batch}) + "
              f"{batch_size - 2*n_pairs_per_batch} random benign")

    def __iter__(self):
        random.seed(SEED + 2000)
        pair_pool   = self.pairs.copy()
        benign_pool = self.benign_indices.copy()
        random.shuffle(pair_pool); random.shuffle(benign_pool)

        for _ in range(self.n_batches):
            # Pair positions first - DO NOT shuffle these
            batch = []
            for _ in range(self.n_pairs):
                if not pair_pool:
                    pair_pool = self.pairs.copy()
                    random.shuffle(pair_pool)
                v, b = pair_pool.pop()
                batch.append(v); batch.append(b)
            # Fill remaining slots with benign
            n_fill = self.batch_size - 2 * self.n_pairs
            for _ in range(n_fill):
                if not benign_pool:
                    benign_pool = self.benign_indices.copy()
                    random.shuffle(benign_pool)
                batch.append(benign_pool.pop())
            for idx in batch:
                yield idx

    def __len__(self):
        return self.n_batches * self.batch_size


class PairMarginLoss(nn.Module):
    """Hinge loss on commit-paired examples at fixed batch positions."""

    def __init__(self, margin: float = PAIR_MARGIN,
                 n_pairs_per_batch: int = N_PAIR_PER_BATCH):
        super().__init__()
        self.margin = margin
        self.n_pairs = n_pairs_per_batch

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        # scores: (B,) - score is logit_vuln - logit_benign per example
        # PairBalancedSampler guarantees first 2*N slots are interleaved (v,b)
        n = self.n_pairs
        if scores.shape[0] < 2 * n:
            return scores.sum() * 0.0
        pair_scores = scores[:2 * n].view(n, 2)
        v_score = pair_scores[:, 0]  # vuln
        b_score = pair_scores[:, 1]  # benign
        diff    = v_score - b_score
        loss    = F.relu(self.margin - diff).mean()
        return loss


class CombinedPairAwareLoss(nn.Module):
    """alpha*ranking + (1-alpha)*CE + lambda_pair*pair_margin."""

    def __init__(self, alpha: float = ALPHA, tau: float = TAU,
                 hard_neg_k: int = HARD_NEG_K,
                 lambda_pair: float = LAMBDA_PAIR,
                 margin: float = PAIR_MARGIN,
                 n_pairs_per_batch: int = N_PAIR_PER_BATCH):
        super().__init__()
        self.alpha        = alpha
        self.lambda_pair  = lambda_pair
        self.ranking_loss = PairwiseRankingLoss(tau, hard_neg_k)
        self.ce_loss      = nn.CrossEntropyLoss()
        self.pair_loss    = PairMarginLoss(margin, n_pairs_per_batch)

    def forward(self, logits, labels):
        ce      = self.ce_loss(logits, labels)
        scores  = logits[:, 1] - logits[:, 0]
        ranking = self.ranking_loss(scores, labels)
        pair    = self.pair_loss(scores)
        return (self.alpha * ranking
                + (1 - self.alpha) * ce
                + self.lambda_pair * pair)


# ---------------------------------------------------------------------------
# Class-balanced batch sampler - ensures vulnerable functions in every batch
# ---------------------------------------------------------------------------

class BalancedSampler(torch.utils.data.Sampler):
    """
    Each batch contains MIN_VULN_PER_BATCH vulnerable + remaining benign.
    Without this, with 35:1 imbalance many batches contain ZERO vulnerable
    functions, which makes pairwise loss meaningless for those batches.
    """

    def __init__(self, dataset, batch_size: int = TRAIN_BATCH_SIZE,
                 min_vuln_per_batch: int = MIN_VULN_PER_BATCH):
        self.dataset = dataset
        self.batch_size = batch_size
        self.min_vuln = min_vuln_per_batch

        self.vuln_indices   = [i for i, r in enumerate(dataset.rows)
                               if r["target"] == 1]
        self.benign_indices = [i for i, r in enumerate(dataset.rows)
                               if r["target"] == 0]

        n_batches = len(dataset) // batch_size
        self.n_batches = n_batches

        print(f"  BalancedSampler: {len(self.vuln_indices):,} vuln, "
              f"{len(self.benign_indices):,} benign, "
              f"{n_batches:,} batches/epoch")
        print(f"  Each batch: {min_vuln_per_batch} vuln "
              f"+ {batch_size - min_vuln_per_batch} benign")

    def __iter__(self):
        random.seed(SEED + 1000)
        # Shuffle and create batches with guaranteed min_vuln per batch
        vuln_pool = self.vuln_indices.copy()
        benign_pool = self.benign_indices.copy()
        random.shuffle(vuln_pool)
        random.shuffle(benign_pool)

        # Use vuln examples with replacement - they will appear ~35x per epoch
        # to balance against benign which appears once
        for batch_idx in range(self.n_batches):
            batch = []
            # Pick min_vuln vulnerable
            for _ in range(self.min_vuln):
                if not vuln_pool:
                    vuln_pool = self.vuln_indices.copy()
                    random.shuffle(vuln_pool)
                batch.append(vuln_pool.pop())
            # Fill the rest with benign
            for _ in range(self.batch_size - self.min_vuln):
                if not benign_pool:
                    benign_pool = self.benign_indices.copy()
                    random.shuffle(benign_pool)
                batch.append(benign_pool.pop())

            random.shuffle(batch)
            for idx in batch:
                yield idx

    def __len__(self):
        return self.n_batches * self.batch_size


# ---------------------------------------------------------------------------
# Training loop with AUC loss
# ---------------------------------------------------------------------------

def train_system_auc(name, backbone_key, use_graph,
                      train_path, valid_path, test_path,
                      retriever, out_dir, device, epochs=EPOCHS,
                      max_train=None, fixed_len_pad=False,
                      use_structural=False, pair_aware=False,
                      backbone_mode="full", lr=LEARNING_RATE,
                      structural_only=False, batch_size=TRAIN_BATCH_SIZE):
    set_seed(SEED)
    backbone_name, d_model = BACKBONE_CONFIGS[backbone_key]
    is_decoder = backbone_key in DECODER_BACKBONES

    print(f"\n{'='*64}")
    print(f"System: {name} (AUC-LOSS)")
    print(f"  Backbone:   {backbone_name}")
    print(f"  is_decoder: {is_decoder}")
    print(f"  Use graph:  {use_graph}")
    print(f"  Loss:       alpha*PairwiseRanking + (1-alpha)*CE")
    print(f"  alpha:      {ALPHA}")
    print(f"  tau:        {TAU}")
    print(f"  hard_neg_k: {HARD_NEG_K}")
    print(f"  Batch:      {batch_size} ({MIN_VULN_PER_BATCH} vuln min)")
    print(f"  fixed_len_pad:             {fixed_len_pad}")
    print(f"  use_structural:            {use_structural}")
    print(f"  pair_aware:                {pair_aware}")
    print(f"  backbone_mode  (cost):     {backbone_mode}")
    print(f"  structural_only:           {structural_only}")
    print(f"  active features:           {n_active_features()}/{N_STRUCT}")
    print(f"{'='*64}")

    out_path = Path(out_dir) / name
    out_path.mkdir(parents=True, exist_ok=True)

    tok    = AutoTokenizer.from_pretrained(backbone_name)
    if tok.pad_token_id is None:
        # Decoder LLMs often lack a pad token; reuse EOS for padding.
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id or 0

    print("Building datasets ...")
    ret = retriever if use_graph else None
    train_ds = PrimeVulDataset(train_path, tok, ret, use_graph,
                               max_examples=max_train)
    valid_ds = PrimeVulDataset(valid_path, tok, ret, use_graph)
    test_ds  = PrimeVulDataset(test_path,  tok, ret, use_graph)

    col = lambda b: collate(b, pad_id)

    # Sampler: pair-aware places (vuln, benign) commit pairs at fixed
    # positions so the pair-margin loss can index into them. Otherwise
    # the legacy BalancedSampler is used.
    if pair_aware:
        sampler = PairBalancedSampler(train_ds, batch_size,
                                      N_PAIR_PER_BATCH)
    else:
        sampler = BalancedSampler(train_ds, batch_size,
                                  MIN_VULN_PER_BATCH)
    train_ldr = DataLoader(train_ds,
                           batch_size=batch_size,
                           sampler=sampler,
                           collate_fn=col, num_workers=0,
                           drop_last=True)
    valid_ldr = DataLoader(valid_ds, batch_size=EVAL_BATCH_SIZE,
                           shuffle=False, collate_fn=col, num_workers=0)

    model = VulDetector(backbone_name, d_model, use_graph=use_graph,
                        fixed_len_pad=fixed_len_pad,
                        use_structural=use_structural,
                        backbone_mode=backbone_mode,
                        structural_only=structural_only,
                        is_decoder=is_decoder).to(device)
    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = model.count_trainable()
    print(f"  Params: {n_trainable:,} trainable / {n_total:,} total "
          f"({100*n_trainable/n_total:.2f}%)")

    # Loss: pair-aware adds a hinge on commit-paired examples placed at
    # fixed batch positions by PairBalancedSampler.
    if pair_aware:
        criterion = CombinedPairAwareLoss(
            alpha=ALPHA, tau=TAU, hard_neg_k=HARD_NEG_K,
            lambda_pair=LAMBDA_PAIR, margin=PAIR_MARGIN,
            n_pairs_per_batch=N_PAIR_PER_BATCH,
        )
    else:
        criterion = CombinedAUCLoss(alpha=ALPHA, tau=TAU,
                                     hard_neg_k=HARD_NEG_K)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )
    print(f"  learning rate: {lr:.1e}")
    total_steps = epochs * len(train_ldr)
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    best_val_loss  = float("inf")
    best_val_auc   = 0.0
    best_val_fnr   = float("inf")
    best_fnr_epoch = 0
    best_auc_epoch = 0
    log            = []
    global_step    = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0; n_batches = 0
        epoch_ce   = 0.0; epoch_rank = 0.0

        for batch in tqdm(train_ldr, desc=f"Ep {epoch} [{name}-auc]"):
            ii = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            lb = batch["labels"].to(device)
            cq = batch["C_AQ"].to(device) if use_graph else None
            gm = batch["graph_mask"].to(device) if use_graph else None
            sf = batch["struct_feats"].to(device) \
                 if use_structural and "struct_feats" in batch else None

            logits = model(ii, am, C_AQ=cq, graph_mask=gm, struct_feats=sf)
            loss   = criterion(logits, lb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            global_step += 1
            epoch_loss  += loss.item(); n_batches += 1

            if global_step % 200 == 0:
                avg = epoch_loss / n_batches
                print(f"  step {global_step:5d} | loss {avg:.4f} | "
                      f"lr {scheduler.get_last_lr()[0]:.2e}")

        # Validate after each epoch
        val_labels, val_probs = evaluate_full(model, valid_ds, device,
                                              use_graph)
        best_thresh, val_f1 = find_best_threshold(val_labels, val_probs)
        val_metrics = compute_metrics(val_labels, val_probs,
                                       threshold=best_thresh)

        val_auc = val_metrics["AUC"]
        val_fnr = val_metrics["VD-S_FNR"]

        print(f"\nEpoch {epoch} | avg loss: {epoch_loss/n_batches:.4f} | "
              f"val AUC: {val_auc:.4f} | "
              f"val F1: {val_f1:.4f} | "
              f"val VD-S FNR: {val_fnr:.2f}%")

        # Save best by VD-S FNR (paper's headline metric - lower is better).
        # Earlier runs selected by AUC and saved overconfident-but-wrong-at-tip
        # checkpoints. Keep best_auc.pt as a side-channel for comparison.
        if val_fnr < best_val_fnr:
            best_val_fnr = val_fnr
            best_fnr_epoch = epoch
            torch.save(model.state_dict(), out_path / "best.pt")
            json.dump({"best_threshold": best_thresh,
                       "selected_by": "val_VD-S_FNR",
                       "epoch": epoch,
                       "val_fnr": val_fnr,
                       "val_auc": val_auc},
                      open(out_path / "threshold.json", "w"))
            print(f"  [checkpoint] saved (FNR={best_val_fnr:.2f}% ep{epoch})")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_auc_epoch = epoch
            torch.save(model.state_dict(), out_path / "best_auc.pt")

        log.append({"epoch": epoch, "val": val_metrics})
        model.train()

    print(f"\n  best by FNR: epoch {best_fnr_epoch} (val FNR={best_val_fnr:.2f}%)")
    print(f"  best by AUC: epoch {best_auc_epoch} (val AUC={best_val_auc:.4f})")

    # Test evaluation
    print(f"\nTest evaluation: {name}")
    model.load_state_dict(torch.load(out_path / "best.pt",
                                     map_location=device, weights_only=True))

    val_labels, val_probs = evaluate_full(model, valid_ds, device, use_graph)
    best_thresh, _ = find_best_threshold(val_labels, val_probs)

    test_labels, test_probs = evaluate_full(model, test_ds, device, use_graph)
    test_m = compute_metrics(test_labels, test_probs,
                              threshold=best_thresh, dataset=test_ds)
    test_m_05 = compute_metrics(test_labels, test_probs,
                                 threshold=0.5, dataset=test_ds)

    print(f"\n  Results for {name} (AUC LOSS):")
    print(f"  --- At threshold=0.50 (PAPER COMPARISON) ---")
    print(f"  F1:          {test_m_05['F1']:.4f}")
    print(f"  AUC:         {test_m_05['AUC']:.4f}")
    print(f"  VD-S (FNR):  {test_m_05['VD-S_FNR']:.2f}%  (LOWER=BETTER)")
    print(f"  VD-S (TPR):  {test_m_05['VD-S_TPR']:.2f}%  (HIGHER=BETTER)")
    if "P-C" in test_m_05:
        print(f"  P-C: {test_m_05['P-C']:.2f}%  P-V: {test_m_05['P-V']:.2f}%  "
              f"P-B: {test_m_05['P-B']:.2f}%  P-R: {test_m_05['P-R']:.2f}%")
    print(f"  --- At threshold={best_thresh:.2f} (CALIBRATED) ---")
    print(f"  F1:          {test_m['F1']:.4f}")
    if "P-C" in test_m:
        print(f"  P-C: {test_m['P-C']:.2f}%")

    test_m["at_threshold_05"] = test_m_05

    json.dump({"system": name, "test": test_m,
               "best_threshold": best_thresh, "log": log,
               "loss": "combined_auc",
               "alpha": ALPHA, "tau": TAU},
              open(out_path / "results.json", "w"), indent=2)

    return test_m


# ---------------------------------------------------------------------------
# Main - same systems as Phase 1 but with AUC loss
# ---------------------------------------------------------------------------

SYSTEMS_CONFIG = {
    "A_codebert":              ("codebert",      False),
    "B_unixcoder":             ("unixcoder",     False),
    "C_graphcodebert":         ("graphcodebert", False),
    "D_codebert_bridge":       ("codebert",      True),
    "E_unixcoder_bridge":      ("unixcoder",     True),
    "F_graphcodebert_bridge":  ("graphcodebert", True),
    # Decoder LLM backbones - run with --backbone_mode lora. Each has a
    # no-bridge baseline (use_graph=False) so the bridge's contribution on
    # the LLM can be measured, mirroring the encoder A/B/C baselines.
    "G0_starcoder_base":       ("starcoder",     False),
    "G_starcoder_bridge":      ("starcoder",     True),
    "H0_deepseek_base":        ("deepseek",      False),
    "H_deepseek_bridge":       ("deepseek",      True),
    "I0_qwen_base":            ("qwen",          False),
    "I_qwen_bridge":           ("qwen",          True),
    "J0_starcoder2_base":      ("starcoder2",    False),
    "J_starcoder2_bridge":     ("starcoder2",    True),
}


def main(train_path, valid_path, test_path, graphs_dir,
         out_dir, device_str, systems_filter, epochs,
         max_train=None, fixed_len_pad=False, use_structural=False,
         pair_aware=False, backbone_mode="full", lr=LEARNING_RATE,
         structural_only=False, drop_feature_group=None,
         batch_size=TRAIN_BATCH_SIZE):
    device = torch.device(device_str)

    # Configure active structural features BEFORE building datasets/models,
    # since both the retriever-side extractor and the model projector read
    # the module-level active-feature selection.
    set_active_feature_group(drop_feature_group)
    if drop_feature_group:
        print(f"  [feature-group ablation] dropped group: {drop_feature_group} "
              f"-> {n_active_features()}/{N_STRUCT} features active")

    print("Initialising per-project retriever ...")
    retriever = PerProjectRetriever(graphs_dir, fixed_max_k=fixed_len_pad) \
        if Path(graphs_dir).exists() else None
    if fixed_len_pad:
        print("  fixed_max_k retrieval + learnable [PAD] in CA2")
    if use_structural:
        print("  security-relevant structural features as bridge tokens")
    if pair_aware:
        print(f"  pair-aware margin loss (lambda={LAMBDA_PAIR}, "
              f"margin={PAIR_MARGIN}, n_pairs/batch={N_PAIR_PER_BATCH})")

    if systems_filter == "baselines":
        run = {k: v for k, v in SYSTEMS_CONFIG.items() if not v[1]}
    elif systems_filter == "bridge":
        run = {k: v for k, v in SYSTEMS_CONFIG.items() if v[1]}
    elif systems_filter == "all":
        run = SYSTEMS_CONFIG
    else:
        wanted = [s.strip() for s in systems_filter.split(",")]
        run = {k: v for k, v in SYSTEMS_CONFIG.items()
               if any(w in k for w in wanted)}

    print(f"\nRunning {len(run)} systems with AUC loss: {list(run.keys())}")

    all_results = {}
    for sys_name, (backbone_key, use_graph) in run.items():
        result = train_system_auc(
            sys_name, backbone_key, use_graph,
            train_path, valid_path, test_path,
            retriever, out_dir, device, epochs=epochs,
            max_train=max_train, fixed_len_pad=fixed_len_pad,
            use_structural=use_structural, pair_aware=pair_aware,
            backbone_mode=backbone_mode, lr=lr,
            structural_only=structural_only, batch_size=batch_size,
        )
        all_results[sys_name] = result

    # Comparison table
    print(f"\n{'='*80}")
    print("PHASE 1 AUC-LOSS RESULTS  (paper convention: VD-S = FNR, LOWER=BETTER)")
    print(f"{'='*80}")
    print(f"{'System':<32} {'F1':>7} {'AUC':>7} "
          f"{'VD-S FNR%':>10} {'VD-S TPR%':>10} {'P-C%':>7}")
    print("-"*80)

    paper_ref = {
        "CodeBERT (paper)":   {"F1": 0.209, "VD-S_FNR": 88.78, "P-C": 1.77},
        "UnixCoder (paper)":  {"F1": 0.214, "VD-S_FNR": 89.21, "P-C": 1.60},
    }
    for ref_name, ref_m in paper_ref.items():
        print(f"  {ref_name:<30} {ref_m['F1']:>7.3f}     -   "
              f"{ref_m['VD-S_FNR']:>10.2f}         -   "
              f"{ref_m['P-C']:>7.2f}")
    print("-"*80)

    for s, m in all_results.items():
        pc = m.get('P-C', 0)
        tpr = m.get('VD-S_TPR', 100 - m.get('VD-S_FNR', 100))
        marker = " <- better than PrimeVul CodeBERT" if m.get("VD-S_FNR", 100) < 88.78 else ""
        print(f"  {s:<30} {m['F1']:>7.3f} {m['AUC']:>7.3f} "
              f"{m['VD-S_FNR']:>10.2f} {tpr:>10.2f} "
              f"{pc:>7.2f}{marker}")

    json.dump(all_results,
              open(Path(out_dir) / "phase1_auc_results.json", "w"), indent=2)
    print(f"\nAll results: {Path(out_dir) / 'phase1_auc_results.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train",     required=True)
    ap.add_argument("--valid",     required=True)
    ap.add_argument("--test",      required=True)
    ap.add_argument("--graphs",    default="./datasets/primevul_graphs")
    ap.add_argument("--out",       default="./checkpoints/primevul_phase1_auc")
    ap.add_argument("--systems",   default="A_codebert,D_codebert_bridge",
                    help="baselines | bridge | all | comma-separated names")
    ap.add_argument("--epochs",    type=int, default=EPOCHS)
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--device",    default="cuda"
                    if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fixed_len_pad", action="store_true",
                    help="Pad C^AQ to MAX_K with a learnable [PAD] "
                         "token in CA2. Removes the count-as-label shortcut "
                         "a retrieval analysis showed (vulns saturate MAX_K 60% vs 36%).")
    ap.add_argument("--use_structural", action="store_true",
                    help="Add per-function security-relevant "
                         "structural features as additional bridge tokens. "
                         "Provides intrinsic pair-level discrimination that "
                         "project-KB tokens cannot supply.")
    ap.add_argument("--pair_aware", action="store_true",
                    help="Place (vuln, benign) commit pairs at "
                         "fixed batch positions and add a margin hinge "
                         "loss requiring score(vuln) > score(benign) + "
                         "margin. Direct pair-level supervision targeting "
                         "the P-C metric.")
    ap.add_argument("--backbone_mode", default="full",
                    choices=["full", "frozen", "lora"],
                    help="Cost-efficiency study: 'full' fine-tunes the whole "
                         "backbone (~124M params), 'frozen' trains only the "
                         "bridge+classifier (~0.2%%), 'lora' injects LoRA "
                         "adapters into the backbone (~2%%).")
    ap.add_argument("--lr", type=float, default=LEARNING_RATE,
                    help="Learning rate. Default 2e-5 is tuned for full "
                         "fine-tuning; LoRA typically needs ~1e-4 to 5e-4.")
    ap.add_argument("--structural_only", action="store_true",
                    help="Per-backbone signal ablation: bridge attends ONLY "
                         "to structural-feature tokens, ignoring KG nodes.")
    ap.add_argument("--drop_feature_group", default=None,
                    choices=["complexity", "security", "detail"],
                    help="Feature-group ablation: remove one structural "
                         "feature group to measure its contribution.")
    ap.add_argument("--batch_size", type=int, default=TRAIN_BATCH_SIZE,
                    help="Train batch size. Default 64; reduce for large "
                         "decoder LLMs (e.g. 16 for a 7B model) to fit memory.")
    args = ap.parse_args()

    main(args.train, args.valid, args.test, args.graphs,
         args.out, args.device, args.systems, args.epochs,
         max_train=args.max_train, fixed_len_pad=args.fixed_len_pad,
         use_structural=args.use_structural, pair_aware=args.pair_aware,
         backbone_mode=args.backbone_mode, lr=args.lr,
         structural_only=args.structural_only,
         drop_feature_group=args.drop_feature_group,
         batch_size=args.batch_size)
