"""
primevul_phase1.py
==================
Phase 1: Exact replication of PrimeVul paper (Ding et al., ICSE 2025)
         + Bridge Attention variants.

Matches the paper's exact training setup from their GitHub:
  --epoch 10
  --block_size 512
  --train_batch_size 64
  --eval_batch_size 128
  --learning_rate 2e-5
  --warmup_steps 1000
  --max_grad_norm 1.0
  --seed 123456
  Loss: plain cross-entropy (no focal, no class weights)

Systems trained:
  Paper baselines (exact replication):
    A_codebert          : CodeBERT fine-tune (their CB row, Train=PV Test=PV)
    B_unixcoder         : UnixCoder fine-tune (their UC row)
    C_graphcodebert     : GraphCodeBERT fine-tune (not in paper, our addition)

  Bridge Attention variants (our contribution):
    D_codebert_bridge   : CodeBERT + per-project graph CA2
    E_unixcoder_bridge  : UnixCoder + per-project graph CA2
    F_graphcodebert_bridge : GraphCodeBERT + per-project graph CA2

Reported metrics (matching paper):
  VD-S : False Negative Rate @ FPR <= 0.5%  (LOWER IS BETTER - paper convention)
  F1   : at optimal threshold (calibrated on validation set)
  AUC  : ROC area under curve
  P-C  : Pair-wise Correct (both elements of a vulnerable/patched pair correct)
  P-V  : Pair-wise Vulnerable (both predicted vulnerable - bad)
  P-B  : Pair-wise Benign (both predicted benign - bad)

Run (from src/; plain cross-entropy baselines):
    python primevul_phase1.py \\
        --train  ../data/primevul/primevul_train.jsonl \\
        --valid  ../data/primevul/primevul_valid.jsonl \\
        --test   ../data/primevul/primevul_test.jsonl \\
        --graphs ../data/primevul_graphs \\
        --out    ../checkpoints/phase1 \\
        --systems baselines --device cuda
"""

import json
import re
import sys
import random
import argparse
from pathlib import Path
from typing import Optional

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

# ---------------------------------------------------------------------------
# EXACT paper hyperparameters - DO NOT CHANGE
# ---------------------------------------------------------------------------

BLOCK_SIZE       = 512     # paper: --block_size 512
TRAIN_BATCH_SIZE = 64      # paper: --train_batch_size 64
EVAL_BATCH_SIZE  = 128     # paper: --eval_batch_size 128
LEARNING_RATE    = 2e-5    # paper: --learning_rate 2e-5
WARMUP_STEPS     = 1000    # paper: --warmup_steps 1000
MAX_GRAD_NORM    = 1.0     # paper: --max_grad_norm 1.0
EPOCHS           = 10      # paper: --epoch 10
SEED             = int(__import__("os").environ.get("PRIMEVUL_SEED", 123456))  # paper: --seed 123456 (override via PRIMEVUL_SEED env var)

# Bridge Attention hyperparameters (our addition)
D_G        = 128
MAX_K      = 32
LORA_RANK  = 16
LORA_ALPHA = 32


# ---------------------------------------------------------------------------
# Security-relevant structural features
#
# A retrieval analysis showed that vuln/benign pairs in PrimeVul have nearly
# identical KB retrievals (15% with Jaccard >= 0.9, and even outside that
# tail the *meaningful* tokens are shared). The discriminating signal lives
# in the function diff - added bounds checks, changed alloc sizes, missing
# null guards, etc. The bridge's project-KB inputs are pair-invariant.
#
# This injects per-function structural features as additional bridge
# tokens. These differ between vuln and benign by construction (because the
# bug fix changes code structure), giving the bridge intrinsic pair-level
# discrimination that the KB tokens cannot provide.
# ---------------------------------------------------------------------------

def _max_brace_depth(src: str) -> int:
    depth, max_d = 0, 0
    for c in src:
        if c == '{':
            depth += 1
            if depth > max_d: max_d = depth
        elif c == '}':
            depth -= 1
    return max_d

def _count_any(src: str, terms) -> int:
    return sum(src.count(t) for t in terms)

# Halstead helpers - distinct/total operators and operands from a function body.
_HALSTEAD_OP_RE = re.compile(
    r'(\+\+|--|<<=|>>=|<=|>=|==|!=|&&|\|\||->|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<|>>|'
    r'[+\-*/%=<>!&|^~?:,;()\[\]{}])'
)
_HALSTEAD_OPERAND_RE = re.compile(r'\b[A-Za-z_]\w*\b|\b\d+\b')

def _halstead(src: str):
    ops      = _HALSTEAD_OP_RE.findall(src)
    operands = _HALSTEAD_OPERAND_RE.findall(src)
    n1 = len(set(ops));       N1 = len(ops)
    n2 = len(set(operands));  N2 = len(operands)
    import math
    vocab = max(n1 + n2, 1)
    length = N1 + N2
    volume = length * math.log2(vocab) if vocab > 1 else 0.0
    return n1, n2, N1, N2, float(volume)


# (name, extractor) - fixed order, defines token positions in C^AQ.
STRUCTURAL_FEATURES = [
    ("loc",           lambda s: s.count('\n') + 1),
    ("cyclomatic",    lambda s: len(re.findall(
                          r'\b(if|for|while|case|&&|\|\|)\b', s))),
    ("max_depth",     _max_brace_depth),
    ("n_returns",     lambda s: len(re.findall(r'\breturn\b', s))),
    ("n_derefs",      lambda s: s.count('->')),
    ("n_arrays",      lambda s: s.count('[')),
    ("n_alloc",       lambda s: _count_any(
                          s, ('malloc(','calloc(','realloc(','alloca('))),
    ("n_free",        lambda s: len(re.findall(r'\bfree\s*\(', s))),
    ("n_dangerous",   lambda s: _count_any(
                          s, ('strcpy(','strcat(','sprintf(','gets(',
                              'memcpy(','memmove(','scanf(','sscanf('))),
    ("n_format",      lambda s: _count_any(
                          s, ('printf(','fprintf(','snprintf('))),
    ("n_size_t",      lambda s: s.count('size_t')),
    ("n_ptr_arith",   lambda s: len(re.findall(r'[\w\)\]]\s*[+\-]\s*\w+\s*[+\-]', s))),
    ("n_null_check",  lambda s: len(re.findall(
                          r'==\s*NULL|!=\s*NULL|==\s*0\b|!=\s*0\b', s))),
    ("n_neg_check",   lambda s: len(re.findall(r'<\s*0\b|>=\s*0\b', s))),
    ("n_overflow_kw", lambda s: _count_any(
                          s, ('SIZE_MAX','INT_MAX','UINT_MAX','overflow'))),
    ("n_user_input",  lambda s: _count_any(
                          s, ('user_buf','copy_from_user','get_user'))),
    # Halstead, coupling and further security patterns
    ("halstead_n1",   lambda s: _halstead(s)[0]),  # distinct operators
    ("halstead_n2",   lambda s: _halstead(s)[1]),  # distinct operands
    ("halstead_N1",   lambda s: _halstead(s)[2]),  # total operators
    ("halstead_N2",   lambda s: _halstead(s)[3]),  # total operands
    ("halstead_vol",  lambda s: _halstead(s)[4]),  # volume
    ("fan_out",       lambda s: len({m for m in re.findall(
                          r'\b([a-zA-Z_]\w*)\s*\(', s)
                          if m not in {
                              'if','for','while','switch','sizeof','typeof',
                              'return','do','else','case'}})),
    ("n_signed_cmp",  lambda s: len(re.findall(
                          r'\bint\s+\w+\s*[<>]=?\s*\w+|'
                          r'\b\w+\s*<\s*(?:strlen|size|len|count)\b', s))),
    ("has_recursion", lambda s: int(bool(re.search(
                          r'(\w+)\s*\([^)]*\)\s*\{[^}]*\b\1\s*\(', s)))),
]
N_STRUCT = len(STRUCTURAL_FEATURES)

# Feature groups for the feature-group ablation. Each name maps to
# the feature names it contains; --drop_feature_group <name> removes that
# group so we can measure how much each contributes.
FEATURE_GROUPS = {
    "complexity": ["loc", "cyclomatic", "max_depth", "n_returns", "halstead_vol"],
    "security":   ["n_alloc", "n_free", "n_dangerous", "n_format",
                   "n_null_check", "n_neg_check", "n_overflow_kw",
                   "n_user_input", "n_signed_cmp", "has_recursion"],
    "detail":     ["n_derefs", "n_arrays", "n_size_t", "n_ptr_arith",
                   "halstead_n1", "halstead_n2", "halstead_N1",
                   "halstead_N2", "fan_out"],
}

# Module-level active-feature selection. Both the dataset (extract_structural)
# and the model (StructuralFeatureProjector) read this, so they must be
# configured BEFORE constructing either. Default: all features active.
_ALL_IDX = list(range(N_STRUCT))
_ACTIVE_FEATURE_IDX = list(_ALL_IDX)

def set_active_feature_group(drop_group: Optional[str] = None):
    """Restrict active structural features by dropping one named group."""
    global _ACTIVE_FEATURE_IDX
    if not drop_group:
        _ACTIVE_FEATURE_IDX = list(_ALL_IDX)
        return
    if drop_group not in FEATURE_GROUPS:
        raise ValueError(f"unknown feature group: {drop_group} "
                         f"(choices: {list(FEATURE_GROUPS)})")
    dropped = set(FEATURE_GROUPS[drop_group])
    _ACTIVE_FEATURE_IDX = [i for i, (name, _) in enumerate(STRUCTURAL_FEATURES)
                           if name not in dropped]

def n_active_features() -> int:
    return len(_ACTIVE_FEATURE_IDX)


def extract_structural(func_src: str) -> torch.Tensor:
    """Return tensor of the currently-active security-relevant features."""
    return torch.tensor(
        [float(STRUCTURAL_FEATURES[i][1](func_src))
         for i in _ACTIVE_FEATURE_IDX],
        dtype=torch.float,
    )


class StructuralFeatureProjector(nn.Module):
    """Map per-feature scalars to D_G-dim 'structural tokens'.

    Each feature gets a learnable D_G vector; the structural token for that
    feature is value * vector (after LayerNorm across features so that loc=200
    and num_returns=2 are commensurate). Features with no signal will see
    their vectors driven toward zero.
    """

    def __init__(self, n_features: Optional[int] = None, d_g: int = D_G):
        super().__init__()
        if n_features is None:
            n_features = n_active_features()
        self.norm   = nn.LayerNorm(n_features)
        self.scales = nn.Parameter(torch.randn(n_features, d_g) * 0.02)
        self.bias   = nn.Parameter(torch.zeros(n_features, d_g))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, N_STRUCT)
        normed = self.norm(features.float())              # (B, N_STRUCT)
        tokens = normed.unsqueeze(-1) * self.scales + self.bias  # (B, N_STRUCT, D_G)
        return tokens

BACKBONE_CONFIGS = {
    "codebert":      ("microsoft/codebert-base",      768),
    "unixcoder":     ("microsoft/unixcoder-base",     768),
    "graphcodebert": ("microsoft/graphcodebert-base", 768),
    # Decoder code LLMs - test whether the bridge generalises beyond encoder
    # code-LMs. Must be run with backbone_mode in {lora,frozen}.
    "starcoder":     ("bigcode/starcoderbase-3b", 2816),
    "deepseek":      ("deepseek-ai/deepseek-coder-6.7b-base", 4096),
    "qwen":          ("Qwen/Qwen2.5-Coder-3B", 2048),
    "starcoder2":    ("bigcode/starcoder2-3b", 3072),
}

# Decoder-only backbones use last-token pooling and are loaded in bf16.
DECODER_BACKBONES = {"starcoder", "deepseek", "qwen", "starcoder2"}

# LoRA target attention submodules differ by architecture family.
LORA_TARGETS = {
    "gpt_bigcode": ["c_attn"],            # StarCoder (v1)
    "llama":       ["q_proj", "v_proj"],  # DeepSeek-Coder
    "qwen2":       ["q_proj", "v_proj"],  # Qwen2.5-Coder
    "starcoder2":  ["q_proj", "v_proj"],  # StarCoder2
    "_default":    ["query", "value"],    # RoBERTa family (CodeBERT/UnixCoder)
}


# ---------------------------------------------------------------------------
# Seed everything (paper uses seed=123456)
# ---------------------------------------------------------------------------

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Bridge Attention CA2 (per-project graph injection)
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    def __init__(self, d_in, d_out, rank=LORA_RANK, alpha=LORA_ALPHA):
        super().__init__()
        import math
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, d_in))
        self.B = nn.Parameter(torch.empty(d_out, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.normal_(self.B, std=0.001)

    def forward(self, x):
        orig = x.dtype
        return (x.float() @ (self.B @ self.A).T * self.scaling).to(orig)


class GraphCA2(nn.Module):
    """CA2 cross-attention: pooled hidden state attends to C^AQ graph tokens.

    Two operating modes:
      fixed_len_pad=False (default, original):
        Variable-length C^AQ; padded positions are masked out of attention.
        Vulnerable code retrieves more tokens than benign (retrieval analysis:
        60% saturate at MAX_K vs 36%), making the *count* itself a label
        shortcut that hurts VD-S FNR at the high-precision tip.

      fixed_len_pad=True:
        Every example presents exactly MAX_K K/V slots. Empty slots are
        filled with a learnable [PAD] embedding before attention, and no
        mask is applied. The attention layer cannot tell short retrievals
        from long ones, so the count shortcut is removed.
    """

    def __init__(self, d_model: int, n_heads: int = 8,
                 fixed_len_pad: bool = False):
        super().__init__()
        import math
        assert d_model % n_heads == 0
        self.d_head        = d_model // n_heads
        self.n_heads       = n_heads
        self.d_model       = d_model
        self.scale         = math.sqrt(self.d_head)
        self.fixed_len_pad = fixed_len_pad
        self.sigma   = nn.Linear(D_G, d_model, bias=False)
        self.q_proj  = LoRALinear(d_model, d_model)
        self.k_proj  = LoRALinear(d_model, d_model)
        self.v_proj  = LoRALinear(d_model, d_model)
        self.o_proj  = LoRALinear(d_model, d_model)
        self.norm    = nn.LayerNorm(d_model)
        self.gate    = nn.Parameter(torch.full((1,), 0.1))
        nn.init.normal_(self.sigma.weight, std=0.02)
        if fixed_len_pad:
            self.pad_token = nn.Parameter(torch.zeros(D_G))
            nn.init.normal_(self.pad_token, std=0.02)

    def _split(self, x, T):
        return x.view(x.shape[0], T, self.n_heads,
                      self.d_head).transpose(1, 2)

    def forward(self, h, C_AQ, graph_mask=None):
        orig  = h.dtype
        h_f   = h.float().unsqueeze(1)

        # Substitute padded slots with a learnable [PAD] embedding
        # and remove the attention mask, so the count of real tokens cannot
        # leak through as a feature.
        if self.fixed_len_pad and graph_mask is not None:
            pad_emb = self.pad_token.to(C_AQ.dtype).view(1, 1, -1)
            C_AQ = torch.where(graph_mask.unsqueeze(-1), pad_emb, C_AQ)
            graph_mask = None

        kv    = self.sigma(C_AQ.float())
        Q = self._split(self.q_proj(h_f), 1)
        K = self._split(self.k_proj(kv), kv.shape[1])
        V = self._split(self.v_proj(kv), kv.shape[1])
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if graph_mask is not None:
            scores = scores.masked_fill(
                graph_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        w   = torch.softmax(scores, dim=-1)
        out = torch.matmul(w, V).squeeze(2)
        out = out.transpose(1, 2).contiguous().view(h.shape[0], self.d_model)
        out = self.o_proj(out)
        return self.norm(h.float() + torch.tanh(self.gate) * out).to(orig)


# ---------------------------------------------------------------------------
# Detection model (matches paper architecture)
# ---------------------------------------------------------------------------

class VulDetector(nn.Module):
    """
    Paper architecture: backbone encoder + mean pool + linear classifier.
    Optional: graph CA2 injected between pooling and classifier.
    """

    def __init__(self, backbone_name: str, d_model: int,
                 use_graph: bool = False, dropout: float = 0.1,
                 fixed_len_pad: bool = False,
                 use_structural: bool = False,
                 backbone_mode: str = "full",
                 structural_only: bool = False,
                 is_decoder: bool = False):
        super().__init__()
        self.use_graph       = use_graph
        self.use_structural  = use_structural
        # structural_only: the bridge attends ONLY to structural-feature
        # tokens, ignoring retrieved KG nodes. Used for the per-backbone
        # signal ablation (features-only vs KG-only vs both).
        self.structural_only = structural_only
        self.backbone_mode   = backbone_mode
        # Decoder LLMs (e.g. StarCoder) have no pooler and use causal
        # attention, so the summary lives in the last non-pad token.
        self.is_decoder      = is_decoder

        if is_decoder:
            # Load large decoders in bf16 to fit memory; they are never
            # full-fine-tuned (backbone_mode must be lora/frozen).
            self.backbone = AutoModel.from_pretrained(
                backbone_name, torch_dtype=torch.bfloat16)
        else:
            self.backbone = AutoModel.from_pretrained(backbone_name)

        # Cost-efficiency study: how much of the backbone do we train?
        #   "full"   - fine-tune the entire backbone (paper-style, ~124M params)
        #   "frozen" - freeze the backbone; only bridge + classifier train (~0.2%)
        #   "lora"   - inject LoRA adapters into the backbone (~2% trainable)
        if backbone_mode == "frozen":
            for p in self.backbone.parameters():
                p.requires_grad = False
        elif backbone_mode == "lora":
            from peft import LoraConfig, get_peft_model
            # Attention submodule names differ by family; pick by model_type.
            mtype  = getattr(self.backbone.config, "model_type", "_default")
            target = LORA_TARGETS.get(mtype, LORA_TARGETS["_default"])
            lora_cfg = LoraConfig(
                r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.1,
                target_modules=target, bias="none",
            )
            self.backbone = get_peft_model(self.backbone, lora_cfg)
        elif backbone_mode != "full":
            raise ValueError(f"unknown backbone_mode: {backbone_mode}")

        if use_graph:
            self.graph_ca2 = GraphCA2(d_model, fixed_len_pad=fixed_len_pad)
            if use_structural:
                self.struct_proj = StructuralFeatureProjector()
        self.dropout    = nn.Dropout(dropout)
        # Paper uses a single linear layer as classifier
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, input_ids, attention_mask,
                C_AQ=None, graph_mask=None, struct_feats=None):
        out    = self.backbone(input_ids=input_ids,
                               attention_mask=attention_mask)
        if self.is_decoder:
            # Causal LLM: pool the last non-pad token (the only position that
            # has attended to the whole sequence). Cast bf16 -> float32 so the
            # bridge/classifier compute in full precision.
            hidden  = out.last_hidden_state.float()      # (B, T, d)
            last_ix = attention_mask.sum(dim=1) - 1       # (B,)
            h_pool  = hidden[torch.arange(hidden.size(0)), last_ix]
        elif hasattr(out, 'pooler_output') and out.pooler_output is not None:
            # pooler_output = tanh(Linear(CLS)) - the PrimeVul/CodeBERT pattern.
            h_pool = out.pooler_output  # (B, d_model)
        else:
            # Fallback for encoders without a pooler (some GraphCodeBERT configs)
            hidden = out.last_hidden_state
            h_pool = hidden[:, 0, :]

        if self.use_graph and C_AQ is not None:
            # Prepend structural feature tokens to KB tokens. They
            # are always "real" (not padded), so we extend graph_mask with
            # False entries for those positions.
            if self.use_structural and struct_feats is not None:
                struct_tokens = self.struct_proj(struct_feats)  # (B, n_feat, D_G)
                n_feat = struct_tokens.shape[1]
                if self.structural_only:
                    # Features-only: discard KG tokens entirely.
                    C_AQ = struct_tokens
                    graph_mask = None  # all structural tokens are real
                else:
                    C_AQ = torch.cat([struct_tokens, C_AQ], dim=1)
                    if graph_mask is not None:
                        B = graph_mask.shape[0]
                        extra = torch.zeros(B, n_feat, dtype=torch.bool,
                                            device=graph_mask.device)
                        graph_mask = torch.cat([extra, graph_mask], dim=1)
            h_pool = self.graph_ca2(h_pool, C_AQ, graph_mask)

        return self.classifier(self.dropout(h_pool))

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Per-project graph retrieval
# ---------------------------------------------------------------------------

class PerProjectRetriever:
    def __init__(self, graphs_dir: str, fixed_max_k: bool = False):
        self.graphs_dir = Path(graphs_dir)
        self._cache = {}
        # When True, get_caqs always returns shape (MAX_K, D_G),
        # padding short retrievals with zeros. Combined with GraphCA2's
        # fixed_len_pad, this removes the token-count label shortcut.
        self.fixed_max_k = fixed_max_k
        idx_path = self.graphs_dir / "global_index.json"
        self.func_hash_to_project = {}
        if idx_path.exists():
            self.func_hash_to_project = json.load(open(idx_path))

    def _load_project(self, project: str) -> bool:
        if project in self._cache:
            return True
        emb = self.graphs_dir / project / "entity_embeddings.pt"
        idx = self.graphs_dir / project / "entity_to_idx.json"
        if not emb.exists() or not idx.exists():
            return False
        self._cache[project] = (
            json.load(open(idx)),
            torch.load(emb, map_location="cpu", weights_only=True)
        )
        return True

    def _func_name(self, src: str) -> str:
        skip = {"if","for","while","switch","return","sizeof","typeof",
                "static","inline","void","int","char","unsigned","struct"}
        m = re.search(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', src)
        return m.group(1) if m and m.group(1) not in skip else "unknown"

    def _extract_called_apis(self, src: str) -> set:
        """Return the set of API/function names actually called in this source."""
        skip = {"if","for","while","switch","sizeof","typeof","return",
                "do","else","case","break","continue","goto","defined"}
        calls = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', src)
        return {c for c in calls if c not in skip}

    def _extract_local_vars(self, src: str) -> set:
        """Variables declared in the function body."""
        pat = (r'(?:int|char|unsigned|long|short|float|double|void\s*\*|'
               r'bool|size_t|uint\w*|int\w*|__\w+)\s+\*{0,3}'
               r'([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:=|;|,|\[)')
        return set(re.findall(pat, src))

    def get_caqs(self, func_src: str, project: str,
                 max_k: int = MAX_K) -> torch.Tensor:
        """
        Return per-function structural context as C^AQ tokens.

        Retrieval priority (each level fills before the next is queried):
          1. The function's own node embedding (1 token)
          2. Variables DEFINED by this function (function-specific)
          3. ExternalAPIs ACTUALLY CALLED by this function (parsed from src)
          4. Types referenced in this function's body
          5. Other functions in the same project (only if room remaining)

        This replaces the previous "any node prefixed with project::ExternalAPI"
        behaviour which returned the same project-wide boilerplate for every
        query in the same project.
        """
        project = re.sub(r'[^a-zA-Z0-9_\-]', '_', project)
        if not self._load_project(project):
            return torch.zeros(1, D_G)
        entity_to_idx, emb = self._cache[project]
        func_name = self._func_name(func_src)

        retrieved_indices = []  # list of embedding row indices
        seen = set()             # avoid duplicates

        def add_if_present(entity_id):
            if entity_id in entity_to_idx and entity_id not in seen:
                seen.add(entity_id)
                retrieved_indices.append(entity_to_idx[entity_id])

        # Priority 1: the function's own node
        add_if_present(f"{project}::Function::{func_name}")

        # Priority 2: variables defined BY this function
        for var in self._extract_local_vars(func_src):
            if len(retrieved_indices) >= max_k: break
            add_if_present(f"{project}::Variable::{func_name}::{var}")

        # Priority 3: ExternalAPIs ACTUALLY CALLED by this function
        called_apis = self._extract_called_apis(func_src)
        for api in called_apis:
            if len(retrieved_indices) >= max_k: break
            # Try project-namespaced first, then GLOBAL
            for prefix in [f"{project}::ExternalAPI::{api}",
                           f"GLOBAL::ExternalAPI::{api}",
                           f"{project}::Function::{api}"]:
                if prefix in entity_to_idx and prefix not in seen:
                    seen.add(prefix)
                    retrieved_indices.append(entity_to_idx[prefix])
                    break

        # Priority 4: types referenced in this function
        type_pattern = r'\b(?:struct\s+([a-zA-Z_]\w*)|([a-zA-Z_]\w*_t)\b)'
        type_matches = re.findall(type_pattern, func_src)
        types_used = {m[0] or m[1] for m in type_matches if m[0] or m[1]}
        for typ in types_used:
            if len(retrieved_indices) >= max_k: break
            add_if_present(f"{project}::Type::{typ}")

        # Priority 5: 1-hop neighbours via shared types
        # Only if we still have room - find other functions that use the
        # same types as the query function (project-level structural context)
        if len(retrieved_indices) < max_k and types_used:
            for k in entity_to_idx:
                if len(retrieved_indices) >= max_k: break
                if k in seen: continue
                if k.startswith(f"{project}::Function::"):
                    # Heuristic: include if function name contains a related
                    # keyword from the called APIs (cheap proximity signal)
                    fname = k.split("::")[-1]
                    if any(api[:6] in fname or fname[:6] in api
                           for api in called_apis):
                        seen.add(k)
                        retrieved_indices.append(entity_to_idx[k])

        if not retrieved_indices:
            if self.fixed_max_k:
                return torch.zeros(MAX_K, D_G)
            return torch.zeros(1, D_G)

        result = emb[torch.tensor(retrieved_indices)]
        if self.fixed_max_k and result.shape[0] < MAX_K:
            pad = torch.zeros(MAX_K - result.shape[0], D_G,
                              dtype=result.dtype)
            result = torch.cat([result, pad], dim=0)
        return result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PrimeVulDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer,
                 retriever: Optional[PerProjectRetriever] = None,
                 use_graph: bool = False,
                 max_examples: Optional[int] = None):
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.use_graph = use_graph
        self.rows = []
        self.pairs = []  # for pair-wise evaluation
        pair_groups = {}

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: row = json.loads(line)
                except: continue
                func = (row.get("func") or "").strip()
                if not func: continue
                self.rows.append({
                    "func":      func,
                    "target":    int(row.get("target", 0)),
                    "project":   (row.get("project") or "unknown").strip(),
                    "commit_id": row.get("commit_id", ""),
                })
                # Group by commit_id for pair-wise eval
                cid = row.get("commit_id", "")
                if cid:
                    if cid not in pair_groups:
                        pair_groups[cid] = []
                    pair_groups[cid].append(len(self.rows) - 1)

        # Extract pairs (one vuln + one benign per commit)
        for cid, idxs in pair_groups.items():
            vuln   = [i for i in idxs if self.rows[i]["target"] == 1]
            benign = [i for i in idxs if self.rows[i]["target"] == 0]
            if vuln and benign:
                self.pairs.append((vuln[0], benign[0]))

        # Cap training examples if requested
        if max_examples is not None and len(self.rows) > max_examples:
            import random as _r
            _r.seed(SEED)
            _r.shuffle(self.rows)
            self.rows = self.rows[:max_examples]
            # Rebuild pairs after capping
            self.pairs = []
            pair_groups_capped = {}
            for i, row in enumerate(self.rows):
                cid = row.get("commit_id", "")
                if cid:
                    if cid not in pair_groups_capped:
                        pair_groups_capped[cid] = []
                    pair_groups_capped[cid].append(i)
            for cid, idxs in pair_groups_capped.items():
                vuln   = [i for i in idxs if self.rows[i]["target"] == 1]
                benign = [i for i in idxs if self.rows[i]["target"] == 0]
                if vuln and benign:
                    self.pairs.append((vuln[0], benign[0]))

        n_v = sum(1 for r in self.rows if r["target"] == 1)
        cap_str = f" [capped at {max_examples:,}]" if max_examples else ""
        print(f"  {Path(jsonl_path).name}: {len(self.rows):,} examples "
              f"({n_v:,} vuln, {len(self.rows)-n_v:,} benign, "
              f"{len(self.pairs):,} pairs){cap_str}")

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        row  = self.rows[idx]
        enc  = self.tokenizer(row["func"], max_length=BLOCK_SIZE,
                              truncation=True, padding=False)
        C_AQ = torch.zeros(1, D_G)
        n_real = 1
        if self.use_graph and self.retriever:
            C_AQ = self.retriever.get_caqs(row["func"], row["project"])
            if getattr(self.retriever, "fixed_max_k", False):
                n_real = int((C_AQ.abs().sum(dim=-1) > 0).sum().item())
            else:
                n_real = C_AQ.shape[0]
        out = {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label":          row["target"],
            "C_AQ":           C_AQ,
            "n_real":         n_real,
        }
        # Per-function security-relevant structural features.
        # Always extracted when use_graph is on; the model decides whether
        # to consume them via its `use_structural` flag.
        if self.use_graph:
            out["struct_feats"] = extract_structural(row["func"])
        return out


def collate(batch, pad_id):
    def pad(seqs, v):
        mx = max(len(s) for s in seqs)
        return [s + [v]*(mx-len(s)) for s in seqs]
    caqs    = [b["C_AQ"] for b in batch]
    n_reals = [b.get("n_real", c.shape[0]) for b, c in zip(batch, caqs)]
    mk   = max(c.shape[0] for c in caqs)
    pads, masks = [], []
    for c, nr in zip(caqs, n_reals):
        k = c.shape[0]
        if mk - k > 0:
            c = torch.cat([c, torch.zeros(mk - k, D_G)], 0)
        pads.append(c)
        # Pad mask: positions [nr:k] (within-example padding from fixed_max_k)
        # plus [k:mk] (batch-level padding) are both pads.
        m = torch.zeros(mk, dtype=torch.bool)
        m[nr:] = True
        masks.append(m)
    out = {
        "input_ids":      torch.tensor(pad([b["input_ids"]     for b in batch], pad_id), dtype=torch.long),
        "attention_mask": torch.tensor(pad([b["attention_mask"] for b in batch], 0),      dtype=torch.long),
        "labels":         torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "C_AQ":           torch.stack(pads),
        "graph_mask":     torch.stack(masks),
    }
    # Structural features are present only when use_graph=True
    if "struct_feats" in batch[0]:
        out["struct_feats"] = torch.stack([b["struct_feats"] for b in batch])
    return out


# ---------------------------------------------------------------------------
# Metrics (paper convention: VD-S = FNR @ FPR <= 0.5%, LOWER IS BETTER)
# ---------------------------------------------------------------------------

def compute_metrics(labels, probs, threshold: float = 0.5,
                    dataset: Optional[PrimeVulDataset] = None):
    from sklearn.metrics import roc_curve, roc_auc_score, f1_score
    labels = np.array(labels); probs = np.array(probs)
    if len(np.unique(labels)) < 2:
        return {"F1": 0, "AUC": 0, "VD-S_FNR": 100.0,
                "VD-S_TPR": 0.0, "threshold": threshold}

    auc      = roc_auc_score(labels, probs)
    fpr, tpr, thresholds = roc_curve(labels, probs)

    # VD-S: FNR @ FPR <= 0.5% (paper convention - lower is better)
    valid    = fpr <= 0.005
    best_tpr = float(tpr[valid].max()) if valid.any() else 0.0
    vds_fnr  = (1 - best_tpr) * 100    # paper reports FNR as percentage

    preds = (probs >= threshold).astype(int)
    f1    = f1_score(labels, preds, zero_division=0)

    metrics = {
        "F1":        float(f1),
        "AUC":       float(auc),
        "VD-S_FNR":  float(vds_fnr),   # paper metric (lower=better)
        "VD-S_TPR":  float(best_tpr * 100),  # our reporting (higher=better)
        "threshold": float(threshold),
    }

    # Pair-wise evaluation (paper: P-C, P-V, P-B, P-R)
    if dataset is not None and dataset.pairs:
        pc = pv = pb = pr = 0
        for vuln_idx, benign_idx in dataset.pairs:
            vp = probs[vuln_idx] >= threshold
            bp = probs[benign_idx] >= threshold
            if vp and not bp:   pc += 1   # correct
            elif vp and bp:     pv += 1   # both predicted vulnerable
            elif not vp and not bp: pb += 1  # both predicted benign
            else:               pr += 1   # reversed
        n_pairs = len(dataset.pairs)
        metrics.update({
            "P-C": pc / n_pairs * 100,
            "P-V": pv / n_pairs * 100,
            "P-B": pb / n_pairs * 100,
            "P-R": pr / n_pairs * 100,
        })

    return metrics


def find_best_threshold(labels, probs):
    from sklearn.metrics import f1_score
    labels = np.array(labels); probs = np.array(probs)
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.01, 1.0, 0.01):
        f1 = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1; best_t = float(t)
    return best_t, best_f1


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_full(model, dataset, device, use_graph,
                  threshold=0.5, batch_size=EVAL_BATCH_SIZE):
    model.eval()
    tok = AutoTokenizer.from_pretrained(
        model.backbone.config._name_or_path
        if hasattr(model.backbone.config, '_name_or_path')
        else "microsoft/codebert-base")
    pad_id = tok.pad_token_id or 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: collate(b, pad_id),
                        num_workers=0)
    all_labels, all_probs = [], []
    for batch in loader:
        ii = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)
        lb = batch["labels"].to(device)
        cq = batch["C_AQ"].to(device) if use_graph else None
        gm = batch["graph_mask"].to(device) if use_graph else None
        sf = batch["struct_feats"].to(device) if "struct_feats" in batch else None
        logits = model(ii, am, C_AQ=cq, graph_mask=gm, struct_feats=sf)
        probs  = torch.softmax(logits, -1)[:, 1].cpu().numpy()
        all_labels.extend(lb.cpu().numpy())
        all_probs.extend(probs)
    return all_labels, all_probs


@torch.no_grad()
def eval_loss(model, dataset, device, use_graph, criterion, pad_id,
              batch_size=EVAL_BATCH_SIZE):
    """Compute average cross-entropy loss on a dataset."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: collate(b, pad_id),
                        num_workers=0)
    total = 0.0; count = 0
    for batch in loader:
        ii = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)
        lb = batch["labels"].to(device)
        cq = batch["C_AQ"].to(device) if use_graph else None
        gm = batch["graph_mask"].to(device) if use_graph else None
        sf = batch["struct_feats"].to(device) if "struct_feats" in batch else None
        logits = model(ii, am, C_AQ=cq, graph_mask=gm, struct_feats=sf)
        loss   = criterion(logits, lb)
        total += loss.item(); count += 1
    model.train()
    return total / max(count, 1)


def train_system(name, backbone_key, use_graph,
                 train_path, valid_path, test_path,
                 retriever, out_dir, device, epochs=EPOCHS,
                 max_train=None):
    set_seed(SEED)
    backbone_name, d_model = BACKBONE_CONFIGS[backbone_key]

    print(f"\n{'='*64}")
    print(f"System: {name}")
    print(f"  Backbone:   {backbone_name}")
    print(f"  Use graph:  {use_graph}")
    print(f"  Seed:       {SEED}")
    print(f"  Batch size: {TRAIN_BATCH_SIZE}")
    print(f"  LR:         {LEARNING_RATE}")
    print(f"  Warmup:     {WARMUP_STEPS} steps")
    print(f"{'='*64}")

    out_path = Path(out_dir) / name
    out_path.mkdir(parents=True, exist_ok=True)

    tok    = AutoTokenizer.from_pretrained(backbone_name)
    pad_id = tok.pad_token_id or 0

    print("Building datasets ...")
    ret = retriever if use_graph else None
    train_ds = PrimeVulDataset(train_path, tok, ret, use_graph,
                               max_examples=max_train)
    valid_ds = PrimeVulDataset(valid_path, tok, ret, use_graph)
    test_ds  = PrimeVulDataset(test_path,  tok, ret, use_graph)

    col = lambda b: collate(b, pad_id)
    train_ldr = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE,
                           shuffle=True, collate_fn=col, num_workers=0)
    valid_ldr = DataLoader(valid_ds, batch_size=EVAL_BATCH_SIZE,
                           shuffle=False, collate_fn=col, num_workers=0)

    model = VulDetector(backbone_name, d_model, use_graph=use_graph).to(device)
    print(f"  Trainable params: {model.count_trainable():,}")

    # EXACT paper loss: plain cross-entropy, no weights
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE, weight_decay=0.01,  # AdamW default, paper uses this
    )
    total_steps = epochs * len(train_ldr)
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    best_val_loss  = float("inf")
    best_val_f1    = 0.0
    global_step    = 0
    log            = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0; n_batches = 0

        for batch in tqdm(train_ldr, desc=f"Epoch {epoch} [{name}]"):
            ii = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            lb = batch["labels"].to(device)
            cq = batch["C_AQ"].to(device) if use_graph else None
            gm = batch["graph_mask"].to(device) if use_graph else None
            sf = batch["struct_feats"].to(device) if "struct_feats" in batch else None

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

        # Validate after every epoch
        labels, probs = evaluate_full(model, valid_ds, device, use_graph)
        val_loss = -roc_auc_score_safe(labels, probs)  # use AUC as val signal
        best_thresh, val_f1 = find_best_threshold(labels, probs)
        val_metrics = compute_metrics(labels, probs, threshold=best_thresh)

        print(f"\nEpoch {epoch} | avg loss: {epoch_loss/n_batches:.4f} | "
              f"val AUC: {val_metrics['AUC']:.4f} | "
              f"val F1: {val_f1:.4f} | "
              f"val VD-S FNR: {val_metrics['VD-S_FNR']:.2f}%")

        # Paper saves best model by eval loss (evaluate_during_training)
        # We approximate this with best validation cross-entropy loss
        val_ce_loss = eval_loss(model, valid_ds, device, use_graph,
                                nn.CrossEntropyLoss(), pad_id)
        if val_ce_loss < best_val_loss:
            best_val_loss = val_ce_loss
            best_val_f1   = val_metrics['AUC']
            torch.save(model.state_dict(), out_path / "best.pt")
            json.dump({"best_threshold": best_thresh},
                      open(out_path / "threshold.json", "w"))
            print(f"  [checkpoint] saved (val_loss={best_val_loss:.4f} "
                  f"AUC={best_val_f1:.4f})")

        log.append({"epoch": epoch, "val": val_metrics})
        model.train()

    # Test evaluation
    print(f"\nTest evaluation: {name}")
    model.load_state_dict(torch.load(out_path / "best.pt",
                                     map_location=device, weights_only=True))
    best_thresh = json.load(open(out_path / "threshold.json"))["best_threshold"]

    # Calibrate threshold on validation set
    val_labels, val_probs = evaluate_full(model, valid_ds, device, use_graph)
    best_thresh, _ = find_best_threshold(val_labels, val_probs)

    test_labels, test_probs = evaluate_full(
        model, test_ds, device, use_graph)
    # Results at calibrated threshold (our method)
    test_m = compute_metrics(test_labels, test_probs,
                             threshold=best_thresh, dataset=test_ds)
    # Results at fixed 0.5 threshold (paper comparison)
    test_m_05 = compute_metrics(test_labels, test_probs,
                                threshold=0.5, dataset=test_ds)

    print(f"\n  Results for {name}:")
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
               "best_threshold": best_thresh, "log": log},
              open(out_path / "results.json", "w"), indent=2)
    return test_m


def roc_auc_score_safe(labels, probs):
    from sklearn.metrics import roc_auc_score
    labels = np.array(labels)
    if len(np.unique(labels)) < 2:
        return 0.5
    return roc_auc_score(labels, probs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SYSTEMS_CONFIG = {
    # Paper baselines (exact replication)
    "A_codebert":          ("codebert",      False),
    "B_unixcoder":         ("unixcoder",     False),
    "C_graphcodebert":     ("graphcodebert", False),
    # Bridge Attention variants (our contribution)
    "D_codebert_bridge":       ("codebert",      True),
    "E_unixcoder_bridge":      ("unixcoder",     True),
    "F_graphcodebert_bridge":  ("graphcodebert", True),
}


def main(train_path, valid_path, test_path, graphs_dir,
         out_dir, device_str, systems_filter, epochs,
         max_train=None):
    device = torch.device(device_str)

    print("Initialising per-project retriever ...")
    retriever = PerProjectRetriever(graphs_dir) \
        if Path(graphs_dir).exists() else None
    if retriever is None:
        print("  WARNING: graphs_dir not found - bridge systems will use "
              "zero tensors (run build_primevul_graphs.py first)")

    # Select systems to run
    if systems_filter == "baselines":
        run = {k: v for k, v in SYSTEMS_CONFIG.items()
               if not v[1]}
    elif systems_filter == "bridge":
        run = {k: v for k, v in SYSTEMS_CONFIG.items()
               if v[1]}
    elif systems_filter == "all":
        run = SYSTEMS_CONFIG
    else:
        wanted = [s.strip() for s in systems_filter.split(",")]
        run = {k: v for k, v in SYSTEMS_CONFIG.items()
               if any(w in k for w in wanted)}

    print(f"\nRunning {len(run)} systems: {list(run.keys())}")

    all_results = {}
    for sys_name, (backbone_key, use_graph) in run.items():
        result = train_system(
            sys_name, backbone_key, use_graph,
            train_path, valid_path, test_path,
            retriever, out_dir, device, epochs=epochs,
            max_train=max_train,
        )
        all_results[sys_name] = result

    # Final comparison table
    print(f"\n{'='*80}")
    print("PHASE 1 RESULTS  (paper convention: VD-S = FNR, LOWER IS BETTER)")
    print(f"{'='*80}")
    print(f"{'System':<30} {'F1':>7} {'AUC':>7} "
          f"{'VD-S FNR%':>10} {'VD-S TPR%':>10} {'P-C%':>6}")
    print("-"*80)

    paper_ref = {
        "CodeT5  (paper)":    {"F1": 0.197, "VD-S_FNR": 89.93, "P-C": 1.06},
        "CodeBERT (paper)":   {"F1": 0.209, "VD-S_FNR": 88.78, "P-C": 1.77},
        "UnixCoder (paper)":  {"F1": 0.214, "VD-S_FNR": 89.21, "P-C": 1.60},
        "StarCoder2 (paper)": {"F1": 0.181, "VD-S_FNR": 89.64, "P-C": 2.30},
    }

    for ref_name, ref_m in paper_ref.items():
        pc = ref_m.get("P-C", "-")
        print(f"  {ref_name:<28} {ref_m['F1']:>7.3f}     -   "
              f"{ref_m['VD-S_FNR']:>10.2f}         -   "
              f"{pc:>6}")

    print("-"*80)
    for s, m in all_results.items():
        pc = f"{m.get('P-C', 0):.1f}" if "P-C" in m else "-"
        tpr = m.get('VD-S_TPR', 100 - m.get('VD-S_FNR', 100))
        marker = " <- better than PrimeVul CodeBERT" if m.get("VD-S_FNR", 100) < 88.78 else ""
        print(f"  {s:<28} {m['F1']:>7.3f} {m['AUC']:>7.3f} "
              f"{m['VD-S_FNR']:>10.2f} {tpr:>10.2f} "
              f"{pc:>6}{marker}")

    json.dump(all_results,
              open(Path(out_dir) / "phase1_results.json", "w"), indent=2)
    print(f"\nAll results: {Path(out_dir) / 'phase1_results.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train",   required=True)
    ap.add_argument("--valid",   required=True)
    ap.add_argument("--test",    required=True)
    ap.add_argument("--graphs",  default="./datasets/primevul_graphs")
    ap.add_argument("--out",     default="./checkpoints/primevul_phase1")
    ap.add_argument("--systems", default="all",
                    help="baselines | bridge | all | comma-separated names")
    ap.add_argument("--epochs",    type=int, default=EPOCHS)
    ap.add_argument("--max_train", type=int, default=None,
                    help="Cap training examples for quick iteration. "
                         "None = full dataset (paper setting).")
    ap.add_argument("--device",    default="cuda"
                    if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    main(args.train, args.valid, args.test, args.graphs,
         args.out, args.device, args.systems, args.epochs,
         max_train=args.max_train)