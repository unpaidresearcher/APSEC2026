"""
train_primevul_transh.py
========================
Trains TransH embeddings for each per-project PrimeVul graph.

Runs the same TransH training loop used for BigVul but:
  - Processes each project graph independently
  - Uses a shared entity vocabulary across projects for ExternalAPI nodes
    (so kfree, malloc etc. have consistent embeddings across projects)
  - Saves one embedding file per project

Run:
    python train_primevul_transh.py \\
        --graphs ../data/primevul_graphs \\
        --out    ../data/primevul_graphs \\
        --epochs 50 \\
        --dim    128
"""

import json
import sys
import pickle
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, ".")

DIM      = 128
MARGIN   = 1.0
LR       = 1e-3
EPOCHS   = 50
BATCH    = 512
NEG_RATE = 5      # negative samples per positive triple
LOG_EVERY = 10


# ---------------------------------------------------------------------------
# TransH model (same as original, reused)
# ---------------------------------------------------------------------------

class TransH(nn.Module):
    def __init__(self, n_entities: int, n_relations: int,
                 dim: int = DIM, margin: float = MARGIN):
        super().__init__()
        self.dim    = dim
        self.margin = margin

        self.entity_emb   = nn.Embedding(n_entities,  dim)
        self.relation_emb = nn.Embedding(n_relations, dim)
        self.normal_vecs  = nn.Embedding(n_relations, dim)  # hyperplane normals

        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        nn.init.xavier_uniform_(self.normal_vecs.weight)

    def project(self, e_emb, w):
        """Project entity onto relation hyperplane."""
        return e_emb - (e_emb * w).sum(-1, keepdim=True) * w

    def score(self, h_idx, r_idx, t_idx):
        h = F.normalize(self.entity_emb(h_idx), dim=-1)
        t = F.normalize(self.entity_emb(t_idx), dim=-1)
        r = self.relation_emb(r_idx)
        w = F.normalize(self.normal_vecs(r_idx), dim=-1)
        h_p = self.project(h, w)
        t_p = self.project(t, w)
        return torch.norm(h_p + r - t_p, p=2, dim=-1)

    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        pos_score = self.score(pos_h, pos_r, pos_t)
        neg_score = self.score(neg_h, neg_r, neg_t)
        loss = F.relu(self.margin + pos_score - neg_score).mean()
        return loss


# ---------------------------------------------------------------------------
# Triple dataset
# ---------------------------------------------------------------------------

class TripleDataset(Dataset):
    def __init__(self, triples, n_entities, neg_rate=NEG_RATE):
        self.triples    = triples
        self.n_entities = n_entities
        self.neg_rate   = neg_rate

    def __len__(self): return len(self.triples) * self.neg_rate

    def __getitem__(self, idx):
        triple = self.triples[idx % len(self.triples)]
        h, r, t = triple
        # Corrupt head or tail randomly
        if torch.rand(1).item() > 0.5:
            neg_h = torch.randint(self.n_entities, (1,)).item()
            neg_t = t
        else:
            neg_h = h
            neg_t = torch.randint(self.n_entities, (1,)).item()
        return (torch.tensor(h), torch.tensor(r), torch.tensor(t),
                torch.tensor(neg_h), torch.tensor(r), torch.tensor(neg_t))


def collate_triples(batch):
    ph, pr, pt, nh, nr, nt = zip(*batch)
    return (torch.stack(ph), torch.stack(pr), torch.stack(pt),
            torch.stack(nh), torch.stack(nr), torch.stack(nt))


# ---------------------------------------------------------------------------
# Train one project
# ---------------------------------------------------------------------------

RELATION_TO_IDX = {r: i for i, r in enumerate([
    "CALLS", "DEFINES", "CONTAINS", "HAS_TYPE",
    "PRECEDES", "WRITES", "READS", "RETURNS", "USES"
])}


def train_project(project: str, graph_path: Path, out_dir: Path,
                  dim: int = DIM, epochs: int = EPOCHS,
                  device: torch.device = torch.device("cpu")):
    """Train TransH embeddings for one project graph."""

    # Load graph
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    if G.number_of_edges() < 10:
        return None  # skip trivially small graphs

    # Build entity vocabulary
    entities = sorted(G.nodes())
    entity_to_idx = {e: i for i, e in enumerate(entities)}
    n_entities  = len(entities)
    n_relations = len(RELATION_TO_IDX)

    # Extract triples as integer indices
    triples = []
    for h, t, data in G.edges(data=True):
        r = data.get("relation", "CALLS")
        r_idx = RELATION_TO_IDX.get(r, 0)
        if h in entity_to_idx and t in entity_to_idx:
            triples.append((entity_to_idx[h], r_idx, entity_to_idx[t]))

    if len(triples) < 5:
        return None

    ds     = TripleDataset(triples, n_entities)
    loader = DataLoader(ds, batch_size=min(BATCH, len(ds)),
                        shuffle=True, collate_fn=collate_triples,
                        num_workers=0)

    model = TransH(n_entities, n_relations, dim=dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0; count = 0
        for batch in loader:
            ph, pr, pt, nh, nr, nt = [b.to(device) for b in batch]
            loss = model(ph, pr, pt, nh, nr, nt)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); count += 1
        avg = total / max(count, 1)
        if avg < best_loss:
            best_loss = avg

    # Save embeddings
    proj_out = out_dir / project
    proj_out.mkdir(parents=True, exist_ok=True)

    emb = F.normalize(model.entity_emb.weight.detach().cpu(), dim=-1)
    torch.save(emb, proj_out / "entity_embeddings.pt")
    json.dump(entity_to_idx, open(proj_out / "entity_to_idx.json", "w"))
    json.dump({"n_entities": n_entities, "n_relations": n_relations,
               "n_triples": len(triples), "best_loss": best_loss},
              open(proj_out / "transh_stats.json", "w"), indent=2)

    return {"project": project, "n_entities": n_entities,
            "n_triples": len(triples), "best_loss": best_loss}


# ---------------------------------------------------------------------------
# Main: train all projects
# ---------------------------------------------------------------------------

def train_all(graphs_dir: str, out_dir: str, dim: int, epochs: int,
              device_str: str):
    device    = torch.device(device_str)
    graphs    = Path(graphs_dir)
    out_path  = Path(out_dir)

    # Find all project graph files
    graph_files = sorted(graphs.glob("*/graph.gpickle"))
    print(f"Found {len(graph_files):,} project graphs to train")

    stats   = []
    skipped = 0
    for graph_path in tqdm(graph_files, desc="Training TransH"):
        project = graph_path.parent.name
        result  = train_project(project, graph_path, out_path,
                                dim=dim, epochs=epochs, device=device)
        if result:
            stats.append(result)
        else:
            skipped += 1

    json.dump(stats, open(out_path / "transh_stats.json", "w"), indent=2)

    total_entities = sum(s["n_entities"] for s in stats)
    total_triples  = sum(s["n_triples"]  for s in stats)
    print(f"\nTransH training complete:")
    print(f"  Projects trained: {len(stats):,}")
    print(f"  Projects skipped (too small): {skipped:,}")
    print(f"  Total entities: {total_entities:,}")
    print(f"  Total triples: {total_triples:,}")
    print(f"\nNext: train HGT encoders")
    print(f"  python train_primevul_hgt.py --graphs {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs",  default="./datasets/primevul_graphs")
    ap.add_argument("--out",     default="./datasets/primevul_graphs")
    ap.add_argument("--dim",     type=int, default=128)
    ap.add_argument("--epochs",  type=int, default=50)
    ap.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    train_all(args.graphs, args.out, args.dim, args.epochs, args.device)
