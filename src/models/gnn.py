"""
Model definitions, moved verbatim (architecture unchanged) from the
Week 4/6/10 notebook cells. Do not redesign architecture here — this
issue is refactor-only (see "Do Not Do In This Issue").
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


class SAGEEmbedder(torch.nn.Module):
    """Node embedder using GraphSAGE aggregation. Inductive -> generalises
    to unseen nodes (used for BATADAL cross-dataset transfer, Week 5).
    Reference: Hamilton et al., NeurIPS 2017, arxiv.org/abs/1706.02216"""

    def __init__(self, in_dim, hidden_dim=16, out_dim=16):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

    def encode(self, x, edge_index):
        return self.forward(x, edge_index)


class NodeClassifier(torch.nn.Module):
    """Node-level risk classifier. Notebook Cell 77 docstring says "9 nodes
    — no meaningful test split possible"; per week12_metricsnew.json the
    corrected ICS-Flow device count is 8 (Asset+Attacker nodes; 7,669
    Alert nodes are separate and need a table footnote). Fix the printed
    node count here if you wire up ICS-Flow, but the "too few to split"
    architectural point holds either way."""

    def __init__(self, in_dim, hidden_dim=16):
        super().__init__()
        self.embedder = SAGEEmbedder(in_dim, hidden_dim, out_dim=hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index):
        return self.classifier(self.embedder(x, edge_index))


class EdgeClassifier(torch.nn.Module):
    """Flow-level classifier, topology only (no edge attributes).
    Used for the ICS-Flow flow-level model and the SCADANet Week 6
    baseline (Track A/B topology-only comparison)."""

    def __init__(self, in_dim, hidden_dim=16):
        super().__init__()
        self.embedder = SAGEEmbedder(in_dim, hidden_dim, out_dim=hidden_dim)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, edge_index, query_edges):
        h = self.embedder(x, edge_index)
        combined = torch.cat([h[query_edges[0]], h[query_edges[1]]], dim=1)
        return self.head(combined)


class EdgeClassifierWithAttr(torch.nn.Module):
    """Flow-level classifier with per-flow edge attributes concatenated
    onto the endpoint embeddings. This is the SCADANet Track B / enriched
    model (Week 7 Part G.2) and the BATADAL temporal edge model (Part X)."""

    def __init__(self, in_dim, edge_attr_dim, hidden_dim=16, edge_proj_dim=16):
        super().__init__()
        self.embedder = SAGEEmbedder(in_dim, hidden_dim, out_dim=hidden_dim)
        self.edge_proj = torch.nn.Sequential(
            torch.nn.Linear(edge_attr_dim, edge_proj_dim),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2 + edge_proj_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, edge_index, query_edges, query_edge_attr):
        h = self.embedder(x, edge_index)
        edge_repr = self.edge_proj(query_edge_attr)
        combined = torch.cat(
            [h[query_edges[0]], h[query_edges[1]], edge_repr], dim=1
        )
        return self.head(combined)


class WindowGraphClassifier(torch.nn.Module):
    """GraphSAGE embedder + mean pooling + linear head -> per-window
    (graph-level) anomaly classification. Used for the BATADAL windowed
    cross-dataset validation (Week 5 Part D) and the BATADAL temporal
    split (Part Q/T)."""

    def __init__(self, in_dim, hidden_dim=16):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.head = torch.nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index, batch):
        h = F.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        h = global_mean_pool(h, batch)
        return self.head(h)


def build_class_weights(train_y: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency class weights computed on the TRAIN split only
    (never on the full dataset — that would leak test-set class balance
    into training). Used by SCADANet-temporal, BATADAL-temporal, and
    BATADAL-static-cv (all of which weight binary attack/normal only)."""
    class_counts = torch.bincount(train_y, minlength=2).float()
    return train_y.shape[0] / (2.0 * class_counts.clamp(min=1))


def compute_subtype_balanced_weights(df, label_col: str, idx: torch.Tensor) -> dict:
    """SCADANet Track B ONLY (Week 8 Cell 123) — every individual label
    (each attack subtype + normal) weighted inversely to its own frequency
    in the training set, not just attack-vs-normal. This is deliberately
    different from build_class_weights(): Track B covers several
    single-fixed-IP attack subtypes at very different volumes (scans vs.
    floods), and a plain binary weight would let the dominant subtype
    drown out the rest. Do not substitute build_class_weights() here —
    it changes the experiment, not just refactors it."""
    labels_subset = df.iloc[idx.numpy()][label_col]
    counts = labels_subset.value_counts()
    n = len(labels_subset)
    n_classes = len(counts)
    return {label: n / (n_classes * count) for label, count in counts.items()}
