"""Model definitions: node/edge/window classifiers built on GraphSAGE, GCN and GAT
embedders.
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool


class SAGEEmbedder(torch.nn.Module):
    """Node embedder using GraphSAGE aggregation. Inductive -> generalises to unseen nodes.
    Reference: Hamilton et al., NeurIPS 2017, arxiv.org/abs/1706.02216
    """

    def __init__(self, in_dim, hidden_dim=16, out_dim=16, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

    def encode(self, x, edge_index):
        return self.forward(x, edge_index)


class NodeClassifier(torch.nn.Module):
    """Node-level risk classifier."""

    def __init__(self, in_dim, hidden_dim=16):
        super().__init__()
        self.embedder = SAGEEmbedder(in_dim, hidden_dim, out_dim=hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index):
        return self.classifier(self.embedder(x, edge_index))


class EdgeClassifier(torch.nn.Module):
    """Flow-level classifier, topology only (no edge attributes)."""

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
    """Flow-level classifier with per-flow edge attributes concatenated onto the endpoint
    embeddings. This is the SCADANet Track B / enriched model and the BATADAL temporal
    edge model.
    """

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


class GCNEmbedder(torch.nn.Module):
    """GCN counterpart to SAGEEmbedder, with the same depth, hidden_dim and dropout."""

    def __init__(self, in_dim, hidden_dim=16, out_dim=16, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


class GATEmbedder(torch.nn.Module):
    """GAT counterpart to SAGEEmbedder. heads=4/concat=False (mean over heads) keeps the
    output width equal to hidden_dim/out_dim, so this drops into the exact same
    EdgeClassifierWithAttr-style head as the SAGE and GCN variants with no downstream
    dimension changes — the fairness requirement is same classifier capacity where
    architecturally possible, not just same data/splits.
    """

    def __init__(self, in_dim, hidden_dim=16, out_dim=16, heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, concat=False)
        self.conv2 = GATConv(hidden_dim, out_dim, heads=heads, concat=False)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


_EMBEDDER_REGISTRY = {
    "graphsage": SAGEEmbedder,
    "gcn": GCNEmbedder,
    "gat": GATEmbedder,
}


class GNNEdgeClassifierWithAttr(torch.nn.Module):
    """Same architecture as EdgeClassifierWithAttr, parameterized over the conv type that
    builds the node embedder. `gnn_edge_classifier(conv_type="graphsage", ...)` is
    architecturally identical to EdgeClassifierWithAttr.
    """

    def __init__(self, conv_type, in_dim, edge_attr_dim, hidden_dim=16,
                 edge_proj_dim=16, dropout=0.3):
        super().__init__()
        if conv_type not in _EMBEDDER_REGISTRY:
            raise ValueError(f"Unknown conv_type {conv_type!r}, expected one of {list(_EMBEDDER_REGISTRY)}")
        self.conv_type = conv_type
        embedder_cls = _EMBEDDER_REGISTRY[conv_type]
        self.embedder = embedder_cls(in_dim, hidden_dim, out_dim=hidden_dim, dropout=dropout)
        self.edge_proj = torch.nn.Sequential(
            torch.nn.Linear(edge_attr_dim, edge_proj_dim),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2 + edge_proj_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, edge_index, query_edges, query_edge_attr):
        h = self.embedder(x, edge_index)
        edge_repr = self.edge_proj(query_edge_attr)
        combined = torch.cat(
            [h[query_edges[0]], h[query_edges[1]], edge_repr], dim=1
        )
        return self.head(combined)


class EndpointOnlyEdgeClassifier(torch.nn.Module):
    """Keeps a per-node representation but disables neighborhood aggregation entirely.
    `x`/`edge_index` are accepted (same call signature as GNNEdgeClassifierWithAttr) but
    never used to pass information between nodes.

    A per-node MLP over a constant input would give every node an identical vector,
    collapsing this condition into content-only. To keep "endpoint identity without
    aggregation" distinct, this class uses a learnable per-node embedding table
    (nn.Embedding): each node gets its own trainable vector, updated only by gradients
    from edges touching that node. This tests whether knowing which node is on each end
    helps, without any message passing.
    """

    def __init__(self, num_nodes, edge_attr_dim, hidden_dim=16, edge_proj_dim=16, dropout=0.3):
        super().__init__()
        self.node_embedding = torch.nn.Embedding(num_nodes, hidden_dim)
        self.edge_proj = torch.nn.Sequential(
            torch.nn.Linear(edge_attr_dim, edge_proj_dim),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2 + edge_proj_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, edge_index, query_edges, query_edge_attr):
        # x, edge_index accepted but unused — no aggregation, by design.
        h = self.node_embedding.weight
        edge_repr = self.edge_proj(query_edge_attr)
        combined = torch.cat(
            [h[query_edges[0]], h[query_edges[1]], edge_repr], dim=1
        )
        return self.head(combined)


class GNNWindowClassifier(torch.nn.Module):
    """Same architecture as WindowGraphClassifier, parameterized over
    conv type — the BATADAL counterpart to GNNEdgeClassifierWithAttr
    above, needed for Experiment B3's "GCN or GAT" requirement."""

    def __init__(self, conv_type, in_dim, hidden_dim=16, dropout=0.3):
        super().__init__()
        if conv_type not in _EMBEDDER_REGISTRY:
            raise ValueError(f"Unknown conv_type {conv_type!r}, expected one of {list(_EMBEDDER_REGISTRY)}")
        self.conv_type = conv_type
        embedder_cls = _EMBEDDER_REGISTRY[conv_type]
        self.embedder = embedder_cls(in_dim, hidden_dim, out_dim=hidden_dim, dropout=dropout)
        self.head = torch.nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index, batch):
        h = self.embedder(x, edge_index)
        h = global_mean_pool(h, batch)
        return self.head(h)


class WindowGraphClassifier(torch.nn.Module):
    """GraphSAGE embedder + mean pooling + linear head -> per-window (graph-level) anomaly
    classification. Used for the BATADAL windowed cross-dataset validation and the
    BATADAL temporal split.
    """

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
    """SCADANet Track B ONLY — every individual label (each attack subtype + normal)
    weighted inversely to its own frequency in the training set, not just attack-vs-
    normal. This is deliberately different from build_class_weights(): Track B covers
    several single-fixed-IP attack subtypes at very different volumes (scans vs.
    floods), and a plain binary weight would let the dominant subtype drown out the
    rest. Do not substitute build_class_weights() here — it changes the experiment, not
    just refactors it.
    """
    labels_subset = df.iloc[idx.numpy()][label_col]
    counts = labels_subset.value_counts()
    n = len(labels_subset)
    n_classes = len(counts)
    return {label: n / (n_classes * count) for label, count in counts.items()}
