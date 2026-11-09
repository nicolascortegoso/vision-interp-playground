"""Analysis on cached activations: linear probes, difference-of-means directions, geometry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import FACTORS, PlaymobilData

# For each probed factor, the factor whose values we hold out to build test folds.
# Character and pose are tested on unseen viewpoints; angle on unseen characters
# (holding out an angle would make it impossible to predict that angle).
DEFAULT_GROUPS = {"character": "angle", "arm_position": "angle", "angle": "character"}


def _np(x):
    return x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def probe_accuracy(X, y, groups, C: float = 1.0, pca: int | None = None) -> float:
    """Mean held-out accuracy of a logistic-regression probe, one fold per group value.

    pca: if given, reduce to this many principal components first (fit on each training
    fold only). Useful for very wide inputs such as flattened conv maps.
    """
    from sklearn.decomposition import PCA

    X, y, groups = _np(X), _np(y), _np(groups)
    accs = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        steps = [StandardScaler()] + ([PCA(pca, random_state=0)] if pca else []) + [LogisticRegression(C=C, max_iter=3000)]
        clf = make_pipeline(*steps)
        clf.fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    return float(np.mean(accs))


def probe_all_layers(
    acts: dict[str, torch.Tensor],
    data: PlaymobilData,
    factors=("character", "arm_position", "angle"),
    groups: dict[str, str] = DEFAULT_GROUPS,
    C: float = 1.0,
) -> pd.DataFrame:
    """Probe accuracy for every (layer, factor). Rows: layers in order; columns: factors."""
    rows = {}
    for layer, X in acts.items():
        rows[layer] = {f: probe_accuracy(X, data.y(f), data.y(groups[f]), C=C) for f in factors}
    return pd.DataFrame.from_dict(rows, orient="index")


def class_means(X: torch.Tensor, y: torch.Tensor, n_classes: int | None = None) -> torch.Tensor:
    """[K, D] mean activation per class. Because the dataset is fully factorial, each class
    mean already averages uniformly over every combination of the other factors."""
    n_classes = n_classes or int(y.max()) + 1
    return torch.stack([X[y == k].mean(0) for k in range(n_classes)])


def diff_of_means(X: torch.Tensor, y: torch.Tensor, a: int, b: int) -> torch.Tensor:
    """mean(X | y == a) - mean(X | y == b)."""
    return X[y == a].mean(0) - X[y == b].mean(0)


def factor_directions(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """[K, D] directions: each class mean minus the grand mean."""
    m = class_means(X, y)
    return m - m.mean(0, keepdim=True)


def cosine_matrix(A: torch.Tensor, B: torch.Tensor | None = None) -> torch.Tensor:
    A = torch.nn.functional.normalize(A, dim=-1)
    B = A if B is None else torch.nn.functional.normalize(B, dim=-1)
    return A @ B.T


def nearest_neighbors(queries: torch.Tensor, X: torch.Tensor, k: int = 1, metric: str = "cosine") -> torch.Tensor:
    """[Q, k] indices into X of the nearest rows to each query."""
    if metric == "cosine":
        sim = cosine_matrix(queries, X)
    else:
        sim = -torch.cdist(queries, X)
    return sim.topk(k, dim=1).indices


def pca(X: torch.Tensor, k: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Project X onto its top-k principal components. Returns (projection [N, k], explained variance ratio [k])."""
    Xc = X - X.mean(0, keepdim=True)
    U, S, _ = torch.linalg.svd(Xc, full_matrices=False)
    var = S**2 / (S**2).sum()
    return U[:, :k] * S[:k], var[:k]


def _basis(A: torch.Tensor, tol: float = 1e-5) -> torch.Tensor:
    """Orthonormal basis [D, r] for the span of the rows of A (rank-aware)."""
    U, S, _ = torch.linalg.svd(A.T, full_matrices=False)
    return U[:, S > tol * S.max()]


def subspace_overlap(A: torch.Tensor, B: torch.Tensor) -> float:
    """Mean squared cosine of the principal angles between span(A) and span(B) (rows are vectors),
    averaged over the smaller subspace. 0 = orthogonal, 1 = the smaller lies inside the larger.
    Two random subspaces of dims a <= b in D dimensions give about b / D."""
    Qa, Qb = _basis(A), _basis(B)
    s = torch.linalg.svdvals(Qa.T @ Qb)
    return float((s**2).sum() / min(Qa.shape[1], Qb.shape[1]))


def shuffled_overlap(X: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor, n: int = 20, seed: int = 0) -> float:
    """Null baseline for `subspace_overlap`: the mean overlap between the factor directions of two
    independently shuffled label sets. Same class sizes and same activation geometry, but no real
    factor. Unlike random Gaussian directions, this accounts for activations being concentrated
    in a few high-variance directions."""
    g = torch.Generator().manual_seed(seed)
    vals = [
        subspace_overlap(
            factor_directions(X, y1[torch.randperm(len(y1), generator=g)]),
            factor_directions(X, y2[torch.randperm(len(y2), generator=g)]),
        )
        for _ in range(n)
    ]
    return float(np.mean(vals))


def variance_explained(X: torch.Tensor, y: torch.Tensor) -> float:
    """Fraction of total variance explained by the class means of `y` (between-class / total)."""
    m = X.mean(0, keepdim=True)
    total = ((X - m) ** 2).sum()
    means = class_means(X, y)
    counts = torch.bincount(y, minlength=len(means)).to(X.dtype)
    between = (counts[:, None] * (means - m) ** 2).sum()
    return float(between / total)


_ORDER = ("character", "arm_position", "angle")


def analogy_accuracy(X: torch.Tensor, data: PlaymobilData, swap: str = "arm_position", across: str = "character") -> dict[str, float]:
    """Vector arithmetic X_p - X_q + Y_q -> Y_p, scored by nearest neighbour over all images.

    For every pair of `across` values (X != Y), pair of `swap` values (p != q) and every value
    of the remaining factor, build the query from three images and check whether its nearest
    neighbour (cosine, on mean-centred activations, the three input images excluded) is the
    true image Y_p. `baseline` uses Y_q alone as the query: how often its nearest neighbour is
    Y_p without any arithmetic. `analogy (any <rest>)` is lenient: the nearest neighbour only
    needs the right `across` and `swap` values, at any value of the remaining factor.
    """
    shape = [len(FACTORS[f]) for f in _ORDER]
    keys = data.labels[[f"{f}_idx" for f in _ORDER]].to_numpy().T
    assert (np.ravel_multi_index(keys, shape) == np.arange(len(X))).all(), "data must be in (character, arm_position, angle) order"
    rest = next(f for f in _ORDER if f not in (swap, across))
    perm = [_ORDER.index(across), _ORDER.index(swap), _ORDER.index(rest)]
    Xc = X - X.mean(0, keepdim=True)
    Xn = torch.nn.functional.normalize(Xc, dim=-1)
    A = Xc.reshape(*shape, -1).permute(*perm, 3)
    idx = torch.arange(len(X)).reshape(*shape).permute(*perm)
    n_x, n_s, n_r = idx.shape
    off_diag = ~torch.eye(n_s, dtype=torch.bool)[:, :, None].expand(n_s, n_s, n_r)

    lab_a, lab_s = data.y(across), data.y(swap)
    hits = loose_hits = base_hits = total = 0
    for x in range(n_x):
        for y in range(n_x):
            if x == y:
                continue
            # query[p, q, r] = X_p - X_q + Y_q ; target = Y_p
            q = A[x][:, None] - A[x][None, :] + A[y][None, :]
            target = idx[y][:, None, :].expand(n_s, n_s, n_r)
            sims = torch.nn.functional.normalize(q, dim=-1) @ Xn.T
            for ex in (idx[x][:, None, :], idx[x][None, :, :], idx[y][None, :, :]):
                sims.scatter_(-1, ex.expand(n_s, n_s, n_r)[..., None], float("-inf"))
            pred = sims.argmax(-1)
            hits += (pred == target)[off_diag].sum().item()
            loose = (lab_a[pred] == lab_a[target]) & (lab_s[pred] == lab_s[target])
            loose_hits += loose[off_diag].sum().item()

            base = Xn[idx[y]] @ Xn.T  # [q, r, N]: query Y_q alone
            base.scatter_(-1, idx[y][..., None], float("-inf"))
            pred = base.argmax(-1)  # [q, r]
            base_hits += (pred[None, :, :] == target)[off_diag].sum().item()
            total += off_diag.sum().item()
    return {"analogy": hits / total, f"analogy (any {rest})": loose_hits / total, "baseline": base_hits / total}


def ring_order_ok(P: torch.Tensor) -> bool:
    """True if the 8 points P [8, 2] (in ANGLES order), sorted by polar angle around their
    centroid, go round in angle order (either direction)."""
    c = P - P.mean(0)
    order = torch.atan2(c[:, 1], c[:, 0]).argsort().tolist()
    k = order.index(0)
    seq = order[k:] + order[:k]
    return seq == list(range(8)) or seq == [0] + list(range(7, 0, -1))


def mirror_collapse(angle_means: torch.Tensor) -> float:
    """For [8, D] per-angle means in ANGLES order: mean distance between mirror-image views
    (45/315, 90/270, 135/225) divided by the mean distance over all pairs of views.
    Near 0: mirror views coincide (the ring folds in half). Near 1: no special relation."""
    D = torch.cdist(angle_means, angle_means)
    mirror = torch.stack([D[1, 7], D[2, 6], D[3, 5]]).mean()
    allp = D[torch.triu(torch.ones(8, 8, dtype=torch.bool), 1)].mean()
    return float(mirror / allp)


@torch.no_grad()
def steering_sweep(model, x, layer, vector, alphas, readout, target, batch_size: int = 64, device=None, **steer_kwargs):
    """Fraction of `x` whose readout prediction equals `target` while adding alpha * vector at `layer`.

    readout: maps the model output to logits, e.g. `lambda out: out["character"]`.
    target: a class index, or a tensor of per-image labels (to measure accuracy under steering).
    steer_kwargs: passed to `steer`, e.g. `token=0` to steer only a ViT's CLS token.
    Returns (rates, logits) where logits[i] holds the readout logits at alphas[i].
    """
    from .hooks import steer

    device = device or next(model.parameters()).device
    rates, all_logits = [], []
    for alpha in alphas:
        outs = []
        with steer(model, layer, vector, alpha, **steer_kwargs):
            for i in range(0, len(x), batch_size):
                outs.append(readout(model(x[i : i + batch_size].to(device))).float().cpu())
        logits = torch.cat(outs)
        all_logits.append(logits)
        rates.append((logits.argmax(1) == target).float().mean().item())
    return rates, all_logits
