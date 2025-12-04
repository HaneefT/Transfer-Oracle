# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: TOvenv (3.9.13)
#     language: python
#     name: python3
# ---

# %% [markdown]
# Clustering/KNN with leak-free pipeline helpers

# %%
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from src.data_preprocessing.features import (
    stratified_split,
    feature_mask_from_train,
    fit_transforms,
    evaluate_k,
    stability_score,
    build_per_cluster_knn,
)
from src.data_preprocessing.groups import GROUPS

CATEGORICAL_COLS = [
    "Rk",
    "Player",
    "Nation",
    "Pos",
    "Squad",
    "Comp",
    "Age",
    "Born",
    "MP",
    "Starts",
    "Min",
    "90s",
    "numeric_wage",
    "foot",
    "W",
    "D",
    "L",
]

# Resolve data dir relative to repository root
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def load_position_df(pos: str) -> pd.DataFrame:
    path = DATA_DIR / f"players_data_{pos}_normalized.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing normalized parquet for {pos} at {path}")
    return pd.read_parquet(path).reset_index(drop=True)


def select_group_columns(df: pd.DataFrame, use_groups) -> list[str] | None:
    if not use_groups:
        return None
    selected: list[str] = []
    for g in use_groups:
        cols = GROUPS.get(g, [])
        selected.extend(cols)
    selected = list(dict.fromkeys(selected))  # dedupe, keep order
    return [c for c in selected if c in df.columns]


def select_top_loading_cols(
    train_df: pd.DataFrame,
    base_allowed: list[str] | None,
    top_n: int = 15,
) -> list[str]:
    """
    Train-only PCA to rank raw numeric features by loading on PC1/PC2,
    then return the union of the top_n from each component.
    """
    numeric_cols = [c for c in train_df.columns if c not in CATEGORICAL_COLS]
    if base_allowed is not None:
        numeric_cols = [c for c in numeric_cols if c in base_allowed]

    num_df = train_df[numeric_cols]
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    num_imp = imputer.fit_transform(num_df)
    num_scaled = scaler.fit_transform(num_imp)

    pca = PCA(random_state=42)
    pca.fit(num_scaled)
    comps = pca.components_

    if comps.shape[0] < 2:
        # Not enough components to rank; fall back to all numeric cols
        return numeric_cols

    loadings = pd.DataFrame(
        comps[:2].T,
        columns=["PC1", "PC2"],
        index=numeric_cols,
    )
    top_pc1 = loadings["PC1"].abs().nlargest(top_n).index
    top_pc2 = loadings["PC2"].abs().nlargest(top_n).index
    selected = list(dict.fromkeys(list(top_pc1) + list(top_pc2)))
    return selected


def sweep_k(train_X, val_X, val_df, k_grid, seed: int = 42) -> pd.DataFrame:
    rows = []
    for k in k_grid:
        metrics, _, _ = evaluate_k(train_X, val_X, k, seed=seed)
        stab = stability_score(train_X, val_X, k, seed=seed)
        rows.append({"k": k, **metrics, "stability_ari": stab})
    return pd.DataFrame(rows)


def choose_k(results: pd.DataFrame) -> int:
    """
    Choose k by prioritizing:
    1) higher silhouette
    2) higher CH
    3) lower DB
    """
    scored = results.copy()
    scored = scored.dropna(subset=["silhouette"])
    if scored.empty:
        return int(results.iloc[0]["k"])
    scored = scored.sort_values(
        by=["silhouette", "ch", "db"],
        ascending=[False, False, True],
    )
    return int(scored.iloc[0]["k"])


def fit_final_model(train_X, k: int, seed: int = 42) -> KMeans:
    km = KMeans(
        n_clusters=k,
        n_init=20,
        max_iter=300,
        tol=1e-4,
        random_state=seed,
        algorithm="elkan",
    )
    km.fit(train_X)
    return km


def nearest_train_neighbors(
    km: KMeans,
    train_X: np.ndarray,
    train_df: pd.DataFrame,
    test_X: np.ndarray,
    test_df: pd.DataFrame,
    test_labels: np.ndarray,
    k: int = 5,
    id_col: str = "Player",
    prefer_same_foot: bool = False,
    prefer_same_side: bool = False,
    max_age_diff: float | None = None,
    exclude_query_from_neighbors: bool = False,
):
    """
    Build per-cluster KNN on the training pool, then provide a callable
    that returns top-k neighbors in train_df for any test index.
    """

    per_cluster = build_per_cluster_knn(
        train_X,
        km.labels_,
        n_neighbors=k + 1,
        metric="cosine",
    )

    def for_test_idx(test_idx: int) -> pd.DataFrame:
        cluster = test_labels[test_idx]
        nn, idx = per_cluster[int(cluster)]
        n_q = min(k * 2, len(idx))  # grab more to allow post-filters

        dists, inds = nn.kneighbors(
            test_X[test_idx].reshape(1, -1),
            n_neighbors=n_q,
        )
        global_inds = idx[inds[0]]

        # Columns to carry through for inspection and later filters
        cols_to_show = [
            c
            for c in [
                id_col,
                "Rk",
                "Pos",
                "Squad",
                "Comp",
                "foot",
                "Age",
                "Nation",
                "numeric_wage",
            ]
            if c in train_df.columns
        ]
        out = train_df.iloc[global_inds][cols_to_show].copy()
        out["distance"] = dists[0]
        out["cluster"] = cluster

        query = test_df.iloc[test_idx]

        # Optionally drop the query itself from its own neighbor list
        if exclude_query_from_neighbors:
            self_mask = np.ones(len(out), dtype=bool)
            q_idx = getattr(query, "name", None)
            if q_idx is not None:
                self_mask &= out.index != q_idx
            if id_col in out.columns and id_col in query.index:
                qid = str(query[id_col]).lower()
                self_mask &= out[id_col].astype(str).str.lower() != qid
            out = out[self_mask]

        # Optional post-hoc filters
        if prefer_same_foot and "foot" in train_df.columns and "foot" in query.index:
            qf = str(query["foot"]).lower()
            if "foot" in out.columns:
                out = out[out["foot"].astype(str).str.lower() == qf]

        if prefer_same_side and "Pos" in train_df.columns and "Pos" in query.index:

            def side(val):
                if not isinstance(val, str):
                    return None
                val = val.upper()
                if "L" in val:
                    return "L"
                if "R" in val:
                    return "R"
                return None

            qs = side(query["Pos"])
            if qs is not None and "Pos" in out.columns:
                out = out[out["Pos"].apply(side) == qs]

        if max_age_diff is not None and "Age" in train_df.columns and "Age" in query.index:
            try:
                qa = float(query["Age"])
                if "Age" in out.columns:
                    out = out[abs(out["Age"].astype(float) - qa) <= max_age_diff]
            except (TypeError, ValueError):
                pass

        out = out.head(k)
        return out.reset_index(drop=True)

    return for_test_idx


def knn_reciprocity_stats(X: np.ndarray, labels: np.ndarray, k: int = 5) -> dict:
    """
    Debug utility: simple KNN graph stats on a given split (uses that split
    as both query and pool). Not needed for the main recommender,
    but useful for geometry inspection.
    """
    per_cluster = build_per_cluster_knn(X, labels, n_neighbors=k + 1, metric="cosine")
    total = 0
    mutual = 0
    mean_kth_dist: list[float] = []

    for _, (nn, idx) in per_cluster.items():
        if len(idx) < 2:
            continue
        n_q = min(k + 1, len(idx))
        dists, inds = nn.kneighbors(X[idx], n_neighbors=n_q)

        for row_idx, (row_inds, row_dists) in enumerate(zip(inds, dists)):
            global_inds = idx[row_inds]
            mask = global_inds != idx[row_idx]
            neighs = global_inds[mask][:k]
            neigh_dists = row_dists[mask][:k]
            if len(neighs) == 0:
                continue
            total += len(neighs)
            mean_kth_dist.append(neigh_dists[-1])

            for n in neighs:
                n_neighbors = idx[
                    inds[nn.kneighbors(X[n].reshape(1, -1), n_neighbors=n_q)[1][0]]
                ]
                if idx[row_idx] in n_neighbors[1:]:  # exclude self at [0]
                    mutual += 1

    reciprocity = mutual / total if total else np.nan
    return {
        "reciprocity": reciprocity,
        "mean_kth_dist": float(np.mean(mean_kth_dist)) if mean_kth_dist else np.nan,
    }


def self_hit_rate(
    X: np.ndarray,
    labels: np.ndarray,
    k: int = 5,
    eps: float = 1e-9,
) -> float:
    """
    Debug utility: leave-one-out self hit.
    For each point, query within its cluster excluding itself,
    count a hit if the nearest neighbor is effectively identical (distance <= eps).
    Useful to flag collapses/duplicates.
    """
    per_cluster = build_per_cluster_knn(X, labels, n_neighbors=k + 1, metric="cosine")
    hits = 0
    total = 0

    for _, (nn, idx) in per_cluster.items():
        if len(idx) < 2:
            continue
        n_q = min(k + 1, len(idx))
        dists, inds = nn.kneighbors(X[idx], n_neighbors=n_q)

        for row_idx, (row_inds, row_dists) in enumerate(zip(inds, dists)):
            global_inds = idx[row_inds]
            mask = global_inds != idx[row_idx]
            neighs = global_inds[mask][:k]
            neigh_dists = row_dists[mask][:k]
            if len(neighs) == 0:
                continue
            total += 1
            if neigh_dists[0] <= eps:
                hits += 1

    return hits / total if total else np.nan


def knn_distance_baselines(
    X: np.ndarray,
    labels: np.ndarray,
    df: pd.DataFrame,
    query_idx: int,
    k: int = 10,
    n_random: int = 500,
    seed: int = 42,
) -> dict:
    """
    Compute simple distance-based baselines for a single query player.

    X         : feature matrix used for clustering/KNN (same as used for KMeans).
    labels    : cluster labels for X (e.g., km_full.labels_).
    df        : dataframe aligned with X (must contain 'Pos').
    query_idx : integer index of the query row in X/df.
    k         : number of nearest neighbours for the "real" KNN distance.
    n_random  : number of random samples per baseline.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    if query_idx < 0 or query_idx >= n:
        raise IndexError(f"query_idx {query_idx} out of bounds for X with shape {X.shape}")

    query_vec = X[query_idx].reshape(1, -1)
    query_pos = df.loc[query_idx, "Pos"] if "Pos" in df.columns else None
    query_cluster = labels[query_idx]

    # Real KNN neighbors within the same cluster (your actual system)
    per_cluster = build_per_cluster_knn(X, labels, n_neighbors=k + 1, metric="cosine")
    nn, idx = per_cluster[int(query_cluster)]
    n_q = min(k + 1, len(idx))
    dists, inds = nn.kneighbors(query_vec, n_neighbors=n_q)
    global_inds = idx[inds[0]]

    # Drop self if present, then take top-k
    mask_self = global_inds != query_idx
    knn_dists = dists[0][mask_self][:k]
    mean_knn_dist = float(knn_dists.mean()) if len(knn_dists) else float("nan")

    def random_distances(candidates: np.ndarray) -> float:
        candidates = candidates[candidates != query_idx]
        if len(candidates) == 0:
            return float("nan")
        size = min(n_random, len(candidates))
        replace = len(candidates) < n_random
        sample_idx = rng.choice(candidates, size=size, replace=replace)
        rand_dists = cosine_distances(query_vec, X[sample_idx])[0]
        return float(rand_dists.mean())

    # Random players from same cluster
    same_cluster_idx = np.where(labels == query_cluster)[0]
    mean_rand_same_cluster = random_distances(same_cluster_idx)

    # Random players with same position (if available)
    if query_pos is not None:
        same_pos_idx = df.index[df["Pos"] == query_pos].to_numpy()
        mean_rand_same_pos = random_distances(same_pos_idx)
    else:
        mean_rand_same_pos = float("nan")

    # Random players from entire pool
    all_idx = np.arange(n)
    mean_rand_global = random_distances(all_idx)

    return {
        "mean_knn_dist": mean_knn_dist,
        "mean_rand_same_cluster": mean_rand_same_cluster,
        "mean_rand_same_pos": mean_rand_same_pos,
        "mean_rand_global": mean_rand_global,
        "cluster": int(query_cluster),
        "pos": query_pos,
    }


def _embed_2d(train_X: np.ndarray, test_X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ensure 2D embeddings for plotting; pad with zeros if only 1 feature."""
    if train_X.shape[1] >= 2:
        return train_X[:, :2], test_X[:, :2]
    # pad one extra dimension of zeros
    def pad(arr: np.ndarray) -> np.ndarray:
        if arr.shape[0] == 0:
            return np.zeros((0, 2))
        z = np.zeros((arr.shape[0], 1))
        return np.hstack([arr, z])

    return pad(train_X), pad(test_X)


def save_cluster_plot(
    train_X: np.ndarray,
    test_X: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    pos: str,
    path: Path,
):
    train_emb, test_emb = _embed_2d(train_X, test_X)
    unique_labels = np.unique(np.concatenate([train_labels, test_labels]))
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 6))

    for lbl in unique_labels:
        mask_tr = train_labels == lbl
        if mask_tr.any():
            ax.scatter(
                train_emb[mask_tr, 0],
                train_emb[mask_tr, 1],
                s=18,
                alpha=0.6,
                color=cmap(int(lbl) % 10),
                label=f"Train c{lbl}",
            )
        mask_te = test_labels == lbl
        if mask_te.any():
            ax.scatter(
                test_emb[mask_te, 0],
                test_emb[mask_te, 1],
                s=28,
                alpha=0.9,
                marker="x",
                color=cmap(int(lbl) % 10),
                label=f"Test c{lbl}",
            )

    ax.set_title(f"{pos} clusters (train circles, test x)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved cluster plot to {path}")

def run_position_for_testing(
    pos: str,
    k: int,
    seed: int = 42,
    with_pca: float | bool = 0.95,
    use_groups=None,
):
    """
    Lightweight version of run_position() that:
      - loads data
      - does train/val/test split
      - builds numeric masks & PCA
      - fits final model with provided k
      - returns exactly the data needed for tests:
            X_train, X_test, train_df, test_df, labels_train, labels_test, model
    """
    df = load_position_df(pos)
    train_df, val_df, test_df = stratified_split(df, seed=seed)

    allowed_numeric = select_group_columns(train_df, use_groups)
    numeric_cols = feature_mask_from_train(train_df, allowed_numeric=allowed_numeric)

    feats = fit_transforms(
        train_df, val_df, test_df,
        numeric_cols,
        with_pca=with_pca,
        seed=seed,
    )

    X_train = feats["X_train"]
    X_test = feats["X_test"]

    model = fit_final_model(X_train, k, seed=seed)

    labels_train = model.labels_
    labels_test = model.predict(X_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "train_df": train_df,
        "test_df": test_df,
        "labels_train": labels_train,
        "labels_test": labels_test,
        "model": model,
    }



def run_position(
    pos: str = "FW",
    k_grid: tuple[int, ...] = (2, 3, 4),
    max_missing: float = 0.4,
    min_variance: float = 1e-6,
    corr_thresh: float = 0.9,
    with_pca: float | bool = 0.95,
    with_pca_grid: tuple[float | bool, ...] | None = None,
    seed: int = 42,
    use_groups=None,
    group_presets=None,
    include_pca_top: bool = False,
    pca_top_n: int = 15,
    example_players=None,
    recommend_players=None,
    compute_graph_stats: bool = False,
    plot_clusters: bool = False,
    plot_path: str | None = None,
    plot_all_pca: bool = False,
):
    df = load_position_df(pos)

    if isinstance(example_players, str):
        example_players = [example_players]
    if isinstance(recommend_players, str):
        recommend_players = [recommend_players]

    # Leak-free split (all fitting on train only)
    train_df, val_df, test_df = stratified_split(df, seed=seed)

    combos = group_presets if group_presets else [use_groups]
    if include_pca_top and "pca_top" not in combos:
        combos = list(combos) + ["pca_top"]

    results_summary = []
    best_combo = None
    best_row = None
    best_state = None
    runs_for_plot = []
    
    # -------- Model selection phase: choose groups + k ----------
    for combo in combos:
        if combo == "pca_top":
            base_allowed = select_group_columns(train_df, use_groups) if use_groups else None
            allowed_numeric = select_top_loading_cols(train_df, base_allowed, top_n=pca_top_n)
        else:
            allowed_numeric = select_group_columns(train_df, combo)

        numeric_cols = feature_mask_from_train(
            train_df,
            max_missing=max_missing,
            min_variance=min_variance,
            corr_thresh=corr_thresh,
            allowed_numeric=allowed_numeric,
        )

        pca_options = list(with_pca_grid) if with_pca_grid is not None else [with_pca]
        for pca_opt in pca_options:
            try:
                feats = fit_transforms(
                    train_df,
                    val_df,
                    test_df,
                    numeric_cols,
                    with_pca=pca_opt,
                    seed=seed,
                )
            except ValueError as e:
                print(f"Skipping combo {combo} with_pca={pca_opt}: {e}")
                continue

            group_msg = f"groups={combo}" if combo else "groups=all"
            print(f"\n{pos}: kept {len(numeric_cols)} numeric cols; with_pca={pca_opt}; {group_msg}")
            val_results = sweep_k(feats["X_train"], feats["X_val"], val_df, k_grid, seed=seed)
            val_results["with_pca"] = pca_opt
            print("Validation metrics:")
            print(val_results)

            best_k = choose_k(val_results)
            chosen_row = val_results[val_results["k"] == best_k].iloc[0]
            results_summary.append(
                {
                    "groups": combo,
                    "with_pca": pca_opt,
                    "best_k": best_k,
                    "silhouette": chosen_row["silhouette"],
                }
            )

            if best_row is None or chosen_row["silhouette"] > best_row["silhouette"]:
                best_row = chosen_row
                best_combo = combo
                best_state = (numeric_cols, feats, best_k, group_msg, pca_opt)
            if plot_clusters and plot_all_pca:
                runs_for_plot.append(
                    {
                        "numeric_cols": numeric_cols,
                        "feats": feats,
                        "best_k": best_k,
                        "group_msg": group_msg,
                        "pca_opt": pca_opt,
                        "combo": combo,
                    }
                )

    # -------- Final clustering on train split, test evaluation ----------
    if best_state is None:
        raise RuntimeError("No valid feature sets found; relax filtering thresholds or adjust group selections.")
    numeric_cols, feats, best_k, group_msg, best_with_pca = best_state
    print(f"\nSelected combo: {group_msg} with k={best_k}; with_pca={best_with_pca}")
    km = fit_final_model(feats["X_train"], best_k, seed=seed)
    test_labels = km.predict(feats["X_test"])

    if len(np.unique(test_labels)) > 1:
        test_sil = silhouette_score(feats["X_test"], test_labels)
        test_db = davies_bouldin_score(feats["X_test"], test_labels)
        test_ch = calinski_harabasz_score(feats["X_test"], test_labels)
    else:
        test_sil = test_db = test_ch = np.nan

    test_knn_stats = {"reciprocity": np.nan, "mean_kth_dist": np.nan}
    test_self_hit = np.nan
    if compute_graph_stats:
        test_knn_stats = knn_reciprocity_stats(feats["X_test"], test_labels, k=10)
        test_self_hit = self_hit_rate(feats["X_test"], test_labels, k=10)

    print("\nTest metrics:")
    print(
        {
            "silhouette": float(test_sil) if test_sil == test_sil else np.nan,
            "db": float(test_db) if test_db == test_db else np.nan,
            "ch": float(test_ch) if test_ch == test_ch else np.nan,
            **test_knn_stats,
            "self_hit_at_10": test_self_hit,
        }
    )

    if plot_clusters:
        try:
            plot_out = Path(plot_path) if plot_path else Path(__file__).resolve().parents[2] / "plots" / f"{pos}_clusters.png"
            save_cluster_plot(feats["X_train"], feats["X_test"], km.labels_, test_labels, pos, plot_out)
        except Exception as e:
            print(f"Could not plot clusters: {e}")

    if plot_clusters and plot_all_pca and runs_for_plot:
        base_dir = Path(plot_path).parent if plot_path else Path(__file__).resolve().parents[2] / "plots"
        for run in runs_for_plot:
            try:
                km_plot = fit_final_model(run["feats"]["X_train"], run["best_k"], seed=seed)
                test_labels_plot = km_plot.predict(run["feats"]["X_test"])
                combo_tag = "all" if run["combo"] is None else "_".join(run["combo"])
                pca_tag = str(run["pca_opt"]).replace(".", "p")
                fname = f"{pos}_{combo_tag}_pca{pca_tag}_k{run['best_k']}.png"
                plot_out = base_dir / fname
                save_cluster_plot(run["feats"]["X_train"], run["feats"]["X_test"], km_plot.labels_, test_labels_plot, pos, plot_out)
            except Exception as e:
                print(f"Could not plot combo {run['combo']} with_pca={run['pca_opt']}: {e}")

                
    # -------- In-split neighbor inspection (test -> train) ----------
    neighbors_fn = nearest_train_neighbors(
        km,
        feats["X_train"],
        train_df,
        feats["X_test"],
        test_df,
        test_labels,
        k=5,
        prefer_same_foot=False,
        prefer_same_side=False,
        max_age_diff=None,
    )

    if len(test_df) > 0:
        if example_players and "Player" not in test_df.columns:
            print("\nCannot show named examples because 'Player' column is missing.")
        elif example_players:
            for name in example_players:
                matches = test_df.index[test_df["Player"].str.lower() == str(name).lower()]
                if len(matches) == 0:
                    print(f"\nNo test player named '{name}' found.")
                    continue
                for idx in matches:
                    example = neighbors_fn(int(idx))
                    label = test_df.loc[idx, "Player"]
                    print(f"\nNearest train neighbors for test player '{label}':")
                    print(example)
        else:
            example = neighbors_fn(0)
            label = test_df.iloc[0]["Player"] if "Player" in test_df.columns else "index 0"
            print(f"\nNearest train neighbors for first test player ({label}):")
            print(example)

    # -------- Full-pool recommender (train on entire position subset) ----------
    if recommend_players:
        if "Player" not in df.columns:
            print("\nCannot build recommendations because 'Player' column is missing.")
        else:
            full_num = df[numeric_cols]
            full_imputer = SimpleImputer(strategy="median")
            full_scaler = StandardScaler()

            full_imp = full_imputer.fit_transform(full_num)
            full_scaled = full_scaler.fit_transform(full_imp)

            full_pca = None
            if best_with_pca:
                full_pca = PCA(
                    n_components=best_with_pca,
                    svd_solver="full",
                    random_state=seed,
                )
                full_scaled = full_pca.fit_transform(full_scaled)

            km_full = fit_final_model(full_scaled, best_k, seed=seed)
            full_labels = km_full.labels_

            recommend_neighbors = nearest_train_neighbors(
                km_full,
                full_scaled,
                df,
                full_scaled,
                df,
                full_labels,
                k=10,
                prefer_same_foot=False,
                prefer_same_side=False,
                max_age_diff=None,
                exclude_query_from_neighbors=True,
            )

            for name in recommend_players:
                matches = df.index[df["Player"].str.lower() == str(name).lower()]
                if len(matches) == 0:
                    print(f"\nNo player named '{name}' found in this position's data.")
                    continue
                for idx in matches:
                    recs = recommend_neighbors(int(idx))
                    label = df.loc[idx, "Player"]
                    print(f"\nRecommended train neighbors for '{label}' (full pool):")
                    print(recs)

                    eval_stats = knn_distance_baselines(
                        full_scaled,
                        full_labels,
                        df,
                        query_idx=int(idx),
                        k=10,
                        n_random=500,
                        seed=seed,
                    )
                    print("\nKNN distance baselines for this query:")
                    print(eval_stats)

    return {
        "val_results": results_summary,
        "best_combo": best_combo,
        "best_k": best_k,
        "best_with_pca": best_with_pca,
        "test_metrics": {
            "silhouette": test_sil,
            "db": test_db,
            "ch": test_ch,
            **test_knn_stats,
            "self_hit_at_10": test_self_hit,
        },
    }

def run_knn_evaluation(
    pos: str = "FW",
    k_neighbors: int = 10,
    n_queries: int = 100,
    seed: int = 42,
    with_pca: float | bool = 0.95,
    use_groups=None,
):
    """
    End-to-end KNN evaluation on a held-out test set.

    Steps:
      1) Load position-specific data and do leak-free train/val/test split.
      2) Build numeric feature mask on train and fit transforms (imputer, scaler, PCA).
      3) Fit KMeans on train only, get cluster labels for train and test.
      4) For N random test players:
           - Compute mean distance to their k nearest train neighbors (within same cluster).
           - Compute mean distance to k random train players:
               a) from same cluster
               b) from same base position (i.e., any train row)
           - Record distances and ratios.
      5) Return a summary dict + per-query DataFrame for reporting.
    """
    # 1) Load and split
    df = load_position_df(pos)
    train_df, val_df, test_df = stratified_split(df, seed=seed)

    # 2) Feature selection and transforms (no PCA grid search here; single setting)
    allowed_numeric = select_group_columns(train_df, use_groups)
    numeric_cols = feature_mask_from_train(train_df, allowed_numeric=allowed_numeric)

    feats = fit_transforms(
        train_df,
        val_df,
        test_df,
        numeric_cols,
        with_pca=with_pca,
        seed=seed,
    )

    X_train = feats["X_train"]
    X_test = feats["X_test"]

    # 3) Fit KMeans on train, get labels
    #    You can choose k here or reuse a fixed small grid; for simplicity pick a reasonable default.
    #    Option 1: fixed k (e.g., 4); Option 2: small sweep like in run_position.
    #    Here we do a tiny sweep over k_grid=(3,4) on val.
    k_grid = (3, 4)
    val_results = sweep_k(feats["X_train"], feats["X_val"], val_df, k_grid, seed=seed)
    best_k = choose_k(val_results)

    km = fit_final_model(X_train, best_k, seed=seed)
    labels_train = km.labels_
    labels_test = km.predict(X_test)

    # Build per-cluster KNN on TRAIN (cosine distance)
    per_cluster = build_per_cluster_knn(
        X_train,
        labels_train,
        n_neighbors=k_neighbors + 1,
        metric="cosine",
    )

    rng = np.random.default_rng(seed)
    n_test = X_test.shape[0]
    if n_test == 0:
        raise RuntimeError("Empty test set; cannot run KNN evaluation.")

    query_indices = rng.choice(
        np.arange(n_test),
        size=min(n_queries, n_test),
        replace=False,
    )

    rows = []

    for local_idx in query_indices:
        cluster = labels_test[local_idx]
        if int(cluster) not in per_cluster:
            continue

        nn, train_idx = per_cluster[int(cluster)]

        # --- true KNN distances: test -> train within same cluster ---
        n_q = min(k_neighbors + 1, len(train_idx))
        dists, inds = nn.kneighbors(
            X_test[local_idx].reshape(1, -1),
            n_neighbors=n_q,
        )
        global_inds = train_idx[inds[0]]

        # Drop any accidental exact duplicate if it exists in train with same index
        # (unlikely here because pools differ, but keep consistent with other utilities).
        knn_dists = dists[0][:k_neighbors]
        if len(knn_dists) == 0:
            continue

        mean_knn_dist = float(knn_dists.mean())

        # --- random baselines from train ---

        # a) random from same cluster (train side)
        same_cluster_train_idx = np.where(labels_train == cluster)[0]
        same_cluster_train_idx = same_cluster_train_idx[
            np.isin(same_cluster_train_idx, train_idx)
        ]
        if len(same_cluster_train_idx) > 0:
            size_sc = min(k_neighbors, len(same_cluster_train_idx))
            rand_sc = rng.choice(
                same_cluster_train_idx,
                size=size_sc,
                replace=len(same_cluster_train_idx) < k_neighbors,
            )
            rand_sc_dists = cosine_distances(
                X_test[local_idx].reshape(1, -1),
                X_train[rand_sc],
            )[0]
            mean_rand_same_cluster = float(rand_sc_dists.mean())
        else:
            mean_rand_same_cluster = float("nan")

        # b) random from entire train pool (same base position)
        all_train_idx = np.arange(X_train.shape[0])
        size_gl = min(k_neighbors, len(all_train_idx))
        rand_gl = rng.choice(
            all_train_idx,
            size=size_gl,
            replace=len(all_train_idx) < k_neighbors,
        )
        rand_gl_dists = cosine_distances(
            X_test[local_idx].reshape(1, -1),
            X_train[rand_gl],
        )[0]
        mean_rand_global = float(rand_gl_dists.mean())

        # --- ratios ---
        row = {
            "test_idx": int(local_idx),
            "player": (
                test_df.loc[local_idx, "Player"]
                if "Player" in test_df.columns
                else None
            ),
            "cluster": int(cluster),
            "mean_knn_dist": mean_knn_dist,
            "mean_rand_same_cluster": mean_rand_same_cluster,
            "mean_rand_global": mean_rand_global,
            "same_cluster_ratio": (
                mean_rand_same_cluster / mean_knn_dist
                if not np.isnan(mean_rand_same_cluster)
                else np.nan
            ),
            "global_ratio": mean_rand_global / mean_knn_dist,
        }
        rows.append(row)

    results_df = pd.DataFrame(rows)

    # Aggregate summary for the report
    summary = {
        "pos": pos,
        "k_neighbors": k_neighbors,
        "best_k_clusters": best_k,
        "n_queries_effective": int(len(results_df)),
        "mean_knn_dist": float(results_df["mean_knn_dist"].mean())
        if not results_df.empty
        else float("nan"),
        "mean_same_cluster_ratio": float(results_df["same_cluster_ratio"].mean())
        if "same_cluster_ratio" in results_df.columns and not results_df.empty
        else float("nan"),
        "mean_global_ratio": float(results_df["global_ratio"].mean())
        if "global_ratio" in results_df.columns and not results_df.empty
        else float("nan"),
    }

    print("\nKNN evaluation summary (test -> train):")
    print(summary)

    return {
        "summary": summary,
        "per_query": results_df,
        "val_results": val_results.to_dict(orient="records"),
    }



if __name__ == "__main__":
    # run_position(
    #     pos="FW",
    #     k_grid=(3, 4),
    #     with_pca=0.6,
    #     include_pca_top=True,
    #     recommend_players=["Lamine Yamal"],
    #     pca_top_n=15,
    #     group_presets=[
    #         None,
    #         ["goal_shot_creation"],
    #         ["passing", "goal_shot_creation"],
    #         ["passing", "goal_shot_creation", "pass_types", "possession"],
    #         ["passing", "goal_shot_creation", "pass_types", "possession", "defense", "misc"],
    #     ],
    #     compute_graph_stats=False
    # )
    # run_position(
    #     pos="MF",
    #     k_grid=(4, 5),
    #     with_pca=0.6,
    #     include_pca_top=True,
    #     recommend_players=["Pedri"],
    #     pca_top_n=15,
    #     group_presets=[
    #         None,
    #         ["passing", "chance_creation"],
    #         ["passing", "chance_creation", "possession"],
    #         ["passing", "chance_creation", "possession", "defense", "misc"],
    #     ],
    #     compute_graph_stats=False
    # )
#     run_position(
#     pos="FW",
#     group_presets=[
#             None,
#             ["passing", "goal_shot_creation", "pass_types", "possession"],
#         ],
#     k_grid=(3,4),
#     with_pca_grid=(2,3),
#     plot_clusters=True,
#     plot_all_pca=True,  # saves plots for each tested PCA preset
#     include_pca_top=True,
#     pca_top_n=32,
#     recommend_players=["Lamine Yamal"],
#     compute_graph_stats=True
# )
    
    eval_fw = run_knn_evaluation(
        pos="FW",
        k_neighbors=10,
        n_queries=100,
        seed=42,
        with_pca=0.95,
        use_groups=["passing", "goal_shot_creation", "pass_types", "possession"],
    )



    # run_position(
    #     pos="GK",
    #     k_grid=(2, 3),
    #     with_pca=0.6,
    #     include_pca_top=True,
    #     recommend_players=["Thibaut Courtois"],
    #     pca_top_n=15,
    #     group_presets=[
    #         None,
    #         ["goalkeeping_shots"],
    #         ["goalkeeping_shots", "goalkeeping_distribution"],
    #         ["goalkeeping_shots", "goalkeeping_distribution", "goalkeeping_misc"],
    #     ],
    #     compute_graph_stats=False
    # )
