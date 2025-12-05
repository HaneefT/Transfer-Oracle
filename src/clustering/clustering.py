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

# ---------------------------------------------------------------------
# Config / constants
# ---------------------------------------------------------------------

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

# Try to define a stable project root for saving plots
try:
    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
    # Fallback for notebook contexts
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data" / "processed"
PLOTS_DIR = BASE_DIR / "plots"


# ---------------------------------------------------------------------
# Data + feature helpers
# ---------------------------------------------------------------------


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
    selected = list(dict.fromkeys(selected))
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


# ---------------------------------------------------------------------
# KNN helpers
# ---------------------------------------------------------------------


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


def self_hit_rate(
    X: np.ndarray,
    labels: np.ndarray,
    k: int = 5,
    eps: float = 1e-9,
) -> float:
    """
    Leave-one-out self hit: for each point, query within its cluster excluding itself,
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

    Used mainly for illustrative case studies (e.g., Lamine Yamal).
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    if query_idx < 0 or query_idx >= n:
        raise IndexError(f"query_idx {query_idx} out of bounds for X with shape {X.shape}")

    query_vec = X[query_idx].reshape(1, -1)
    query_pos = df.loc[query_idx, "Pos"] if "Pos" in df.columns else None
    query_cluster = labels[query_idx]

    per_cluster = build_per_cluster_knn(X, labels, n_neighbors=k + 1, metric="cosine")
    nn, idx = per_cluster[int(query_cluster)]
    n_q = min(k + 1, len(idx))
    dists, inds = nn.kneighbors(query_vec, n_neighbors=n_q)
    global_inds = idx[inds[0]]

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

    same_cluster_idx = np.where(labels == query_cluster)[0]
    mean_rand_same_cluster = random_distances(same_cluster_idx)

    if query_pos is not None:
        same_pos_idx = df.index[df["Pos"] == query_pos].to_numpy()
        mean_rand_same_pos = random_distances(same_pos_idx)
    else:
        mean_rand_same_pos = float("nan")

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


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------


def _embed_2d(train_X: np.ndarray, test_X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ensure 2D embeddings for plotting; pad with zeros if only 1 feature."""
    if train_X.shape[1] >= 2:
        return train_X[:, :2], test_X[:, :2]

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
    highlight_points: list[dict] | None = None,
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

    # Optional annotated highlights (e.g., worst examples)
    if highlight_points:
        for hp in highlight_points:
            set_name = hp.get("set")
            idx = hp.get("index")
            label = str(hp.get("label", ""))
            cluster = hp.get("cluster")
            color = cmap(int(cluster) % 10) if cluster is not None else "red"
            if set_name == "train" and idx is not None and 0 <= idx < len(train_emb):
                x, y = train_emb[idx]
                ax.scatter([x], [y], color=color, s=60, marker="o", edgecolors="black", zorder=5)
                ax.text(x, y, label, fontsize=8, color=color, weight="bold", ha="left", va="bottom")
            elif set_name == "test" and idx is not None and 0 <= idx < len(test_emb):
                x, y = test_emb[idx]
                ax.scatter([x], [y], color=color, s=60, marker="x", zorder=5)
                ax.text(x, y, label, fontsize=8, color=color, weight="bold", ha="left", va="bottom")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved cluster plot to {path}")


# ---------------------------------------------------------------------
# Main clustering + recommendation
# ---------------------------------------------------------------------


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
    include_pca_top: bool = True,
    pca_top_n: int = 15,
    recommend_players=None,
    compute_graph_stats: bool = False,
    plot_clusters: bool = False,
    plot_path: str | None = None,
    plot_all_pca: bool = False,
    return_artifacts: bool = False,
    plot_val_metrics: bool = False,
    val_plot_dir: str | None = None,
):
    df = load_position_df(pos)

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
    # -------- Model selection phase: choose groups + PCA + k ----------
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

    test_self_hit = np.nan
    if compute_graph_stats:
        test_self_hit = self_hit_rate(feats["X_test"], test_labels, k=10)

    print("\nTest metrics:")
    print(
        {
            "silhouette": float(test_sil) if test_sil == test_sil else np.nan,
            "db": float(test_db) if test_db == test_db else np.nan,
            "ch": float(test_ch) if test_ch == test_ch else np.nan,
            "self_hit_at_10": test_self_hit,
        }
    )

    if plot_clusters:
        try:
            cluster_plot_path = Path(plot_path) if plot_path else PLOTS_DIR / f"{pos}_clusters.png"
            save_cluster_plot(
                feats["X_train"],
                feats["X_test"],
                km.labels_,
                test_labels,
                pos,
                cluster_plot_path,
            )
        except Exception as e:
            print(f"Could not plot clusters: {e}")

    if plot_clusters and plot_all_pca and runs_for_plot:
        base_dir = (Path(plot_path).parent if plot_path else PLOTS_DIR)
        for run in runs_for_plot:
            try:
                km_plot = fit_final_model(run["feats"]["X_train"], run["best_k"], seed=seed)
                test_labels_plot = km_plot.predict(run["feats"]["X_test"])
                combo_tag = "all" if run["combo"] is None else "_".join(run["combo"])
                pca_tag = str(run["pca_opt"]).replace(".", "p")
                fname = f"{pos}_{combo_tag}_pca{pca_tag}_k{run['best_k']}.png"
                plot_out = base_dir / fname
                save_cluster_plot(
                    run["feats"]["X_train"],
                    run["feats"]["X_test"],
                    km_plot.labels_,
                    test_labels_plot,
                    pos,
                    plot_out,
                )
            except Exception as e:
                print(f"Could not plot combo {run['combo']} with_pca={run['pca_opt']}: {e}")

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

    result = {
        "val_results": results_summary,
        "best_combo": best_combo,
        "best_k": best_k,
        "best_with_pca": best_with_pca,
        "test_metrics": {
            "silhouette": test_sil,
            "db": test_db,
            "ch": test_ch,
            "self_hit_at_10": test_self_hit,
        },
    }

    if return_artifacts:
        result["artifacts"] = {
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "numeric_cols": numeric_cols,
            "feats": feats,
            "labels_train": km.labels_,
            "labels_test": test_labels,
            "model": km,
        }

    return result


# ---------------------------------------------------------------------
# Error analysis helpers
# ---------------------------------------------------------------------


def _cross_split_distance_baselines(
    train_X: np.ndarray,
    train_labels: np.ndarray,
    train_df: pd.DataFrame,
    test_X: np.ndarray,
    test_labels: np.ndarray,
    test_df: pd.DataFrame,
    k: int = 5,
    n_random: int = 200,
    sample_size: int | None = 150,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compare KNN distances for test queries against random baselines drawn from the train pool.
    Returns a DataFrame with per-query gaps for downstream summarization.
    """
    rng = np.random.default_rng(seed)
    per_cluster = build_per_cluster_knn(train_X, train_labels, n_neighbors=k + 1, metric="cosine")
    query_idx = np.arange(len(test_X))
    if sample_size is not None and sample_size < len(query_idx):
        query_idx = rng.choice(query_idx, size=sample_size, replace=False)

    rows = []

    for idx in query_idx:
        cluster = int(test_labels[idx])
        if cluster not in per_cluster:
            continue
        nn, pool_idx = per_cluster[cluster]
        n_q = min(k, len(pool_idx))
        if n_q == 0:
            continue
        dists, inds = nn.kneighbors(test_X[idx].reshape(1, -1), n_neighbors=n_q)
        knn_dists = dists[0][:k]
        mean_knn = float(knn_dists.mean()) if len(knn_dists) else float("nan")

        def rand_mean(candidates: np.ndarray) -> float:
            if len(candidates) == 0:
                return float("nan")
            size = min(n_random, len(candidates))
            replace = len(candidates) < n_random
            sample_idx = rng.choice(candidates, size=size, replace=replace)
            rand_dists = cosine_distances(test_X[idx].reshape(1, -1), train_X[sample_idx])[0]
            return float(rand_dists.mean())

        same_cluster_idx = np.where(train_labels == cluster)[0]
        rand_same_cluster = rand_mean(same_cluster_idx)

        query_pos = test_df.loc[idx, "Pos"] if "Pos" in test_df.columns else None
        if query_pos is not None:
            same_pos_idx = train_df.index[train_df["Pos"] == query_pos].to_numpy()
            rand_same_pos = rand_mean(same_pos_idx)
        else:
            rand_same_pos = float("nan")

        rand_global = rand_mean(np.arange(train_X.shape[0]))

        rows.append(
            {
                "idx": int(idx),
                "player": str(test_df.loc[idx, "Player"]) if "Player" in test_df.columns else f"idx_{idx}",
                "cluster": cluster,
                "mean_knn_dist": mean_knn,
                "mean_rand_same_cluster": rand_same_cluster,
                "mean_rand_same_pos": rand_same_pos,
                "mean_rand_global": rand_global,
            }
        )

    return pd.DataFrame(rows)


def _summarize_baseline_gaps(df: pd.DataFrame) -> dict:
    """
    Aggregate KNN-vs-random gap statistics for error analysis.
    """
    if df.empty:
        return {
            "n_queries": 0,
            "avg_knn_dist": float("nan"),
            "gap_same_cluster": {"mean": float("nan"), "pct_positive": float("nan")},
            "gap_same_pos": {"mean": float("nan"), "pct_positive": float("nan")},
            "gap_global": {"mean": float("nan"), "pct_positive": float("nan")},
        }

    df = df.copy()
    df["gap_same_cluster"] = df["mean_rand_same_cluster"] - df["mean_knn_dist"]
    df["gap_same_pos"] = df["mean_rand_same_pos"] - df["mean_knn_dist"]
    df["gap_global"] = df["mean_rand_global"] - df["mean_knn_dist"]

    def gap_stats(col):
        vals = df[col].dropna()
        return {
            "mean": float(vals.mean()) if len(vals) else float("nan"),
            "pct_positive": float((vals > 0).mean()) if len(vals) else float("nan"),
        }

    return {
        "n_queries": int(len(df)),
        "avg_knn_dist": float(df["mean_knn_dist"].mean()),
        "gap_same_cluster": gap_stats("gap_same_cluster"),
        "gap_same_pos": gap_stats("gap_same_pos"),
        "gap_global": gap_stats("gap_global"),
    }


def run_error_analysis(
    pos: str = "FW",
    k_grid=(3, 4),
    with_pca=2,
    with_pca_grid: tuple[float | bool, ...] | None = None,
    use_groups=None,
    seed: int = 42,
    k_neighbors: int = 10,
    sample_size: int | None = 150,
    n_random: int = 200,
    plot_outliers: bool = False,
    plot_path: str | None = None,
    top_n_knn_outliers: int = 3,
    outlier_method: str = "gap",
):
    """
    Convenience wrapper to pick the best model for a position, then report
    error-analysis diagnostics:
      - self-hit rate (from run_position)
      - distance gaps vs random baselines for sampled queries
      - worst-case examples for qualitative inspection (by gap or by mean KNN)

    If plot_outliers is True, also saves a 2D scatter with the worst examples annotated.
    """
    # Ensure we also get a base cluster plot out of run_position
    cluster_plot_path = (
        Path(plot_path) if plot_path else PLOTS_DIR / f"{pos}_clusters.png"
    )

    result = run_position(
        pos=pos,
        k_grid=k_grid,
        with_pca=with_pca,
        with_pca_grid=with_pca_grid,
        use_groups=use_groups,
        seed=seed,
        compute_graph_stats=True,
        plot_clusters=True,
        plot_path=str(cluster_plot_path),
        plot_all_pca=False,
        return_artifacts=True,
    )
    artifacts = result.get("artifacts")
    if artifacts is None:
        raise RuntimeError("run_position did not return artifacts; set return_artifacts=True.")

    train_df = artifacts["train_df"]
    test_df = artifacts["test_df"]
    train_X = artifacts["feats"]["X_train"]
    test_X = artifacts["feats"]["X_test"]
    train_labels = artifacts["labels_train"]
    test_labels = artifacts["labels_test"]

    # KNN vs random baselines for held-out queries
    baselines = _cross_split_distance_baselines(
        train_X,
        train_labels,
        train_df,
        test_X,
        test_labels,
        test_df,
        k=k_neighbors,
        n_random=n_random,
        sample_size=sample_size,
        seed=seed,
    )
    baseline_summary = _summarize_baseline_gaps(baselines)

    # Worst cases by weakest cluster gap
    baselines = baselines.copy()
    baselines["gap_same_cluster"] = baselines["mean_rand_same_cluster"] - baselines["mean_knn_dist"]
    worst_examples = []
    neighbors_fn = nearest_train_neighbors(
        artifacts["model"],
        train_X,
        train_df,
        test_X,
        test_df,
        test_labels,
        k=k_neighbors,
    )

    for _, row in baselines.nsmallest(3, "gap_same_cluster").iterrows():
        neighbors = neighbors_fn(int(row["idx"]))
        worst_examples.append(
            {
                "idx": int(row["idx"]),
                "player": row.get("player"),
                "cluster": int(row["cluster"]),
                "gap_same_cluster": float(row["gap_same_cluster"]),
                "mean_knn_dist": float(row["mean_knn_dist"]),
                "neighbors": neighbors.to_dict(orient="records"),
            }
        )

    # Highest mean_knn_dist outliers (regardless of gap)
    worst_by_knn = []
    if not baselines.empty:
        for _, row in baselines.nlargest(top_n_knn_outliers, "mean_knn_dist").iterrows():
            neighbors = neighbors_fn(int(row["idx"]))
            worst_by_knn.append(
                {
                    "idx": int(row["idx"]),
                    "player": row.get("player"),
                    "cluster": int(row["cluster"]),
                    "mean_knn_dist": float(row["mean_knn_dist"]),
                    "neighbors": neighbors.to_dict(orient="records"),
                }
            )

    if plot_outliers:
        highlights = []
        chosen = worst_by_knn if outlier_method == "knn" else worst_examples
        for ex in chosen:
            highlights.append(
                {
                    "set": "test",
                    "index": ex["idx"],
                    "label": ex.get("player"),
                    "cluster": ex.get("cluster"),
                }
            )
        out_path = (
            Path(plot_path)
            if plot_path
            else PLOTS_DIR / f"{pos}_outliers.png"
        )
        try:
            save_cluster_plot(
                train_X,
                test_X,
                train_labels,
                test_labels,
                pos,
                out_path,
                highlight_points=highlights,
            )
        except Exception as e:
            print(f"Could not plot outliers: {e}")

    def collect_stats(ex_list):
        out = []
        numeric_cols_used = artifacts.get("numeric_cols", [])
        meta_cols = [c for c in ["Player", "Pos", "Squad", "Comp"] if c in test_df.columns]
        for ex in ex_list:
            idx = ex["idx"]
            if 0 <= idx < len(test_df):
                cols_to_show = meta_cols + [c for c in numeric_cols_used if c in test_df.columns]
                row = test_df.iloc[idx][cols_to_show].to_dict()
                out.append(
                    {
                        "player": ex.get("player"),
                        "cluster": ex.get("cluster"),
                        "stats": row,
                    }
                )
        return out

    worst_stats_gap = collect_stats(worst_examples)
    worst_stats_knn = collect_stats(worst_by_knn)

    if worst_stats_gap:
        print("\nWorst outlier stats (by cluster gap):")
        for entry in worst_stats_gap:
            print(f"{entry['player']} (cluster {entry['cluster']}):")
            print(entry["stats"])
    if worst_stats_knn:
        print("\nWorst outlier stats (by mean KNN distance):")
        for entry in worst_stats_knn:
            print(f"{entry['player']} (cluster {entry['cluster']}):")
            print(entry["stats"])

    return {
        "model_selection": {
            "best_k": result["best_k"],
            "best_combo": result["best_combo"],
            "best_with_pca": result["best_with_pca"],
        },
        "graph_metrics": result["test_metrics"],
        "baseline_summary": baseline_summary,
        "worst_examples": worst_examples,
        "worst_by_knn": worst_by_knn,
        "worst_stats_gap": worst_stats_gap,
        "worst_stats_knn": worst_stats_knn,
    }


if __name__ == "__main__":
    groups_pos = {
        "FW": ["passing", "goal_shot_creation", "pass_types","misc"],
        "MF": ["passing", "goal_shot_creation", "pass_types", "defensive_actions", "misc"],
        "DF": ["defensive_actions","possession", "misc"],
        "GK": ["goalkeeping", "pass_types"],
    }
    # run_position(
    #     pos="GK",
    #     k_grid=3,
    #     with_pca = 2,
    #     group_presets=[groups_pos["GK"],["goalkeeping","pass_types","misc"],None],
    #     compute_graph_stats=True,
    #     plot_clusters=True,
    #     plot_path=None,
    #     include_pca_top=True,
    #     return_artifacts=False,
    #     plot_val_metrics=True,
    # )
    from pprint import pprint
    for pos in ["FW", "MF", "DF", "GK"]:
        k_vals = (3, 4, 5)
        if pos == "GK":
            k_vals = (2, 3)   
        print(f"\n=== Running error analysis for position: {pos} ===")
        res = run_error_analysis(
            pos=pos,
            use_groups=groups_pos[pos],
            k_grid=k_vals,
            with_pca_grid=(2, 3),
            sample_size=None,
            k_neighbors=10,
            top_n_knn_outliers=3,
            outlier_method="knn",
            plot_outliers=False,
            plot_path=None,
        )
        pprint(res["model_selection"])
        pprint(res["baseline_summary"])
        print("Worst examples by cluster gap:")
        pprint([{k: ex[k] for k in ["player", "gap_same_cluster", "mean_knn_dist"]} for ex in res["worst_examples"]])
        print("Worst examples by mean KNN distance:")
        pprint([{k: ex[k] for k in ["player", "mean_knn_dist"]} for ex in res["worst_by_knn"]])
