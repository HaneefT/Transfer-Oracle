import json
import math
from pathlib import Path

import pandas as pd
import numpy as np

from src.clustering.clustering import (
    run_position,
    load_position_df,
    fit_final_model,
    nearest_train_neighbors,
)

# -------------------------------------------------------------------
# TOP 6 FEATURES PER POSITION (for radar)
# -------------------------------------------------------------------
TOP_RADAR_FEATURES = {
    "FW": ["G+A-PK", "xG+xAG", "Sh/90", "Att 3rd_stats_possession", "G/Sh", "SoT/90"],
    "MF": ["PrgP", "PrgC", "xA", "Mid 3rd_stats_possession", "Tkl+Int", "G+A-PK"],
    "DF": ["Tkl+Int", "Clr", "Def 3rd_stats_possession", "PrgP", "PrgDist_stats_possession", "Cmp%"],
    "GK": ["Save%", "/90", "CS%", "Stp%", "#OPA/90", "Cmp%_stats_keeper_adv"],
}


# -------------------------------------------------------------------
# CLEANING FUNCTION — ensures valid JSON (no NaN, no Infinity)
# -------------------------------------------------------------------
def clean_nan(obj):
    """Recursively replaces NaN / Inf / -Inf with None for valid JSON."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [clean_nan(x) for x in obj]

    return obj


# -------------------------------------------------------------------
# BUILD A CLEAN DICTIONARY FROM MODEL OUTPUT
# -------------------------------------------------------------------
def build_rec_map(df, neighbors_fn, labels, pos, radar_features, feature_norms):
    """
    Produces a dictionary:
        {
          player_name: {
            "neighbors": [...],
            "eval_stats": {...},
            "radar_features": {
                feat_name: { "raw": float | None, "norm": float | None }
            },
            "pos": "FW"/"MF"/...
          }
        }

    Distance is scaled by 1000 so that values are more visible
    when formatted to 3 decimal places in the UI.
    """

    rec_map = {}

    for idx, row in df.iterrows():
        name = row["Player"]

        # ----------------- Nearest neighbors -----------------
        neigh_df = neighbors_fn(int(idx))

        neighbors_list = []
        for _, nrow in neigh_df.iterrows():
            raw_dist = nrow.get("distance", np.nan)
            try:
                d = float(raw_dist)
            except (TypeError, ValueError):
                d = float("nan")

            # Scale distance so it's not always ~0.000 when rounded
            if not math.isnan(d):
                d_scaled = d * 1000.0
            else:
                d_scaled = d

            neighbors_list.append(
                {
                    "Player": nrow.get("Player"),
                    "Pos": nrow.get("Pos"),
                    "Squad": nrow.get("Squad"),
                    "Comp": nrow.get("Comp"),
                    "distance": d_scaled,
                }
            )

        # Simple distance baselines (optional)
        dist_values = [
            n["distance"] for n in neighbors_list if not math.isnan(n["distance"])
        ]
        mean_dist = float(np.mean(dist_values)) if dist_values else None

        eval_stats = {
            "mean_knn_dist": mean_dist
        }

        # ----------------- Radar feature payload -----------------
        radar_payload = {}
        for feat in radar_features:
            if feat not in df.columns:
                continue

            raw_val = df.at[idx, feat]
            try:
                raw_f = float(raw_val)
            except (TypeError, ValueError):
                raw_f = float("nan")

            if math.isnan(raw_f) or math.isinf(raw_f):
                raw_clean = None
            else:
                raw_clean = raw_f

            norm_val = None
            series = feature_norms.get(feat)
            if raw_clean is not None and series is not None:
                # same index as df
                s_val = series.get(idx)
                if pd.notna(s_val):
                    norm_val = float(s_val)

            radar_payload[feat] = {
                "raw": raw_clean,
                "norm": norm_val,
            }

        rec_map[name] = {
            "neighbors": neighbors_list,
            "eval_stats": eval_stats,  # snake_case to match App.jsx
            "radar_features": radar_payload,
            "pos": str(pos),
        }

    return rec_map


# -------------------------------------------------------------------
# MAIN FUNCTION: PRECOMPUTE FOR ONE POSITION
# -------------------------------------------------------------------
def precompute_for_position(pos):
    print(f"\n=== Running model for position: {pos} ===")

    # Use new run_position, but request artifacts so we can reuse numeric_cols
    result = run_position(
        pos=pos,
        k_grid=(3, 4),
        with_pca_grid=(False, 2, 3),
        include_pca_top=False,
        compute_graph_stats=False,
        recommend_players=None,
        plot_clusters=False,
        plot_all_pca=False,
        return_artifacts=True,
        plot_val_metrics=False,
    )

    artifacts = result.get("artifacts")
    if artifacts is None:
        raise RuntimeError(
            "run_position did not return artifacts. "
            "Ensure return_artifacts=True in precompute_for_position."
        )

    # Load processed data again (same logic as run_position)
    df = load_position_df(pos)

    # For full-pool KMeans, we need the numeric_cols chosen during model selection
    numeric_cols = artifacts["numeric_cols"]
    best_with_pca = result["best_with_pca"]
    best_k = result["best_k"]

    # ----------------- Full-pool preparation -----------------
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    full_num = df[numeric_cols]
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    full_imp = imputer.fit_transform(full_num)
    full_scaled = scaler.fit_transform(full_imp)

    if best_with_pca:
        pca = PCA(n_components=best_with_pca, svd_solver="full")
        full_scaled = pca.fit_transform(full_scaled)

    km_full = fit_final_model(full_scaled, best_k)
    full_labels = km_full.labels_

    neighbors_fn = nearest_train_neighbors(
        km_full,
        full_scaled,
        df,
        full_scaled,
        df,
        full_labels,
        k=10,
        exclude_query_from_neighbors=True,
    )

    # ----------------- Radar feature percentiles (0–1 per feature) -----------------
    radar_features = TOP_RADAR_FEATURES.get(pos, [])
    feature_norms: dict[str, pd.Series] = {}

    for feat in radar_features:
        if feat not in df.columns:
            continue
        col = pd.to_numeric(df[feat], errors="coerce")
        col = col.replace([np.inf, -np.inf], np.nan)

        if col.notna().sum() == 0:
            feature_norms[feat] = None
            continue

        # percentile rank in [0,1]
        ranks = col.rank(method="average", pct=True)
        feature_norms[feat] = ranks

    # ----------------- Build recommendations + radar features -----------------
    print(f"Building recommendation dictionary for {pos} ...")
    rec_map = build_rec_map(
        df,
        neighbors_fn,
        full_labels,
        pos,
        radar_features,
        feature_norms,
    )

    # Clean NaN / Infinity → None
    rec_map_cleaned = clean_nan(rec_map)

    output_dir = Path("src/Frontend/transfer-oracle-ui/src/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{pos}_recs.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rec_map_cleaned, f, indent=2, ensure_ascii=False)

    print(f"✔ Saved: {out_path}")


# -------------------------------------------------------------------
# RUN ALL POSITIONS
# -------------------------------------------------------------------
if __name__ == "__main__":
    for pos in ["FW", "MF", "DF", "GK"]:
        precompute_for_position(pos)

    print("\n🎉 All recommendation files generated cleanly!")
