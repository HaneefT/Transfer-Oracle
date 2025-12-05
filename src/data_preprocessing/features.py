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

# %%
# imports
import pandas as pd
import numpy as np
import sklearn
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler
from IPython.display import display
# helpers for leak-free split/feature filtering/k-sweep
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)
from sklearn.neighbors import NearestNeighbors

# %%
# load in the features of each base position

def main():
    file_paths  = ['../../data/processed/players_data_processed.parquet', 
                   '../../data/processed/players_data_GK.parquet', 
                   '../../data/processed/players_data_DF.parquet', 
                   '../../data/processed/players_data_MF.parquet', 
                   '../../data/processed/players_data_FW.parquet']

    df, df_gk, df_df, df_mf, df_fw = [pd.read_parquet(file) for file in file_paths]
    base_positions = {'GK':df_gk, 'DF':df_df, 'MF':df_mf, 'FW':df_fw}
    df_df.head()

    categorical_cols = ['Rk','Player','Nation','Pos','Squad','Comp','Age','Born','MP','Starts','Min','90s',
                        'numeric_wage','foot','W','D','L']

    for pos, df_pos in base_positions.items():
        # 1) Ensure Rk is a COLUMN (don’t drop it). If Rk was the index, bring it back:
        if df_pos.index.name == 'Rk' or 'Rk' not in df_pos.columns:
            df_pos = df_pos.reset_index()  # brings index out as a column named 'index' or 'Rk'
            # If it came out as 'index', rename it:
            if 'index' in df_pos.columns and 'Rk' not in df_pos.columns:
                df_pos = df_pos.rename(columns={'index': 'Rk'})

        # 2) Use a clean, contiguous index ONCE for both sides
        df_pos = df_pos.reset_index(drop=True)

        # 3) Split columns
        keep_cats = [c for c in categorical_cols if c in df_pos.columns]
        temp_df = df_pos[keep_cats].copy()

        # numeric = everything else (but drop non-numeric later)
        num_cols = [c for c in df_pos.columns if c not in keep_cats]
        num_df = df_pos[num_cols].apply(pd.to_numeric, errors='coerce')

        # optional: if some numeric cols are entirely NaN after coercion, drop them
        all_nan_cols = num_df.columns[num_df.isna().all()]
        if len(all_nan_cols) > 0:
            num_df = num_df.drop(columns=all_nan_cols)

        scaler = StandardScaler()
        scaled_vals = scaler.fit_transform(num_df.values)  # preserves row count
        scaled_df = pd.DataFrame(scaled_vals, columns=num_df.columns, index=df_pos.index)
        imputer = sklearn.impute.SimpleImputer(strategy='mean')
        scaled_df = pd.DataFrame(imputer.fit_transform(scaled_df), columns=scaled_df.columns, index=scaled_df.index)

        out_df = pd.concat([temp_df, scaled_df], axis=1)
        base_positions[pos] = out_df
        print(pos, "rows:", len(out_df))
        display(out_df.head())


    for pos, df_pos in base_positions.items():
        table = pa.Table.from_pandas(df_pos, preserve_index=False)
        pq.write_table(table, f'../../data/processed/players_data_{pos}_normalized.parquet')


# %%
# Leak-free split/transform/feature-filter helpers for clustering/KNN evaluation

categorical_cols = [
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


def stratified_split(df, val_size=0.15, test_size=0.15, seed=42, stratify_col="Comp"):
    strat_vals = df[stratify_col] if stratify_col in df else None
    train_val, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=strat_vals)
    strat_vals = train_val[stratify_col] if stratify_col in train_val else None
    rel_val = val_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=rel_val, random_state=seed, stratify=strat_vals)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def feature_mask_from_train(
    train_df,
    max_missing=0.4,
    min_variance=1e-6,
    corr_thresh=0.9,
    categorical_cols=categorical_cols,
    allowed_numeric=None,
):
    numeric_cols = [c for c in train_df.columns if c not in categorical_cols]
    if allowed_numeric is not None:
        numeric_cols = [c for c in numeric_cols if c in allowed_numeric]
    num_df = train_df[numeric_cols]

    missing = num_df.isna().mean()
    keep = missing[missing <= max_missing].index.tolist()
    num_df = num_df[keep]

    variances = num_df.var()
    keep = variances[variances >= min_variance].index.tolist()
    num_df = num_df[keep]

    corr = num_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = []
    for col in upper.columns:
        if col in drop_cols:
            continue
        high = upper.index[upper[col] > corr_thresh].tolist()
        drop_cols.extend(high)
    keep = [c for c in num_df.columns if c not in drop_cols]
    return keep


def fit_transforms(train_df, val_df, test_df, numeric_cols, with_pca=None, seed=42):
    if not numeric_cols:
        raise ValueError("No numeric columns to transform; check feature filters or lower thresholds.")
    train_num = train_df[numeric_cols]
    val_num = val_df[numeric_cols]
    test_num = test_df[numeric_cols]

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    train_imp = imputer.fit_transform(train_num)
    val_imp = imputer.transform(val_num)
    test_imp = imputer.transform(test_num)

    train_scaled = scaler.fit_transform(train_imp)
    val_scaled = scaler.transform(val_imp)
    test_scaled = scaler.transform(test_imp)

    pca = None
    if with_pca:
        pca = PCA(n_components=with_pca, svd_solver="full", random_state=seed)
        train_scaled = pca.fit_transform(train_scaled)
        val_scaled = pca.transform(val_scaled)
        test_scaled = pca.transform(test_scaled)

    return {
        "X_train": train_scaled,
        "X_val": val_scaled,
        "X_test": test_scaled,
        "imputer": imputer,
        "scaler": scaler,
        "pca": pca,
    }


def evaluate_k(train_X, val_X, k, seed=42):
    km = KMeans(n_clusters=k, n_init=20, max_iter=300, tol=1e-4, random_state=seed, algorithm="elkan")
    km.fit(train_X)
    val_labels = km.predict(val_X)
    if len(np.unique(val_labels)) > 1:
        metrics = {
            "silhouette": float(silhouette_score(val_X, val_labels)),
            "db": float(davies_bouldin_score(val_X, val_labels)),
            "ch": float(calinski_harabasz_score(val_X, val_labels)),
        }
    else:
        metrics = {"silhouette": np.nan, "db": np.nan, "ch": np.nan}
    return metrics, val_labels, km


def stability_score(train_X, val_X, k, seed=42):
    _, labels_a, _ = evaluate_k(train_X, val_X, k, seed)
    _, labels_b, _ = evaluate_k(train_X, val_X, k, seed + 1)
    if len(np.unique(labels_a)) == 1 or len(np.unique(labels_b)) == 1:
        return np.nan
    return float(adjusted_rand_score(labels_a, labels_b))


def build_per_cluster_knn(X, labels, n_neighbors=6, metric="cosine"):
    per_cluster = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        Xc = X[idx]
        nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(idx)), metric=metric, algorithm="auto")
        nn.fit(Xc)
        per_cluster[int(c)] = (nn, idx)
    return per_cluster


if __name__ == "__main__":
    main()
