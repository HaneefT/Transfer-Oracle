# Error Analysis

This section focuses on failure modes—cases where embeddings and cluster assignments produce unreliable or unintuitive neighbor recommendations—rather than aggregate metrics.

## Identifying problematic players
- High mean KNN distance within their assigned cluster (test queries vs. train pool).
- Small gaps between true KNN distance and random baselines (same cluster/position/global), indicating weak separation.
- These were inspected qualitatively on cluster plots and by reviewing their top neighbors.

## Common failure patterns
- **Statistical extremes**: Players with unusual profiles (e.g., high shot volume, low passing; skewed possession/duels) sit on cluster edges; KNN gaps shrink because few points resemble them.
- **Hybrid/multi-role profiles**: Winger/AM, CM/DM, CB/FB hybrids mix patterns from multiple roles; they get assigned to one cluster but share affinities with others, so neighbors reflect only part of their style and gaps are weak.
- **Coverage quirks**: Uncommon leagues or sparse features can still land a player on the boundary. We removed metadata-based filtering (e.g., foot/side), so we rely on the numeric profile alone; low-minute players are already filtered.

## Position snapshots
- **FW**: Outliers were hybrid creators/finishers and extreme finishers with low involvement—neighbors tilt toward one side of their role.
- **MF**: Box-to-box / attacking mids with notable defensive actions sat between role clusters.
- **DF**: Fullbacks with atypical progression/inversion behaved like hybrids.
- **GK**: Boundary cases tied to uncommon stat distributions rather than minutes.

## Mitigations
- Add role-specific sub-models or finer role tags (AM/CM/DM; FB/CB; creator vs finisher).
- Consider non-linear projections (UMAP) or density-aware clustering (HDBSCAN) to preserve local structure and flag true outliers.
- Surface uncertainty: use small KNN gaps to flag low-confidence recommendations in the UI; apply simple post-filters (foot/age/side) for hybrids.
- Keep the minutes filter; add a “data quality” flag for missing key features.

## Takeaway
The model is strong for well-defined, high-volume, single-role players. It struggles with hybrids, statistical extremes, and cases with sparse/noisy features. Use KNN gaps to flag low-confidence suggestions and consider role-aware or density-aware modeling to reduce these failure modes.
