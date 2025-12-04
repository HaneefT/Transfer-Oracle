from src.clustering.clustering import run_position_for_testing, knn_distance_baselines

bundle = run_position_for_testing(pos="FW", k=4)

for i in range(len(bundle["X_test"])):
    stats = knn_distance_baselines(
        X=bundle["X_train"],
        labels=bundle["labels_train"],
        df=bundle["train_df"],
        query_idx=i,
        k=5
    )
    print(stats)