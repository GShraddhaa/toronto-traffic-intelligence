from __future__ import annotations

import geopandas as gpd
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


INPUT_PATH = "/Users/shraddha/Desktop/TrafficCollisionAnalysis/Toronto_grid_FINAL.gpkg"
OUTPUT_PATH = "/Users/shraddha/Desktop/TrafficCollisionAnalysis/Toronto_grid_Clusters.gpkg"


def main() -> None:
    gdf = gpd.read_file(INPUT_PATH)

    cluster_candidates = [
        "avg_daily_vol",
        "avg_speed",
        "collision_count",
        "signal_count",
        "log_congestion_index_refactored",
        "risk_index_2",
    ]

    features = [c for c in cluster_candidates if c in gdf.columns]
    df = gdf.dropna(subset=features).copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    summary = df.groupby("cluster")[features].mean().round(2)
    print("\n--- Cluster summary ---")
    print(summary)

    gdf["cluster"] = None
    gdf.loc[df.index, "cluster"] = df["cluster"]

    gdf.to_file(OUTPUT_PATH, driver="GPKG")
    print(f"\nSaved clusters to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()