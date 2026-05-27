from __future__ import annotations

import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor


INPUT_PATH = "/Users/shraddha/Desktop/TrafficCollisionAnalysis/Toronto_grid_FINAL.gpkg"
OUTPUT_PATH = "/Users/shraddha/Desktop/TrafficCollisionAnalysis/Toronto_grid_future_scenario.gpkg"


def main() -> None:
    gdf = gpd.read_file(INPUT_PATH)

    feature_candidates = [
        "avg_daily_vol_mean",
        "avg_speed",
        "collision_count",
        "signal_count",
        "vol_norm_refactored",
        "speed_norm_refactored",
        "coll_norm",
        "signal_norm",
    ]
    features = [c for c in feature_candidates if c in gdf.columns]
    target = "log_congestion_index_refactored"

    df = gdf.dropna(subset=features + [target]).copy()

    X = df[features].copy()
    y = df[target]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    # Scenario: 20% increase in traffic volume
    X_future = X.copy()
    X_future["avg_daily_vol_mean"] = X_future["avg_daily_vol_mean"] * 1.20

    # Recompute normalized volume approximately for scenario
    if "vol_norm_refactored" in X_future.columns:
        v = X_future["avg_daily_vol_mean"]
        X_future["vol_norm_refactored"] = (v - v.min()) / (v.max() - v.min())

    df["future_predicted_congestion"] = model.predict(X_future)

    gdf["future_predicted_congestion"] = None
    gdf.loc[df.index, "future_predicted_congestion"] = df["future_predicted_congestion"]

    gdf.to_file(OUTPUT_PATH, driver="GPKG")
    print(f"Saved future scenario output to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()