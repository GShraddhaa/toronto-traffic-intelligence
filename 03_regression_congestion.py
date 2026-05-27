from __future__ import annotations

import geopandas as gpd
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


INPUT_PATH = "/Users/shraddha/Desktop/TrafficCollisionAnalysis/Toronto_grid_FINAL.gpkg"
OUTPUT_PATH = "/Users/shraddha/Desktop/TrafficCollisionAnalysis/Toronto_grid_Predicted_Congestion.gpkg"


def main() -> None:
    gdf = gpd.read_file(INPUT_PATH)

    feature_candidates = [
        "avg_daily_vol",
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

    if target not in gdf.columns:
        raise ValueError(f"Missing target column: {target}")

    df = gdf.dropna(subset=features + [target]).copy()

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n--- Regression results ---")
    print("R2:", round(r2_score(y_test, y_pred), 4))
    print("RMSE:", round(mean_squared_error(y_test, y_pred) ** 0.5, 4))

    feature_importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n--- Feature importance ---")
    print(feature_importance)

    df["predicted_congestion"] = model.predict(X)

    gdf["predicted_congestion"] = None
    gdf.loc[df.index, "predicted_congestion"] = df["predicted_congestion"]

    gdf.to_file(OUTPUT_PATH, driver="GPKG")
    print(f"\nSaved predictions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()