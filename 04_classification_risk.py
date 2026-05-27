from __future__ import annotations

import geopandas as gpd
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


INPUT_PATH = "/Users/shraddha/Desktop/TrafficCollisionAnalysis/Toronto_grid_FINAL.gpkg"
OUTPUT_PATH = "/Users/shraddha/Desktop/TrafficCollisionAnalysis/Toronto_grid_Risk_Probability.gpkg"


def main() -> None:
    gdf = gpd.read_file(INPUT_PATH)

    if "collision_count" not in gdf.columns:
        raise ValueError("Missing collision_count column.")

    # Define high-risk as top 20% of collision cells
    threshold = gdf["collision_count"].quantile(0.80)
    gdf["high_risk"] = (gdf["collision_count"] >= threshold).astype(int)

    feature_candidates = [
        "avg_daily_vol",
        "avg_speed",
        "signal_count",
        "vol_norm_refactored",
        "speed_norm_refactored",
        "coll_norm",
        "log_congestion_index_refactored",
        "risk_index_2",
        "pressure_index_2",
    ]

    features = [c for c in feature_candidates if c in gdf.columns]
    df = gdf.dropna(subset=features + ["high_risk"]).copy()

    X = df[features]
    y = df["high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n--- Classification results ---")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))

    probs = model.predict_proba(X)[:, 1]
    df["risk_probability"] = probs

    feature_importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n--- Feature importance ---")
    print(feature_importance)

    gdf["risk_probability"] = None
    gdf.loc[df.index, "risk_probability"] = df["risk_probability"]

    gdf.to_file(OUTPUT_PATH, driver="GPKG")
    print(f"\nSaved classification output to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()