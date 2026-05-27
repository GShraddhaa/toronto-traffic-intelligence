from __future__ import annotations

import geopandas as gpd
import numpy as np


INPUT_PATH = "/Users/shraddha/Desktop/TrafficCollisionAnalysis/Toronto_grid_Final.gpkg"
OUTPUT_PATH = "/Users/shraddha/Desktop/TrafficCollisionAnalysis/Toronto_grid_Final_Clean.gpkg"


def min_max_normalize(series):
    s = series.astype(float)
    min_val = s.min()
    max_val = s.max()
    if max_val == min_val:
        return np.zeros(len(s))
    return (s - min_val) / (max_val - min_val)


def main() -> None:
    gdf = gpd.read_file(INPUT_PATH)

    # Rename these to match your actual fields from QGIS
    rename_map = {
        # Example placeholders; edit based on your layer
        "OBJECTID_count": "collision_count",
        "OBJECTID_count_2": "signal_count",
    }

    existing_rename_map = {k: v for k, v in rename_map.items() if k in gdf.columns}
    gdf = gdf.rename(columns=existing_rename_map)

    # Keep only rows with the core fields present
    required = ["avg_daily_vol_mean", "avg_speed_mean"]
    missing_required = [c for c in required if c not in gdf.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    # Create missing engineered fields if needed
    if "collision_count" not in gdf.columns:
        print("Warning: collision_count not found. Set this name correctly before continuing.")

    if "signal_count" not in gdf.columns:
        print("Warning: signal_count not found. Set this name correctly before continuing.")

    # Fill reasonable nulls for count-type fields
    for col in ["collision_count", "signal_count"]:
        if col in gdf.columns:
            gdf[col] = gdf[col].fillna(0)

    # Drop rows missing core traffic values
    gdf = gdf.dropna(subset=["avg_daily_vol_mean", "avg_speed_mean"]).copy()

    # Normalize if not already present
    if "vol_norm" not in gdf.columns:
        gdf["vol_norm"] = min_max_normalize(gdf["avg_daily_vol_mean"])

    if "speed_norm" not in gdf.columns:
        gdf["speed_norm"] = min_max_normalize(gdf["avg_speed_mean"])

    if "collision_count" in gdf.columns and "coll_norm" not in gdf.columns:
        gdf["coll_norm"] = min_max_normalize(gdf["collision_count"])

    if "signal_count" in gdf.columns and "sig_norm" not in gdf.columns:
        gdf["sig_norm"] = min_max_normalize(gdf["signal_count"])

    # Congestion index
    if "congestion_index" not in gdf.columns:
        gdf["congestion_index"] = gdf["vol_norm"] * (1 - gdf["speed_norm"])

    # Risk index
    if "collision_count" in gdf.columns and "risk_index" not in gdf.columns:
        gdf["risk_index"] = gdf["coll_norm"] * gdf["vol_norm"]

    # Optional pressure index
    if "signal_count" in gdf.columns and "pressure_index" not in gdf.columns:
        gdf["pressure_index"] = (
            0.5 * gdf["vol_norm"]
            + 0.3 * gdf.get("coll_norm", 0)
            + 0.2 * gdf["sig_norm"]
        )

    print("\nPrepared columns:")
    print(list(gdf.columns))

    gdf.to_file(OUTPUT_PATH, driver="GPKG")
    print(f"\nSaved cleaned dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()