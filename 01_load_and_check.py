from __future__ import annotations

import geopandas as gpd
import pandas as pd


GRID_PATH = "/Users/shraddha/Desktop/TrafficCollisionAnalysis/Toronto_grid_FINAL.gpkg"


def main() -> None:
    gdf = gpd.read_file(GRID_PATH)

    print("\n--- Basic info ---")
    print(gdf.info())

    print("\n--- First 5 rows ---")
    print(gdf.head())

    print("\n--- Columns ---")
    print(list(gdf.columns))

    print("\n--- CRS ---")
    print(gdf.crs)

    print("\n--- Missing values per column ---")
    print(gdf.isna().sum().sort_values(ascending=False))

    print("\n--- Numeric summary ---")
    print(gdf.select_dtypes(include="number").describe().T)


if __name__ == "__main__":
    main()