import pandas as pd
import numpy as np
import rioxarray

from typing import List, Union
from tqdm import tqdm
from itertools import product
from pathlib import Path
from unittest.mock import MagicMock


def aggregate_by_groups(df, groups, output_folder, values, done=None, batch=None):
    batch = "" if batch is None else f"_b{batch}"
    if done is None:
        done = []

    for group_name, group in tqdm(groups.items(), desc="Aggregating"):
        fname = f"{output_folder.name}_{group_name}{batch}.csv"
        if fname in done:
            continue

        pivot = pd.pivot_table(
            df,
            values=values,
            index=group,
            aggfunc=[np.mean, np.median, np.std, "count"],
        )
        pivot.columns = ["_".join(x) for x in pivot.columns]
        pivot.to_csv(output_folder / fname)


def aggregate_feather_splits_and_save(
    input_path: Path, output_path: Path, no_data_value=None
):
    file_parts = list(input_path.glob("*.feather"))
    aggregation = aggregate_feather_splits(file_parts, no_data_value=no_data_value)
    aggregation.to_csv(output_path.with_suffix(".csv"), index=False)


def aggregate_feather_splits(files: list, no_data_value=None):
    if not isinstance(files[0], pd.DataFrame):
        files = [
            robust_read(file, default=pd.DataFrame({"value": []}))
            for file in tqdm(files, desc="Reading")
        ]

    countries = {}
    for df in tqdm(files, desc="Grouping"):
        df = df[df["value"] != no_data_value].copy()
        if df.empty:
            continue

        collapsed = get_weighted_average(
            df, value="value", weight="intersection_area", by=["id"], new_name="a_value"
        )
        collapsed["s_value"] = get_area_weighted_sum(
            df,
            value="value",
            weight="intersection_area",
            area="grid_size",
            by=["id"],
        )["value"]
        collapsed["n_grids"] = df.shape[0]
        collapsed.reset_index(inplace=True)

        country = collapsed.loc[0, "id"]
        countries.setdefault(country, []).append(collapsed)

    dfs = [combine_dataframes(country_dfs) for country_dfs in countries.values()]
    df = combine_dataframes(dfs)
    df.rename(columns={"_weighted_value_": "value"}, inplace=True)
    return df


def get_weighted_average(df, value: str, weight: str, by: list, new_name: str = None):
    df = df.copy()
    df["_weighted_value_"] = df[value] * df[weight]
    df = df.groupby(by)[["_weighted_value_", weight]].sum()
    df["_weighted_value_"] /= df[weight]

    new_name = new_name if new_name else value
    df.rename(columns={"_weighted_value_": new_name}, inplace=True)
    return df


def get_area_weighted_sum(
    df, value: str, weight: str, area: str, by: list, new_name: str = None
):
    df = df.copy()
    df["_proportion_"] = df[value] * (df[weight] / df[area])
    df = df.groupby(by)[["_proportion_"]].sum()

    new_name = new_name if new_name else value
    df.rename(columns={"_proportion_": new_name}, inplace=True)
    return df


def combine_dataframes(dfs: List[pd.DataFrame]):
    df = dfs[0]
    if len(dfs) > 1:
        df = df.append(dfs[1:], ignore_index=True)
    return df


def map_year_month(df, time_col, from_year, to_year, format="{:02d}-{:02d}"):
    year_map = {}
    month_map = {}
    for year, month in product(range(from_year, to_year + 1), range(1, 13)):
        t = format.format(year % 100, month)
        year_map[t] = year
        month_map[t] = month

    df["year"] = df[time_col].map(year_map)
    df["month"] = df[time_col].map(month_map)


def read_tif(path):
    return rioxarray.open_rasterio(path)


def read_csv(file, default=pd.DataFrame(), **kwargs):
    try:
        df = pd.read_csv(file, **kwargs)
    except pd.errors.EmptyDataError:
        df = default
    return df


def robust_read(file: Union[Path, str], default=None, **kwargs) -> pd.DataFrame:
    try:
        df = read(file, **kwargs)
    except Exception as e:
        print(f"Error while reading {file}. {e}")
        df = pd.DataFrame() if default is None else default
    return df


def read(file: Union[Path, str], **kwargs) -> pd.DataFrame:
    suffix = Path(file).suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(file, **kwargs)
    elif suffix == ".feather":
        df = pd.read_feather(file, **kwargs)
    elif suffix == ".tif":
        df = read_tif(file)
    else:
        raise ValueError(f'Suffix "{suffix}" not known')
    return df


def get_mock_polygon_from_df(df: pd.DataFrame, increment: float):
    shape = MagicMock()
    shape.bounds = (
        df["lon"].min() - increment,
        df["lat"].min() - increment,
        df["lon"].max() + increment,
        df["lat"].max() + increment,
    )
    return shape


def yield_missing_shapes(gdf, save_path, prefix):
    for _, row in gdf.iterrows():
        path = save_path / get_save_file_name(prefix, row)
        if path.exists():
            continue

        yield row, path


def get_save_file_name(prefix, row):
    portion = f"_p{row['portion']}" if row["portion"] else ""
    return f'{prefix}_{row["id"]}{portion}.csv'


def make_path(path):
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    return path


def union_and_save_portions(read_from: Path, save_in: Path):
    """
    As part of shapefile preprocessing, some adm2 polygons were split into smaller
    pieces called "portions". Union these back together and save them as feather.
    """
    df_by_region = {}
    for file in tqdm(read_from.glob("*.csv"), desc="Reading"):
        try:
            df = pd.read_csv(file)
        except pd.errors.EmptyDataError:
            continue

        if "id" not in df.columns:
            df["id"] = df["adm2"]
            df["id"].fillna(df["adm1"], inplace=True)
            df["id"].fillna(df["adm0"], inplace=True)
        region = df.loc[0, "id"]
        df_by_region.setdefault(region, []).append(df)

    for region, dfs in tqdm(df_by_region.items(), desc="Saving"):
        df = combine_dataframes(dfs)
        df.to_feather(save_in / f"{region}.feather")
