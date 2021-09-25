import numpy as np
import shutil
import pandas as pd
import xarray as xr
from pathlib import Path
from netCDF4 import Dataset
from tqdm import tqdm

from projections.raster import get_increment_from_tif, merge_df_to_array_by_lat_lon
from projections.utils import (
    get_mock_polygon_from_df,
    make_path,
    aggregate_feather_splits,
)


class NcConverter:
    def __init__(self, nodata_name: str = "_FillValue"):
        self.ds = None
        self.nodata_name = nodata_name

    def read(self, path: Path, **kwargs):
        self.ds = Dataset(path, **kwargs)

    def iter_periods(self, attribute_name: str, lon_offset=0, lat_offset=0):
        for period in range(self.ds[attribute_name].shape[0]):
            yield self.get_xarray(
                attribute_name,
                period=period,
                lon_offset=lon_offset,
                lat_offset=lat_offset,
            )

    def get_xarray(self, attribute_name: str, period=0, lon_offset=0, lat_offset=0):
        values = self.ds[attribute_name][period, :, :].data
        array = xr.DataArray(values.reshape([1, *values.shape]))
        array.attrs["x"] = self.get_lons() + lon_offset
        array.attrs["y"] = self.get_lats() + lat_offset
        array.attrs["nodata"] = getattr(self.ds[attribute_name], self.nodata_name)
        array.name = f"{attribute_name}_p{period}"
        return array

    def get_lons(self):
        return np.array(self.ds["lon"][:].data)

    def get_lats(self):
        return np.array(self.ds["lat"][:].data)

    @property
    def variables(self):
        return list(self.ds.variables.keys())


class Aggregator:
    def __init__(
        self, by_country_path, partial_path, in_memory=False, mapping_dfs=None
    ) -> None:
        self.by_country_path = by_country_path
        self.partial_path = partial_path
        self.in_memory = in_memory
        self.mapping_dfs = mapping_dfs

    def aggregate(self, image):
        self.file_path = self.get_file_path(image)
        self.output_path = self.get_output_path(image)

        if self.output_path.exists():
            shutil.rmtree(self.file_path)
            return image.name

        self.increment = get_increment_from_tif(image)

        if self.in_memory:
            agg = self.in_memory_aggregation(image)
        else:
            agg = self.per_file_aggregation(image)

        self.save_aggregation(agg)
        self.cleanup()

        return image.name

    def get_file_path(self, image):
        return make_path(self.partial_path.parent / image.name)

    def get_output_path(self, image):
        path = self.partial_path.parent / image.name
        return path.with_suffix(".csv")

    def in_memory_aggregation(self, image):
        mapping_dfs = self.get_mapping_dfs()
        dfs = []
        for df in tqdm(mapping_dfs, desc="Mapping"):
            subdf = self._get_mapping_subdf(df, image)
            if subdf.empty:
                print("DF empty")
            else:
                dfs.append(subdf)
        aggregation = aggregate_feather_splits(dfs, no_data_value=image.nodata)
        return aggregation

    def get_mapping_dfs(self):
        if self.mapping_dfs is None:
            self.mapping_dfs = [
                pd.read_feather(df_path)
                for df_path in tqdm(
                    self.by_country_path.glob("*.feather"), desc="Reading"
                )
            ]
        return self.mapping_dfs

    def per_file_aggregation(self, image):
        for df_path in tqdm(self.by_country_path.glob("*.feather"), desc="Mapping"):
            subdf_path = self.file_path / df_path.name
            if subdf_path.exists():
                continue
            df = pd.read_feather(df_path)
            subdf = self._get_mapping_subdf(df, image)
            if subdf.empty:
                print(df_path.name, "is empty")
            else:
                subdf.to_feather(subdf_path)
        files = list(self.file_path.glob("*.feather"))
        aggregation = aggregate_feather_splits(files, no_data_value=image.nodata)
        return aggregation

    def _get_mapping_subdf(self, df, image):
        pol = get_mock_polygon_from_df(df, increment=self.increment)
        subdf = merge_df_to_array_by_lat_lon(df, image, pol)
        return subdf

    def save_aggregation(self, aggregation: pd.DataFrame):
        aggregation.to_csv(self.output_path.with_suffix(".csv"), index=False)

    def cleanup(self):
        shutil.rmtree(self.file_path)
