import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, points_from_xy
from pyhdf.SD import SD, SDC

from projections.constants import STANDARD_CRS, MODIS_CRS


PIXEL_SIZE = 463.312716525
UPPER_LEFT_Y = 10007554.677
UPPER_LEFT_X = -20015109.354


class HDF:
    file = None
    h = None
    v = None
    grid_upper_left_y = None
    grid_upper_left_x = None

    def read_geopandas(self, path: Path, datasets: List[str] = None) -> GeoDataFrame:
        self.read(path)
        return self.get_geopandas(datasets)

    def read(self, path: Path):
        self.file = SD(str(path), SDC.READ)
        grid_position = re.findall(r"\.h(\d\d)v(\d\d)\.", path.name)[0]
        self.h = int(grid_position[0])
        self.v = int(grid_position[1])

    def get_geopandas(self, datasets: List[str] = None) -> GeoDataFrame:
        values = {}
        geometry = []
        datasets = datasets if datasets else self.datasets

        for i, dataset in enumerate(datasets):
            array = self.get_array(dataset)
            values[dataset] = array.reshape(-1)
            if i == 0:
                lats = self.get_lats(array)
                lons = self.get_lons(array)
                geometry = points_from_xy(x=lons, y=lats)

        gdf = GeoDataFrame(values, geometry=geometry, crs=MODIS_CRS)
        gdf = gdf.to_crs(STANDARD_CRS)
        return gdf

    @property
    def datasets(self) -> list:
        return list(self.file.datasets())

    def get_array(self, dataset_name: str) -> np.array:
        sds_obj = self.file.select(dataset_name)
        return sds_obj.get()

    def get_lats(self, array: np.array) -> np.array:
        n_grids = array.shape[1]
        grid_upper_left_y = UPPER_LEFT_Y - self.v * PIXEL_SIZE * n_grids
        grid_lat = np.arange(
            grid_upper_left_y + PIXEL_SIZE * n_grids, grid_upper_left_y, -PIXEL_SIZE
        )
        lats = np.repeat(grid_lat, n_grids)
        return lats

    def get_lons(self, array: np.array) -> np.array:
        n_grids = array.shape[0]
        grid_upper_left_x = UPPER_LEFT_X + self.h * PIXEL_SIZE * n_grids
        grid_lon = np.arange(
            grid_upper_left_x, grid_upper_left_x + PIXEL_SIZE * n_grids, PIXEL_SIZE
        )
        lats = np.tile(grid_lon, n_grids)
        return lats

    def map_categoricals_to_attributes(self, df: GeoDataFrame, categoricals: List[str]):
        for var in categoricals:
            attr = self.get_attributes(var)
            inverted = {v: k for k, v in attr.items() if isinstance(v, int)}
            df[var] = df[var].map(inverted)

    def get_attributes(self, dataset_name: str) -> dict:
        return self.file.select(dataset_name).attributes()


def one_hot_encode(df: pd.DataFrame, variables: List[str]):
    for variable in variables:
        dummies = pd.get_dummies(df[variable], prefix=variable)
        for col in dummies.columns:
            df[col] = dummies[col]
        df.drop(columns=variable, inplace=True)
