import geopandas as gpd
import numpy as np
import xarray as xr
import yaml
from itertools import product
from tqdm import tqdm
from typing import List
from pathlib import Path

from projections.raster import is_shape_in_raster
from projections.utils import read_tif


class SpacialTxt:
    def __init__(self, path: Path):
        self.path = path
        self.npy_path = path.with_suffix(".npy")
        self.attrs = {}
        self.values = np.array([])

    def read(self, save=False):
        if self.npy_path.exists():
            self.read_npy()
        else:
            self.read_txt()
            if save:
                self.save()

    def read_npy(self, file=None):
        file = self.npy_path if file is None else file
        self.values = np.load(file)

        yml = file.with_suffix(".yml")
        with open(yml) as f:
            self.attrs = yaml.safe_load(f)

    def read_txt(self, file=None):
        file = self.path if file is None else file
        with open(file) as stream:
            self.parse_stream(stream)
        self.tidy()

    def parse_stream(self, stream):
        self.values = []
        for line in stream:
            if self.is_attr_line(line):
                self.add_attr_line(line)
            else:
                self.add_value_line(line)

    @staticmethod
    def is_attr_line(line):
        return line[:2].isalpha()

    def add_attr_line(self, line):
        name, value = line.split(" ", maxsplit=1)
        value = value.strip()
        try:
            value = float(value)
        except ValueError:
            pass
        self.attrs[name.strip()] = value

    def add_value_line(self, line):
        line_values = []
        for x in line.split(" "):
            x = x.strip()
            if x:
                line_values.append(float(x))
        self.values.append(line_values)

    def tidy(self):
        self.values = np.array(self.values)

    def save(self, path: Path = None):
        assert type(self.values) == np.ndarray, "No file to save!"

        path = self.npy_path if path is None else path

        np.save(path, self.values)
        with open(path.with_suffix(".yml"), "w") as f:
            yaml.dump(self.attrs, f)

    def get_xarray(self):
        lats = self.get_lats_from_attrs()
        lons = self.get_lons_from_attrs()
        array = xr.DataArray(self.values.reshape([1, *self.values.shape]))
        array.attrs["x"] = lons
        array.attrs["y"] = lats
        array.attrs["nodata"] = self.nodata
        return array

    def get_lons_from_attrs(self):
        n_cols = self.attrs["ncols"]
        from_ = self.attrs["xllcorner"]
        to_ = from_ + n_cols * self.increment
        return np.arange(from_, to_, step=self.increment)

    def get_lats_from_attrs(self):
        n_rows = self.attrs["nrows"]
        to_ = self.attrs["yllcorner"]
        from_ = to_ + n_rows * self.increment
        return np.arange(from_, to_, step=-1 * self.increment)

    @property
    def increment(self):
        return self.attrs["cellsize"]

    @property
    def nodata(self):
        return self.attrs["NODATA_value"]


def get_indices_by_file(gdf: gpd.GeoDataFrame, files: List[Path]):
    indices_by_file = {}
    for file in tqdm(files):
        image = read_tif(file)
        mask = gdf["geometry"].apply(lambda x: is_shape_in_raster(x, image))
        indices_by_file[file.name] = list(gdf[mask].index)
    return indices_by_file


def map_shape_to_elevation_file(pol):
    lons = set()
    lats = set()

    min_lon, min_lat, max_lon, max_lat = pol.bounds

    lons.add(map_lon_to_elevation_file(min_lon))
    lons.add(map_lon_to_elevation_file(max_lon))
    lats.add(map_lat_to_elevation_file(min_lat))
    lats.add(map_lat_to_elevation_file(max_lat))

    files = [f"gt30{lon}{lat}.tif" for lon, lat in product(lons, lats)]
    return files


def map_lat_to_elevation_file(lat):
    gap = 50
    position = -10

    if lat <= position:
        direction = "s"
        while lat < position:
            position -= gap
    else:
        direction = "n"
        while lat > position:
            position += gap

    return f"{direction}{abs(position):02d}"


def map_lon_to_elevation_file(lon):
    gap = 40
    position = 20

    if lon <= position:
        direction = "w"
        # position -= int(((position - lon) // gap + 1) * gap)
        while lon < position:
            position -= gap
    else:
        direction = "e"
        while lon > position:
            position += gap
        position -= gap

    return f"{direction}{abs(position):03d}"
