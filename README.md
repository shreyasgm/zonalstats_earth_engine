# Projections

[projections](src/projections) is a collection of utils written to combine geographical data in various formats (e.g. [TIF](src/projections/utils.py), [NC](src/projections/temperature.py), [TXT](src/projections/elevation.py), [HDF](src/projections/modis.py)) into a set of agregations in tabular format. These modules are tailored for the data they were written for, thus the naming and organisation of the files makes reference to these specific usecases, so it may be necessary to adapt some routines to acommodate new sources of data.

Specific applications of these methods can be found in the [Notebooks](src/projections/Notebooks) directory. As an example, the main elements of [GPCP](https://psl.noaa.gov/data/gridded/data.gpcp.html) processing are highligted below (see [full notebook](src/projections/Notebooks/GPCP.ipynb) for more details):

```python
# Read data
converter = NcConverter(nodata_name="missing_value")
converter.read(read_path / filename)
IMAGE = converter.get_xarray("precip", period=0, lon_offset=-180)

geo_df = gpd.read_file('../Shapefiles/preprocessed/all_countries_with_eth.shp')

# Process all geometries
iterator = partial(utils.yield_missing_shapes, save_path=partial_path, prefix='p0')
n_processes = 30
with ProcessPoolExecutor(n_processes) as ppe:
    for _ in tqdm(
        ppe.map(save_location_mapping, iterator(geo_df)),
        total=geo_df.shape[0]
    ):
        pass

# Aggregate and save into a single output file
def aggregate(image):
    agg = Aggregator(
        by_country_path=by_country_path,
        partial_path=partial_path,
        in_memory=True,
        mapping_dfs=mapping_dfs
    )
    agg.aggregate(image)

for variable in ("precip",):
    with ProcessPoolExecutor(n_processes) as tpe:
        for name in tpe.map(aggregate,
                            converter.iter_periods(attribute_name=variable, lon_offset=-180)):
            pass
```

# Datasets

| Name        | Coordinates                                                   | Raster                                                               | Output variable | Notes |
| ----------- | ------------------------------------------------------------- | -------------------------------------------------------------------- | --------------- | ----- |
| CRU         | [Yes](src/projections/Notebooks/Precipitaciones_cru.ipynb)    | [Yes](src/projections/Notebooks/Precipitaciones_cru_raster.ipynb)    |                 |
| Elevation   | No                                                            | [Yes](src/projections/Notebooks/Elevation.ipynb)                     |                 |
| FAO         | No                                                            | [Yes](src/projections/Notebooks/AgroAllFAO.ipynb)                    |                 |
| GDELT       | [Yes](src/projections/Notebooks/GDELT.ipynb)                  | No                                                                   |                 |
| GPCC        | [Yes](src/projections/Notebooks/Precipitaciones_GPCC.ipynb)   | [Yes](src/projections/Notebooks/Precipitaciones_GPCC_raster.ipynb)   |                 |
| GPCP        | No                                                            | [Yes](src/projections/Notebooks/GPCP.ipynb)                          |                 |
| Nighlights  | [Yes](src/projections/Notebooks/NL_Subnat.ipynb)              | [Yes](src/projections/Notebooks/NL_NOAA.ipynb)                       |                 |
| Population  | No                                                            | [Yes](src/projections/Notebooks/Population.ipynb)                    |                 |
| Ruggedness  | No                                                            | [Yes](src/projections/Notebooks/Elevation-Ruggedness.ipynb)          |                 |
| SCAD        | No                                                            | [Yes](src/projections/Notebooks/SCAD.ipynb)                          |                 |
| SCPDSI      | [Yes](src/projections/Notebooks/Precipitaciones_scpdsi.ipynb) | [Yes](src/projections/Notebooks/Precipitaciones_scpdsi_raster.ipynb) |                 |
| Temperature | No                                                            | [Yes](src/projections/Notebooks/Temperature.ipynb)                   |                 |

# Aggregating results
