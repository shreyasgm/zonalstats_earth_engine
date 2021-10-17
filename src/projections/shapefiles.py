import shapefile
import shapely
import yaml
from glob import glob
from tqdm import tqdm
from pathlib import Path


def load_shapes(preprocessed_loc):
    shps = glob(str(Path(preprocessed_loc) / '*.shp'))
    all_shapes = {}
    for shp_file in tqdm(shps, total=len(shps), desc='Loading shapes'):
        shp_file = Path(shp_file)
        shp_name = shp_file.name.replace('.shp', '')

        shp = shapefile.Reader(str(shp_file))
        shapes = [shapely.geometry.shape(s) for s in shp.shapes()]
        records = [list(record) for record in shp.records()]
        
        all_shapes[shp_name] = (shapes, records)

    return all_shapes


def iter_records(all_records, verbose=True):
    if verbose:
        i = tqdm(all_records.values(), total=len(all_records))
    else:
        i = all_records.values()
    for shapes, records in i:
        yield from zip(shapes, records)


def load_shapes_by_country(preprocessed_loc):
    all_shapes = load_shapes(preprocessed_loc)
    shapes_by_country = {}
    for name, (shapes, records) in all_shapes.items():
        codes = name.split('_')        
        for code in codes:
            shapes_by_country[code] = []
            for shape, record in zip(shapes, records):
                shapes_by_country[code].append((shape, *record))

    return shapes_by_country


def load_countries(path):
    with open(path) as f:
        countries = yaml.safe_load(f)

    codes = {}
    for country in countries:
        for code in country['codes']:
            if code in codes:
                codes[code].update(country['codes'])
            else:
                codes[code] = set(country['codes'])

    return codes


def load_iso3(path):
    with open(path) as f:
        countries = yaml.safe_load(f)

    return [c['iso3'] for c in countries if 'iso3' in c]