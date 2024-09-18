import pandas as pd
from shapely.geometry import Polygon
from rasterio.features import rasterize


def _read_napari_csv(path):
    # read csv table
    shapes = pd.read_csv(path, sep=",")
    shapes.columns = ["index_shape", "shape-type", "vertex-index", "axis-0", "axis-1"]

    # get unqiue shapes
    shape_ids = shapes.index_shape.value_counts().index.tolist()

    polygons = []

    for shape_id in shape_ids:
        _shapes = shapes.loc[shapes.index_shape == shape_id]
        x = _shapes["axis-0"].tolist()
        y = _shapes["axis-1"].tolist()

        polygon = Polygon(zip(x, y))
        polygons.append(polygon)

    return polygons


def _generate_mask_polygon(poly, outshape):
    x, y = outshape
    img = rasterize(poly, out_shape=(x, y))
    return img.astype("bool")
