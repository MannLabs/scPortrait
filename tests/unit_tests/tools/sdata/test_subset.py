import numpy as np
import pytest
from shapely.geometry import box

from scportrait.tools.sdata.processing._subset import get_bounding_box_sdata, mask_region
from scportrait.tools.sdata.write._helper import _get_image


def _as_numpy(image):
    data = image.data
    if hasattr(data, "compute"):
        data = data.compute()
    return np.asarray(data)


def _expected_mask(masked, shape):
    x_coords = masked.coords["x"].values
    y_coords = masked.coords["y"].values
    x_mask = (x_coords >= shape.bounds[0]) & (x_coords <= shape.bounds[2])
    y_mask = (y_coords >= shape.bounds[1]) & (y_coords <= shape.bounds[3])
    return np.outer(y_mask, x_mask).astype(bool)


@pytest.mark.parametrize(
    "mask,crop",
    [
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_mask_region_mask_crop_combinations(sdata_with_selected_region, mask, crop):
    image = _as_numpy(sdata_with_selected_region.images["input_image"])
    shape = sdata_with_selected_region["select_region"].geometry[0]

    result = mask_region(
        sdata_with_selected_region, image_name="input_image", shape_name="select_region", mask=mask, crop=crop
    )
    result_np = _as_numpy(result)

    minx, miny, maxx, maxy = shape.bounds
    x0, y0 = int(np.floor(minx)), int(np.floor(miny))
    x1, y1 = int(np.ceil(maxx)), int(np.ceil(maxy))

    if crop:
        expected = image[:, y0:y1, x0:x1]
        assert result_np.shape == expected.shape
        if mask:
            assert result_np.min() >= 0
            assert result_np.max() == expected.max()
        else:
            np.testing.assert_array_equal(result_np, expected)
    else:
        expected_mask = _expected_mask(result, shape)
        values = result_np[0]
        assert values[expected_mask].min() == 1
        assert values[~expected_mask].max() == 0


def test_mask_region_requires_mask_or_crop(sdata_with_selected_region):
    with pytest.raises(AssertionError):
        mask_region(
            sdata_with_selected_region, image_name="input_image", shape_name="select_region", mask=False, crop=False
        )


def test_mask_region_missing_image_raises(sdata_with_selected_region):
    with pytest.raises(ValueError):
        mask_region(sdata_with_selected_region, image_name="missing", shape_name="select_region", mask=True, crop=False)


def test_mask_region_missing_shape_raises(sdata_with_selected_region):
    with pytest.raises(KeyError):
        mask_region(sdata_with_selected_region, image_name="input_image", shape_name="missing", mask=True, crop=False)


def test_mask_region_multiple_shapes_raises(sdata_builder):
    image = np.ones((1, 6, 6), dtype=np.uint16)
    shapes = [box(1, 1, 3, 3), box(2, 2, 4, 4)]
    sdata = sdata_builder(image, shapes)

    with pytest.raises(ValueError):
        mask_region(sdata, image_name="input_image", shape_name="select_region", mask=True, crop=False)


@pytest.mark.parametrize(
    "max_width,center_x,center_y",
    [
        (4, 5, 5),
        (6, 2, 8),
    ],
)
def test_get_bounding_box_sdata_reduces_extent(sdata_builder, max_width, center_x, center_y):
    image = np.ones((1, 10, 10), dtype=np.uint16)
    shapes = [box(1, 1, 3, 3)]
    sdata = sdata_builder(image, shapes)

    subset = get_bounding_box_sdata(
        sdata,
        max_width=max_width,
        center_x=center_x,
        center_y=center_y,
        drop_points=False,
    )

    image_subset = _get_image(subset.images["input_image"])
    assert image_subset.sizes["x"] <= 10
    assert image_subset.sizes["y"] <= 10
    assert image_subset.sizes["x"] > 0
    assert image_subset.sizes["y"] > 0
    assert image_subset.sizes["x"] <= max_width
    assert image_subset.sizes["y"] <= max_width


@pytest.mark.parametrize(
    "center_x,center_y",
    [
        (0, 0),
        (-5, -5),
    ],
)
def test_get_bounding_box_sdata_clamps_edges(sdata_builder, center_x, center_y):
    image = np.ones((1, 10, 10), dtype=np.uint16)
    shapes = [box(1, 1, 3, 3)]
    sdata = sdata_builder(image, shapes)

    subset = get_bounding_box_sdata(
        sdata,
        max_width=4,
        center_x=center_x,
        center_y=center_y,
        drop_points=False,
    )

    image_subset = _get_image(subset.images["input_image"])
    x_coords = image_subset.coords["x"].values
    y_coords = image_subset.coords["y"].values

    assert x_coords.min() >= sdata.images["input_image"].coords["x"].values.min()
    assert y_coords.min() >= sdata.images["input_image"].coords["y"].values.min()
