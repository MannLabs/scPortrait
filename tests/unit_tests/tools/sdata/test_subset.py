from scportrait.tools.sdata.processing import get_bounding_box_sdata


def test_get_bounding_box_sdata_non_backed_drop_points(sdata_with_labels):
    original_points = set(sdata_with_labels.points.keys())
    assert sdata_with_labels.path is None

    sdata_subset = get_bounding_box_sdata(
        sdata=sdata_with_labels,
        max_width=3000,
        center_x=5000,
        center_y=5000,
        drop_points=True,
    )

    # Original object is restored after temporary point removal.
    assert set(sdata_with_labels.points.keys()) == original_points

    # Subset should be produced without points when drop_points=True.
    if original_points:
        assert len(sdata_subset.points) == 0
