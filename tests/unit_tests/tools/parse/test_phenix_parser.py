import pandas as pd

from scportrait.tools.parse._parse_phenix import CombinedPhenixParser, PhenixParser


def _make_parser() -> PhenixParser:
    parser = PhenixParser.__new__(PhenixParser)
    parser.compress_rows = False
    parser.compress_cols = False
    return parser


def _make_metadata(x_values: list[float], y_values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Row": ["A"] * len(x_values),
            "Well": ["01"] * len(x_values),
            "X": x_values,
            "Y": y_values,
            "Channel": ["ch1"] * len(x_values),
            "Timepoint": [0] * len(x_values),
            "Zstack": [0] * len(x_values),
        }
    )


def test_generate_new_filenames_clusters_jitter_dominated_x_positions():
    parser = _make_parser()
    metadata = _make_metadata(
        x_values=[0.0, 0.001, 0.002, 1.0, 1.001, 1.002, 2.0, 2.001],
        y_values=[5.0] * 8,
    )

    result = parser._generate_new_filenames(metadata)

    assert list(result["X_pos"]) == ["000", "000", "000", "001", "001", "001", "002", "002"]
    assert result["Y_pos"].nunique() == 1


def test_generate_new_filenames_clusters_jitter_dominated_y_positions():
    parser = _make_parser()
    metadata = _make_metadata(
        x_values=[10.0] * 8,
        y_values=[0.0, 0.001, 0.002, 1.0, 1.001, 1.002, 2.0, 2.001],
    )

    result = parser._generate_new_filenames(metadata)

    assert list(result["Y_pos"]) == ["000", "000", "000", "001", "001", "001", "002", "002"]
    assert result["X_pos"].nunique() == 1


def test_generate_new_filenames_preserves_clean_tile_spacing():
    parser = _make_parser()
    metadata = _make_metadata(
        x_values=[0.0, 1.0, 2.0],
        y_values=[0.0, 1.0, 2.0],
    )

    result = parser._generate_new_filenames(metadata)

    assert list(result["X_pos"]) == ["000", "001", "002"]
    assert list(result["Y_pos"]) == ["000", "001", "002"]


def test_combined_metadata_deduplicates_on_tile_positions():
    parser = CombinedPhenixParser.__new__(CombinedPhenixParser)
    parser.flatfield_status = True
    parser.phenix_dirs = ["exp_a", "exp_b"]

    metadata_by_path = {
        "exp_a/Images/Index.ref.xml": pd.DataFrame(
            {
                "Row": ["A", "A"],
                "Well": ["01", "01"],
                "Zstack": [0, 0],
                "Timepoint": [0, 0],
                "X": [0.0, 1.0],
                "Y": [0.0, 0.0],
                "Channel": ["ch1", "ch1"],
                "filename": ["tile_a0.tif", "tile_a1.tif"],
            }
        ),
        "exp_b/Images/Index.ref.xml": pd.DataFrame(
            {
                "Row": ["A", "A"],
                "Well": ["01", "01"],
                "Zstack": [0, 0],
                "Timepoint": [0, 0],
                "X": [0.001, 2.0],
                "Y": [0.0, 0.0],
                "Channel": ["ch1", "ch1"],
                "filename": ["tile_b0_duplicate.tif", "tile_b2.tif"],
            }
        ),
    }

    parser._read_phenix_xml = lambda path: metadata_by_path[path].copy()

    result = parser._get_phenix_metadata()

    assert len(result) == 3
    assert sorted(result["X_pos"].tolist()) == [0, 1, 2]
    assert "tile_b0_duplicate.tif" not in set(result["filename"])
    assert set(result["filename"]) == {"tile_a0.tif", "tile_a1.tif", "tile_b2.tif"}
