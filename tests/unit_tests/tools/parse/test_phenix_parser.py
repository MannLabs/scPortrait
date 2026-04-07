import pandas as pd

from scportrait.tools.parse._parse_phenix import PhenixParser


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
