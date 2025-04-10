#######################################################
# Unit tests for ../pipeline/_utils/helper.py
#######################################################

import sys
import tempfile
from pathlib import Path

import pytest

from scportrait.pipeline._utils.helper import (
    _check_for_spatialdata_plot,
    flatten,
    read_config,
    write_config,
)


def test_write_and_read_config():
    config_data = {"test_section": {"param1": "value1", "param2": 42, "param3": "true"}}

    with tempfile.TemporaryDirectory() as tmpdirname:
        config_path = Path(tmpdirname) / "config.yml"
        write_config(config_data, config_path)
        assert config_path.exists()

        loaded_config = read_config(config_path)
        assert loaded_config == config_data


def test_write_quotes_strings():
    config_data = {"key": "string_value", "num": "123", "bool_like": "true"}

    with tempfile.TemporaryDirectory() as tmpdirname:
        config_path = Path(tmpdirname) / "quoted.yml"
        write_config(config_data, config_path)

        with open(config_path) as f:
            contents = f.read()
            assert '"string_value"' in contents
            assert '"123"' in contents
            assert '"true"' in contents


@pytest.mark.parametrize(
    "input_list, expected_output",
    [
        ([[1, 2, 3]], [1, 2, 3]),
        ([["a", "b", "c"]], ["a", "b", "c"]),
        ([[1, 2], [3, 4, 5], [6]], [1, 2, 3, 4, 5, 6]),
        ([], []),
        ([[]], []),
        ([[[1, 2]], [[3, 4]]], [[1, 2], [3, 4]]),  # tests flatten does not recurse
    ],
)
def test_flatten_list(input_list, expected_output):
    assert flatten(input_list) == expected_output


def test_check_for_spatialdata_plot_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "spatialdata_plot", None)
    with pytest.raises(ImportError, match="Extended plotting capabilities"):
        monkeypatch.delenv("spatialdata_plot", raising=False)
        _check_for_spatialdata_plot()
