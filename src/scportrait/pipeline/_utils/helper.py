from pathlib import PosixPath
from typing import TypeVar

import yaml

T = TypeVar("T")


def read_config(config_path: str | PosixPath) -> dict:
    with open(config_path) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config


class QuotedStringDumper(yaml.SafeDumper):
    def represent_str(self, data):
        return self.represent_scalar("tag:yaml.org,2002:str", data, style='"')


QuotedStringDumper.add_representer(str, QuotedStringDumper.represent_str)


def write_config(config: dict, config_path: str | PosixPath) -> None:
    with open(config_path, "w") as stream:
        try:
            yaml.dump(config, stream, sort_keys=False, Dumper=QuotedStringDumper)
        except yaml.YAMLError as exc:
            print(exc)


def flatten(nested_list: list[list[T]]) -> list[T | tuple[T]]:
    """Flatten a list of lists into a single list.

    Args:
        nested_list: A list containing one or more lists as its elements

    Returns:
        A single list containing all elements from the input lists

    Example:
        >>> nested_list = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        >>> flatten(nested_list)
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    return [item for sublist in nested_list for item in sublist]


def _check_for_spatialdata_plot() -> None:
    """Helper function to check if required package is installed"""
    # check for spatialdata_plot
    try:
        import spatialdata_plot
    except ImportError:
        raise ImportError(
            "Extended plotting capabilities required. Please install with `pip install 'scportrait[plotting]'`."
        ) from None
