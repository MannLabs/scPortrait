from typing import TypeVar

T = TypeVar('T')

def flatten(nested_list: list[list[T]]) -> list[T]:
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