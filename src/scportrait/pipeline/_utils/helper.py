def flatten(list):
    """
    Flatten a list of lists into a single list.

    This function takes in a list of lists (nested lists) and returns a single list
    containing all the elements from the input lists.

    Args:
        list (list of lists): A list containing one or more lists as its elements.

    Returns:
        flattened_list (list): A single list containing all elements from the input lists.

    Example:
    >>> nested_list = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    >>> flatten(nested_list)
    [1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    # Flatten the input list using list comprehension
    return [item for sublist in list for item in sublist]