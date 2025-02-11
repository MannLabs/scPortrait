#######################################################
# Unit tests for ../pipeline/_utils/helper.py
#######################################################

from scportrait.pipeline._utils.helper import flatten


def test_flatten():
    nested_list = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    expected_output = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert flatten(nested_list) == expected_output
