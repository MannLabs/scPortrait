import logging

import pytest


@pytest.mark.parametrize(
    ("level", "message", "visible"),
    [
        (logging.WARNING, "no parent found for %s: %s", False),
        (logging.WARNING, "could not read %s: %s", True),
        (logging.ERROR, "no parent found for %s: %s", True),
    ],
)
def test_ome_zarr_filter_preserves_other_diagnostics(caplog, level, message, visible):
    logger = logging.getLogger("ome_zarr.reader")

    with caplog.at_level(logging.WARNING, logger=logger.name):
        logger.log(level, message, "labels", None)

    assert bool(caplog.records) is visible
