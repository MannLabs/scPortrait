from __future__ import annotations

import pytest
import requests

import scportrait
from scportrait.data._datasets import _test_dataset, dataset_1_omezarr

DATASET_MARKER = "requires_dataset"
DATASET_LOADERS = {
    "dataset_1_config": lambda: scportrait.data.get_config_file("dataset_1_config"),
    "dataset_1_omezarr": dataset_1_omezarr,
    "test_dataset": _test_dataset,
}


@pytest.fixture(scope="session", autouse=True)
def preload_e2e_remote_data(request: pytest.FixtureRequest) -> None:
    required_datasets = set()
    for item in request.session.items:
        for marker in item.iter_markers(name=DATASET_MARKER):
            for dataset_name in marker.args:
                required_datasets.add(dataset_name)

    failures = []
    unknown = sorted(name for name in required_datasets if name not in DATASET_LOADERS)
    if unknown:
        joined_unknown = ", ".join(unknown)
        pytest.exit(
            f"E2E preflight failed due to unknown dataset marker values: {joined_unknown}",
            returncode=1,
        )

    for name in sorted(required_datasets):
        try:
            DATASET_LOADERS[name]()
        except (OSError, requests.RequestException, RuntimeError, ValueError) as exc:
            failures.append(f"{name}: {exc}")

    if failures:
        joined = "\n".join(f"- {failure}" for failure in failures)
        pytest.exit(f"E2E preflight failed while downloading required data:\n{joined}", returncode=1)
