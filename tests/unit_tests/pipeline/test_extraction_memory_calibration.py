from __future__ import annotations

from pathlib import Path

import numpy as np

from scportrait.pipeline.extraction import HDF5CellExtraction


def _make_extraction(tmp_path, config: dict | None = None) -> HDF5CellExtraction:
    base_config = {
        "threads": 4,
        "image_size": 128,
        "cache": str(tmp_path / "cache"),
    }
    if config is not None:
        base_config.update(config)
    return HDF5CellExtraction(config=base_config, directory=tmp_path / "extraction")


def test_get_configured_max_inflight_result_batches_uses_override(tmp_path):
    extraction = _make_extraction(tmp_path, {"max_inflight_result_batches": 50})

    configured = extraction._get_configured_max_inflight_result_batches(n_total_batches=12)

    assert configured == 12


def test_calibrate_max_inflight_result_batches_enforces_worker_floor_and_logs_warning(tmp_path, monkeypatch):
    extraction = _make_extraction(tmp_path, {"target_ram_utilization": 0.3})
    extraction.threads = 4

    # Synthetic first returned batch
    result = [(0, np.zeros((3, 8, 8), dtype=np.float32), 101)]

    class FakeAsyncResult:
        def __init__(self, value):
            self._value = value

        def ready(self):
            return True

        def get(self):
            return self._value

    class FakePool:
        def apply_async(self, func, call_args):
            return FakeAsyncResult(result)

    pool = FakePool()
    args = [[(0, 0, 101, (10.0, 10.0))] for _ in range(6)]

    monkeypatch.setattr(extraction, "_get_current_process_rss_bytes", lambda: int(20 * 1024**3))
    monkeypatch.setattr(extraction, "_get_target_job_ram_bytes", lambda: int(12.8 * 1024**3))

    log_messages: list[str] = []
    monkeypatch.setattr(extraction, "log", log_messages.append)

    calibrated_max, _, first_result, pending_results, next_submit_ix = (
        extraction._calibrate_max_inflight_result_batches(
            pool=pool,
            args=args,
        )
    )

    assert calibrated_max == 4
    assert first_result == result
    assert len(pending_results) == 3
    assert next_submit_ix == 4
    assert any("Warning: target RAM budget would limit max_inflight_result_batches" in msg for msg in log_messages)
    assert any("worker_floor_n=4" in msg for msg in log_messages)
    assert any("parent_rss_gb=" in msg for msg in log_messages)
