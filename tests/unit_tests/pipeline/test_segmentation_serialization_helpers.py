from __future__ import annotations

import json

from scportrait.pipeline.segmentation.segmentation import Segmentation


def _segmentation_helper_instance() -> Segmentation:
    """Create a lightweight Segmentation instance for helper-method tests."""
    return Segmentation.__new__(Segmentation)


def test_slice_dict_roundtrip():
    """Serialized slice dictionaries should convert back to equivalent slices."""
    seg = _segmentation_helper_instance()
    original = slice(3, 11, 2)

    as_dict = seg._slice_to_dict(original)
    reconstructed = seg._dict_to_slice(as_dict)

    assert as_dict == {"start": 3, "stop": 11, "step": 2}
    assert reconstructed.start == original.start
    assert reconstructed.stop == original.stop
    assert reconstructed.step == original.step


def test_write_and_read_window_file(tmp_path):
    """Window JSON helpers should preserve y/x slices across disk roundtrip."""
    seg = _segmentation_helper_instance()
    path = tmp_path / "window.json"
    window = (slice(10, 110), slice(20, 220, 2))

    seg._write_window_file(str(path), window)

    payload = json.loads(path.read_text())
    assert payload["version"] == 1
    assert payload["y"] == {"start": 10, "stop": 110, "step": None}
    assert payload["x"] == {"start": 20, "stop": 220, "step": 2}

    restored = seg._read_window_file(str(path))
    assert restored[0] == window[0]
    assert restored[1] == window[1]


def test_serialize_and_deserialize_sharding_plan(tmp_path):
    """Sharding plan helper methods should preserve ids and window slices."""
    seg = _segmentation_helper_instance()
    plan = [
        (0, (slice(0, 100), slice(0, 200))),
        (17, (slice(50, 175), slice(25, 225, 5))),
    ]

    serialized = seg._serialize_sharding_plan(plan)
    assert serialized == [
        {
            "id": 0,
            "window": {
                "y": {"start": 0, "stop": 100, "step": None},
                "x": {"start": 0, "stop": 200, "step": None},
            },
        },
        {
            "id": 17,
            "window": {
                "y": {"start": 50, "stop": 175, "step": None},
                "x": {"start": 25, "stop": 225, "step": 5},
            },
        },
    ]

    out_path = tmp_path / "sharding_plan.json"
    seg._write_sharding_plan(str(out_path), plan)

    restored = seg._deserialize_sharding_plan(str(out_path))
    assert restored == plan
