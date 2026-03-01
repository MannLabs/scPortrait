import os

import pytest

from scportrait.pipeline._base import Logable, ProcessingStep


def test_logable_log_splits_multiline_strings(tmp_path):
    logable = Logable(directory=tmp_path, debug=False)

    logable.log("line1\nline2")

    log_path = os.path.join(tmp_path, logable.DEFAULT_LOG_NAME)
    with open(log_path) as handle:
        content = handle.read()
    assert "line1" in content
    assert "line2" in content


def test_logable_log_formats_dict_entries(tmp_path):
    logable = Logable(directory=tmp_path, debug=False)

    logable.log({"alpha": 1, "beta": "x"})

    log_path = os.path.join(tmp_path, logable.DEFAULT_LOG_NAME)
    with open(log_path) as handle:
        content = handle.read()
    assert "alpha: 1" in content
    assert "beta: x" in content


def test_processing_step_create_and_clear_temp_dir_in_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    step = ProcessingStep(config={"cache": str(cache_dir)}, directory=tmp_path / "step")

    step.create_temp_dir()
    tmp_dir_path = step._tmp_dir_path
    assert step._tmp_dir_path.startswith(str(cache_dir))
    assert os.path.isdir(step._tmp_dir_path)

    step.clear_temp_dir()
    assert not os.path.exists(tmp_dir_path)


def test_processing_step_call_warns_when_process_missing(tmp_path):
    step = ProcessingStep(config={"cache": str(tmp_path / "cache")}, directory=tmp_path / "step")

    with pytest.warns(UserWarning, match="No process method defined."):
        result = step()

    assert result is None


def test_processing_step_call_empty_warns_when_method_missing(tmp_path):
    step = ProcessingStep(config={"cache": str(tmp_path / "cache")}, directory=tmp_path / "step")

    with pytest.warns(UserWarning, match="No return_empty_mask method defined."):
        result = step.__call_empty__()

    assert result is None


def test_register_parameter_nested_keys_not_supported(tmp_path):
    step = ProcessingStep(config={"cache": str(tmp_path / "cache")}, directory=tmp_path / "step")

    with pytest.raises(NotImplementedError):
        step.register_parameter(["a", "b"], 1)
