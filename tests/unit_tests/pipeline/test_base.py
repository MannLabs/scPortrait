#######################################################
# Unit tests for ../pipeline/base.py
#######################################################

import os
import tempfile

from scportrait.pipeline._base import Logable, ProcessingStep


def test_logable_init():
    with tempfile.TemporaryDirectory() as temp_dir:
        logable = Logable(directory=temp_dir, debug=False)
        assert not logable.debug


def test_logable_log():
    with tempfile.TemporaryDirectory() as temp_dir:
        logable = Logable(directory=temp_dir, debug=True)
        logable.log("Testing")

        log_path = os.path.join(temp_dir, logable.DEFAULT_LOG_NAME)
        assert os.path.isfile(log_path)

        with open(log_path) as f:
            log_content = f.read()
            assert "Testing" in log_content


def test_processing_step_init():
    config = {"setting1": "value1"}
    with tempfile.TemporaryDirectory() as temp_dir:
        processing_step = ProcessingStep(config, f"{temp_dir}/test_step", temp_dir, debug=True)

        assert processing_step.debug
        assert config == processing_step.config


def test_processing_step_register_parameter():
    config = {"setting1": "value1"}
    with tempfile.TemporaryDirectory() as temp_dir:
        processing_step = ProcessingStep(config, f"{temp_dir}/test_step", temp_dir)

        # Test registering a new parameter
        processing_step.register_parameter("setting2", "value2")
        assert "setting2" in processing_step.config
        assert "value2" == processing_step.config["setting2"]


def test_processing_step_get_directory():
    config = {"setting1": "value1"}
    with tempfile.TemporaryDirectory() as temp_dir:
        processing_step = ProcessingStep(config, f"{temp_dir}/test_step", temp_dir)
        assert f"{temp_dir}/test_step" == processing_step.get_directory()
