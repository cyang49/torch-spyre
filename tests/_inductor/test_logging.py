# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for logging infrastructure."""

import os
import logging
from unittest.mock import patch
import torch  # noqa: F401
import torch_spyre._inductor.logging_utils as logging_utils
from torch_spyre._inductor.logging_utils import (
    get_inductor_logger,
    is_inductor_logging_enabled,
)


class TestLoggingConfiguration:
    def test_logging_disabled_by_default(self):
        with patch.object(logging_utils, "_INDUCTOR_LOGGING_ENABLED", None):
            with patch.dict(os.environ, {}, clear=True):
                assert not is_inductor_logging_enabled()

    def test_logging_enabled(self):
        with patch.object(logging_utils, "_INDUCTOR_LOGGING_ENABLED", None):
            with patch.dict(os.environ, {"SPYRE_INDUCTOR_LOG": "1"}):
                assert is_inductor_logging_enabled()

    def test_log_level_when_enabled(self):
        with patch.dict(os.environ, {"SPYRE_INDUCTOR_LOG": "1"}):
            logger = get_inductor_logger("test_enabled")
            assert logger.level == logging.DEBUG

    def test_log_level_when_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            logger = get_inductor_logger("test_disabled")
            assert logger.level == logging.CRITICAL


class TestLoggingOperations:
    def test_logger_creation(self):
        logger = get_inductor_logger("test_module")
        assert logger is not None
        assert logger.name.endswith("test_module")

    def test_logging_calls(self):
        logger = get_inductor_logger("test")
        # These should not raise exceptions
        logger.debug("test message")
        logger.info("test message")
        logger.warning("test message")
        logger.debug("test message with data: shape=[2, 3], device_size=[1, 2, 3]")
