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

"""
Optional unit tests for the logging infrastructure.

These tests verify that:
1. Environment variables are parsed correctly
2. Logging can be enabled/disabled
3. No exceptions are raised during logging operations

Note: These tests are optional since logging is a side effect that doesn't
affect program correctness. The main value is ensuring the logging system
doesn't crash or interfere with normal operation.
"""

import os
import logging
from unittest.mock import patch
import torch  # noqa: F401
import torch_spyre._inductor.logging_utils as logging_utils
from torch_spyre._inductor.logging_utils import (
    get_inductor_logger,
    is_logging_enabled,
)


class TestLoggingConfiguration:
    """Test environment variable parsing and configuration."""

    def test_logging_disabled(self):
        with patch.object(logging_utils, "_TENSOR_LOGGING_ENABLED", False):
            assert not is_logging_enabled()

    def test_logging_enabled(self):
        with patch.object(logging_utils, "_TENSOR_LOGGING_ENABLED", True):
            assert is_logging_enabled()

    def test_log_level_parsing(self):
        """Test that log levels are parsed correctly."""
        # This test creates a new logger, so we need to patch the environment before calling get_inductor_logger
        with patch.dict(os.environ, {"SPYRE_INDUCTOR_LOG_LEVEL": "DEBUG"}):
            logger = get_inductor_logger("test_log_level")
            assert logger.level == logging.DEBUG


class TestLoggingOperations:
    """Test that logging operations don't crash."""

    def test_logger_creation(self):
        """Test that loggers can be created without errors."""
        logger = get_inductor_logger("test_module")
        assert logger is not None
        assert logger.name.endswith("test_module")

    def test_logging_when_disabled(self):
        """Logging calls should not crash when logging is disabled."""
        with patch.object(logging_utils, "_TENSOR_LOGGING_ENABLED", False):
            logger = get_inductor_logger("test")
            # These should not raise exceptions
            logger.debug("test message")
            logger.info("test message")
            logger.warning("test message")

    def test_logging_with_simple_message(self):
        """Logging with simple messages should not crash."""
        with patch.object(logging_utils, "_TENSOR_LOGGING_ENABLED", True):
            logger = get_inductor_logger("test")
            # This should not raise an exception
            logger.debug("test message with data: shape=[2, 3], device_size=[1, 2, 3]")
