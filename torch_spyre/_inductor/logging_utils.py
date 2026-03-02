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
Minimal logging infrastructure for torch_spyre._inductor.

Environment Variables:
    SPYRE_INDUCTOR_LOG_LEVEL: Log level (ERROR|WARNING|INFO|DEBUG)
    SPYRE_TENSOR_LOG: Enable tensor-specific logging (0|1)
    SPYRE_LOG_FILE: Path to log file (default: stderr)
"""

import logging
import os
import sys
from typing import Optional

# Global configuration state
_LOGGING_CONFIGURED = False
_TENSOR_LOGGING_ENABLED: Optional[bool] = None


def _get_env_bool(var_name: str, default: bool = False) -> bool:
    """Get boolean value from environment variable."""
    value = os.getenv(var_name, str(int(default)))
    return value.lower() in ("1", "true", "yes", "on")


def _initialize_config() -> None:
    """Initialize logging configuration from environment variables."""
    global _LOGGING_CONFIGURED, _TENSOR_LOGGING_ENABLED

    if _LOGGING_CONFIGURED:
        return

    _TENSOR_LOGGING_ENABLED = _get_env_bool("SPYRE_TENSOR_LOG", False)
    _LOGGING_CONFIGURED = True


def get_inductor_logger(name: str) -> logging.Logger:
    """
    Get or create a logger for the inductor module.

    Args:
        name: Module name (e.g., "stickify", "lowering")

    Returns:
        Configured logger instance
    """
    _initialize_config()

    logger_name = f"torch_spyre._inductor.{name}"
    logger = logging.getLogger(logger_name)

    # Configure if not already done
    if not logger.handlers:
        level_str = os.getenv("SPYRE_INDUCTOR_LOG_LEVEL", "WARNING").upper()
        level = getattr(logging, level_str, logging.WARNING)
        logger.setLevel(level)

        # Create handler
        log_file = os.getenv("SPYRE_LOG_FILE")
        handler: logging.Handler
        if log_file:
            handler = logging.FileHandler(log_file)
        else:
            handler = logging.StreamHandler(sys.stderr)

        # Set simple text formatter
        formatter = logging.Formatter("[%(levelname)s] [%(module)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger


def is_logging_enabled() -> bool:
    """
    Check if tensor logging is enabled via SPYRE_TENSOR_LOG.

    Returns:
        True if tensor logging is enabled, False otherwise
    """
    _initialize_config()
    return bool(_TENSOR_LOGGING_ENABLED)
