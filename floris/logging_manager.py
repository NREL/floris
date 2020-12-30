# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

import logging
from datetime import datetime

import coloredlogs


# Global variables for logging
LOG_TO_CONSOLE = True
CONSOLE_LEVEL = "INFO"
LOG_TO_FILE = False
FILE_LEVEL = "INFO"


def configure_console_log(enabled=True, level="INFO"):
    """
    Sets whether the log statements are displayed in the console logging, and,
    if enabled, the log level to use. If not explicitly configured, console
    logging is ON at the INFO level.

    Args:
        enabled (bool, optional): Whether to enable console logging.
            Defaults to True.
        level (str, optional): If `enabled` is True, sets the level that the
            logging module displays. This level is the minimum and all
            messages at a higher level are included. Valid values are

                - CRITICAL
                - ERROR
                - WARNING
                - INFO
                - DEBUG

            Defaults to "INFO".
    """
    global LOG_TO_CONSOLE
    global CONSOLE_LEVEL
    LOG_TO_CONSOLE = enabled
    CONSOLE_LEVEL = level
    _setup_logger()


def configure_file_log(enabled=True, level="INFO"):
    """
    Sets whether the log statements are exported to a log file, and,
    if enabled, the log level to use. If not explicitly configured, file
    logging is OFF.

    Args:
        enabled (bool, optional): Whether to enable file logging.
            This argument defaults to True.
        level (str, optional): If `enabled` is True, sets the level that the
            logging module displays. This level is the minimum and all
            messages at a higher level are included. Valid values are

                - CRITICAL
                - ERROR
                - WARNING
                - INFO
                - DEBUG

            Defaults to "INFO".
    """
    global LOG_TO_FILE
    global FILE_LEVEL
    LOG_TO_FILE = enabled
    FILE_LEVEL = level
    _setup_logger()


def _setup_logger():
    """
    Configures the root logger based on the default or user-specified settings.
    As needed, a StreamHandler is created for console logging or FileHandler
    is created for file logging. Either or both are attached to the root
    logger for use throughout FLORIS.

    Returns:
        logging.Logger: The root logger from the `logging` module.
    """
    # Configure logging for the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # level_styles = {'warning': {'color': 'red', 'bold': False}}
    fmt_console = "%(name)s %(levelname)s %(message)s"
    fmt_file = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    file_name = "floris_{:%Y-%m-%d-%H_%M_%S}.log".format(datetime.now())

    # TODO: understand why this doesnt work and fix it!
    # if logger.hasHandlers():
    #     print(logger.handlers, len(logger.handlers))
    #     for i, handler in enumerate(logger.handlers):
    #         print(i, handler)
    #         logger.removeHandler(handler)
    #     print(logger.handlers, len(logger.handlers))

    # Remove all existing handlers before adding new ones
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])

    # Configure and add the console handler
    if LOG_TO_CONSOLE:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(CONSOLE_LEVEL)
        console_format = coloredlogs.ColoredFormatter(
            # level_styles=level_styles,
            fmt=fmt_console
        )
        console_handler.setFormatter(console_format)
        console_handler.addFilter(TracebackInfoFilter(clear=True))
        logger.addHandler(console_handler)

    # Configure and add the file handler
    if LOG_TO_FILE:
        file_handler = logging.FileHandler(file_name)
        file_handler.setLevel(FILE_LEVEL)
        file_format = logging.Formatter(fmt_file)
        file_handler.setFormatter(file_format)
        file_handler.addFilter(TracebackInfoFilter(clear=False))
        logger.addHandler(file_handler)

    return logger


class TracebackInfoFilter(logging.Filter):
    """Clear or restore the exception on log records"""

    def __init__(self, clear=True):
        self.clear = clear

    def filter(self, record):
        if self.clear:
            record._stack_info_hidden, record.stack_info = record.stack_info, None
        elif hasattr(record, "_stack_info_hidden"):
            record.stack_info = record._stack_info_hidden
            del record._stack_info_hidden
        return True


class LoggerBase:
    """
    Convenience super-class to any class requiring access to the logging
    module. The virtual property here allows a simple and dynamic method
    for obtaining the correct logger for the calling class.
    """

    @property
    def logger(self):
        return logging.getLogger(
            "{}.{}".format(type(self).__module__, type(self).__name__)
        )
