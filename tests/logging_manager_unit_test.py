# Copyright 2022 NREL
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

import copy
import logging
from pathlib import Path

import yaml
import floris.logging_manager
from floris.simulation import Floris


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full_v3.yaml"
DICT_INPUT = yaml.load(open(YAML_INPUT, "r"), Loader=yaml.SafeLoader)


def test_configure_console_log():

    logger = logging.getLogger(name="floris")

    # Enable console logging at INFO level
    test_level  = "INFO"
    floris.logging_manager.configure_console_log(enabled=True, level=test_level)

    ## The number of handlers should always be 0, 1, or 2. In this case, either 1 or 2 since we've added one above.
    assert 0 < len(logger.handlers) <= 2

    ## Find the console logger - its a StreamHandler class
    console_logger = None
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler):
            console_logger = h
    
    ## The level should be set to the value above.
    ## The nameToLevel dictionary converts the string to a numeric value used in the logging library
    assert console_logger.level == logging._nameToLevel[test_level]


    # Change the logging level to ERROR
    test_level  = "ERROR"
    floris.logging_manager.configure_console_log(enabled=True, level=test_level)
    assert 0 < len(logger.handlers) <= 2
    console_logger = None
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler):
            console_logger = h
    assert console_logger.level == logging._nameToLevel[test_level]


    # Disbale console logging
    floris.logging_manager.configure_console_log(enabled=False)

    ## Here, there should be only 0 or 1 handlers
    assert 0 <= len(logger.handlers) < 2

    ## Search for a console logger - its an error if one is found
    console_logger = None
    for h in logger.handlers:
        assert not isinstance(h, logging.StreamHandler)


def test_configure_file_log():

    logger = logging.getLogger(name="floris")

    # Enable file logging at INFO level
    test_level  = "INFO"
    floris.logging_manager.configure_file_log(enabled=True, level=test_level)

    ## The number of handlers should always be 0, 1, or 2. In this case, either 1 or 2 since we've added one above.
    assert 0 < len(logger.handlers) <= 2

    ## Find the file logger - its a FileHandler class
    file_logger = None
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            file_logger = h
    
    ## The level should be set to the value above.
    ## The nameToLevel dictionary converts the string to a numeric value used in the logging library
    assert file_logger.level == logging._nameToLevel[test_level]


    # Change the logging level to ERROR
    test_level  = "ERROR"
    floris.logging_manager.configure_file_log(enabled=True, level=test_level)
    assert 0 < len(logger.handlers) <= 2
    file_logger = None
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            file_logger = h
    assert file_logger.level == logging._nameToLevel[test_level]


    # Disbale file logging
    floris.logging_manager.configure_file_log(enabled=False)

    ## Here, there should be only 0 or 1 handlers
    assert 0 <= len(logger.handlers) < 2

    ## Search for a file logger - its an error if one is found
    file_logger = None
    for h in logger.handlers:
        assert not isinstance(h, logging.FileHandler)
