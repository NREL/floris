# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


"""
The :py:obj:`floris` package contains :py:obj:`src.utilities` module
and the modules that make up the FLORIS software. The floris simulation
modules are used to complete a wake simulation for a given wind farm
and turbine configuration.

All modules and package can be imported with

    >>> import floris

The ``__init__.py`` file enables the import of all modules in this
package so any additional modules should be included there.
"""

from pathlib import Path

from . import (
    utilities,
    wake_velocity,
    logging_manager,
    wake_deflection,
    wake_turbulence,
    wake_combination,
)
from .farm import Farm
from .grid import Grid, TurbineGrid, FlowFieldGrid
from .floris import Floris
from .turbine import Turbine
from .base_class import BaseClass
from .flow_field import FlowField
from .model_generator import model_creator


ROOT = Path(__file__).parent
with open(ROOT.parent / "VERSION") as version_file:
    VERSION = version_file.read().strip()
__version__ = VERSION

# initialize the logger
logging_manager._setup_logger()
