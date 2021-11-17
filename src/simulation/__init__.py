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

from src.simulation.farm import Farm
from src.simulation.grid import Grid, TurbineGrid  # , FlowFieldGrid
from src.simulation.wake import Wake
from src.simulation.floris import Floris
from src.simulation.solver import sequential_solver
from src.simulation.turbine import Ct, Turbine, power, axial_induction, average_velocity

# Provide full-path imports here for all modules
# that should be included in the simulation package.
# Since some of these depend on each other, the order
# that they are listed here does matter.
from src.simulation.base_class import BaseClass
from src.simulation.flow_field import FlowField


ROOT = Path(__file__).parent.parent.parent
with open(ROOT / "VERSION") as version_file:
    VERSION = version_file.read().strip()
__version__ = VERSION

# initialize the logger
import src.logging_manager


src.logging_manager._setup_logger()
