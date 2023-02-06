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
The :py:obj:`floris` package contains :py:obj:`floris.utilities` module
and the modules that make up the FLORIS software. The floris simulation
modules are used to complete a wake simulation for a given wind farm
and turbine configuration.

All modules and package can be imported with

    >>> import floris

The ``__init__.py`` file enables the import of all modules in this
package so any additional modules should be included there.

isort:skip_file
"""

# Provide full-path imports here for all modules
# that should be included in the simulation package.
# Since some of these depend on each other, the order
# that they are listed here does matter.

import floris.logging_manager

from .base import BaseClass, BaseModel, State
from .turbine import average_velocity, axial_induction, Ct, power, Turbine
from .farm import Farm
from .grid import FlowFieldGrid, FlowFieldPlanarGrid, Grid, TurbineGrid
from .flow_field import FlowField
from .wake import WakeModelManager
from .solver import (
    cc_solver,
    full_flow_cc_solver,
    full_flow_sequential_solver,
    full_flow_turbopark_solver,
    sequential_solver,
    turbopark_solver,
)
from .floris import Floris

# initialize the logger
floris.logging_manager._setup_logger()
