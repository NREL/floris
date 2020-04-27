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
 
"""
The :py:obj:`floris.simulation` package contains the modules used to
complete a wake simulation for a given wind farm and turbine
configuration.

All classes can be be imported with

    >>> import floris.simulation

The ``__init__.py`` file enables the import of all modules in this
package so any additional modules should be included there.

Examples:
    >>> import floris.simulation
    
    >>> dir(floris.simulation)
    ['Farm', 'Floris', 'FlowField', 'InputReader', 'Turbine',
    'TurbineMap', 'Wake', 'WindMap', '__builtins__', '__cached__',
    '__doc__', '__file__', '__loader__', '__name__', '__package__',
    '__path__', '__spec__', 'farm', 'floris', 'flow_field', 'input_reader',
    'turbine', 'turbine_map', 'wake', 'wake_combination', 'wake_deflection',
    'wake_turbulence', 'wake_velocity', 'wind_map']
"""

from .farm import Farm
from .floris import Floris
from .flow_field import FlowField
from .input_reader import InputReader
from .wind_map import WindMap
from .turbine_map import TurbineMap
from .turbine import Turbine
from . import wake_deflection
from . import wake_velocity
from . import wake_combination
from . import wake_turbulence
from .wake import Wake
