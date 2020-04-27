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
The :py:obj:`floris` package contains :py:obj:`floris.utilities` module
and the packages which make up the FLORIS software. The
:py:obj:`floris.simulation` package contains the modules used to
complete a wake simulation for a given wind farm and turbine
configuration. The :py:obj:`floris.tools` package contains the modules
used to drive FLORIS simulations and perform studies in various areas
of research and analysis. 

All modules and package can be imported with

    >>> import floris

The ``__init__.py`` file enables the import of all modules in this
package so any additional modules should be included there.

Examples:
    >>> import floris
    
    >>> dir(floris)
    ['__builtins__', '__cached__', '__doc__', '__file__', '__loader__',
    '__name__', '__package__', '__path__', '__spec__', 'simulation',
    'tools', 'utilities']

    >>> dir(floris.utilities)
    [Vec3', '__builtins__', '__cached__', '__doc__',
    '__file__', '__loader__', '__name__', '__package__', '__spec__',
    'cosd', 'np', 'sind', 'tand', 'wrap_180', 'wrap_360']

    >>> dir(floris.simulation)
    ['Farm', 'Floris', 'FlowField', 'InputReader', 'Turbine',
    'TurbineMap', 'Wake', 'WakeCombination', 'VelocityDeflection',
    'VelocityDeficit', '__builtins__', '__cached__', '__doc__',
    '__file__', '__loader__', '__name__', '__package__', '__path__',
    '__spec__', 'farm', 'floris', 'flow_field', 'input_reader',
    'turbine', 'turbine_map', 'wake', 'wake_combination',
    'wake_deflection', 'wake_velocity']

    >>> dir(floris.tools)
    ['__builtins__', '__cached__', '__doc__', '__file__', '__loader__',
    '__name__', '__package__', '__path__', '__spec__', 'cut_plane',
    'energy_ratio', 'floris_interface', 'flow_data',
    'layout_functions', 'optimization', 'plotting', 'power_rose',
    'rews', 'sowfa_utilities', 'visualization', 'wind_rose']
"""

from . import utilities
from . import simulation
from . import tools
from .tools import optimization
