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
The :py:obj:`floris.tools` package contains the modules used to drive
FLORIS simulations and perform studies in various areas of research and
analysis.

All modules can be imported with

    >>> import floris.tools

The ``__init__.py`` file enables the import of all modules in this
package so any additional modules should be included there.

Examples:
    >>> import floris.tools

    >>> dir(floris.tools)
    ['__builtins__', '__cached__', '__doc__', '__file__', '__loader__',
    '__name__', '__package__', '__path__', '__spec__', 'cut_plane',
    'floris_interface', 'flow_data',
    'layout_functions', 'optimization', 'plotting', 'power_rose',
    'rews', 'sowfa_utilities', 'visualization', 'wind_rose']
"""

from .floris_interface import FlorisInterface
from .floris_interface_legacy_reader import FlorisInterfaceLegacyV2
from .parallel_computing_interface import ParallelComputingInterface
from .uncertainty_interface import UncertaintyInterface
from .visualization import (
    plot_rotor_values,
    plot_turbines_with_fi,
    visualize_cut_plane,
    visualize_quiver,
)
from .wind_rose import WindRose


# from floris.tools import (
    # cut_plane,
    # floris_interface,
    # flow_data,
    # interface_utilities,
    # layout_functions,
    # optimization,
    # plotting,
    # power_rose,
    # rews,
    # sowfa_utilities,
    # visualization,
    # wind_rose,
# )
