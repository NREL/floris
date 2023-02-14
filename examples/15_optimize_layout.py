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


import os
import numpy as np

from floris.tools import FlorisInterface

from floris.tools.optimization.layout_optimization.layout_optimization_scipy import LayoutOptimizationScipy

"""
This example shows a simple layout optimization using the python module Scipy.

A 4 turbine array is optimized such that the layout of the turbine produces the
highest annual energy production (AEP) based on the given wind resource. The turbines
are constrained to a square boundary and a random wind resource is supplied. The results
of the optimization show that the turbines are pushed to the outer corners of the boundary,
which makes sense in order to maximize the energy production by minimizing wake interactions.
"""

# Initialize the FLORIS interface fi
file_dir = os.path.dirname(os.path.abspath(__file__))
fi = FlorisInterface('inputs/gch.yaml')

# Setup 72 wind directions with a random wind speed and frequency distribution
wind_directions = np.arange(0, 360.0, 5.0)
np.random.seed(1)
wind_speeds = 8.0 + np.random.randn(1) * 0.5
# Shape frequency distribution to match number of wind directions and wind speeds
freq = np.abs(np.sort(np.random.randn(len(wind_directions)))).reshape((len(wind_directions), len(wind_speeds)))
freq = freq / freq.sum()

fi.reinitialize(wind_directions=wind_directions, wind_speeds=wind_speeds)

# The boundaries for the turbines, specified as vertices
boundaries = [(0.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0), (1000.0, 0.0), (0.0, 0.0)]

# Set turbine locations to 4 turbines in a rectangle
D = 126.0 # rotor diameter for the NREL 5MW
layout_x = [0, 0, 6 * D, 6 * D]
layout_y = [0, 4 * D, 0, 4 * D]
fi.reinitialize(layout_x=layout_x, layout_y=layout_y)

# Setup the optimization problem
layout_opt = LayoutOptimizationScipy(fi, boundaries, freq=freq)

# Run the optimization
sol = layout_opt.optimize()

# Get the resulting improvement in AEP
print('... calcuating improvement in AEP')
fi.calculate_wake()
base_aep = fi.get_farm_AEP(freq=freq) / 1e6
fi.reinitialize(layout_x=sol[0], layout_y=sol[1])
fi.calculate_wake()
opt_aep = fi.get_farm_AEP(freq=freq) / 1e6
percent_gain = 100 * (opt_aep - base_aep) / base_aep

# Print and plot the results
print('Optimal layout: ', sol)
print('Optimal layout improves AEP by %.1f%% from %.1f MWh to %.1f MWh' % (percent_gain, base_aep, opt_aep))
layout_opt.plot_layout_opt_results()