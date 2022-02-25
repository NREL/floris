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


import os
import numpy as np

from floris.tools import FlorisInterface
import floris.tools.optimization.pyoptsparse as opt

"""
This example shows a simple layout optimization using the python module pyOptSparse.

A 4 turbine array is optimized such that the layout of the turbine produces the
highest annual energy production (AEP) based on the given wind resource. The turbines
are constrained to a square boundary and a randomw wind resource is supplied. The results
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
freq = np.abs(np.sort(np.random.randn(len(wind_directions))))
freq = freq / freq.sum()
fi.reinitialize(wind_directions=wind_directions, wind_speeds=wind_speeds)

# The boundaries for the turbines, specified as vertices
boundaries = [(0.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0), (1000.0, 0.0), (0.0, 0.0)]

# Set turbine locations to 4 turbines in a rectangle
D = 126.0 # rotor diameter for the NREL 5MW
layout_x = [0, 0, 6 * D, 6 * D]
layout_y = [0, 4 * D, 0, 4 * D]
fi.reinitialize(layout=(layout_x, layout_y))
fi.calculate_wake()

# Setup the optimization problem
model = opt.layout.Layout(fi, boundaries, freq)
tmp = opt.optimization.Optimization(model=model, solver='SLSQP')

# Run the optimization
sol = tmp.optimize()

# Print and plot the results
print(sol)
model.plot_layout_opt_results(sol)
