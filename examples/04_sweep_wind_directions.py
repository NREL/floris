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


from time import perf_counter as timerpc
import matplotlib.pyplot as plt
import numpy as np

from floris.tools import FlorisInterface
from floris.tools.visualization import visualize_cut_plane


"""
04_sweep_wind_directions

This example demonstrates vectorization of wind direction.  
A vector of wind directions is passed to the intialize function 
and the powers of the two simulated turbines is computed for all
wind directions in one call

The power of both turbines for each wind direction is then plotted

"""

# Initialize FLORIS object
X, Y = np.meshgrid(np.arange(50) * 5 * 126.4, np.arange(2) * 3 * 126.4)
fi = FlorisInterface("inputs/gch.yaml")
fi.reinitialize(
    wind_directions=np.arange(0.0, 360.0, 3.0),
    wind_speeds=np.arange(4.0, 8.0, 1.0),
    layout=[X.flatten(), Y.flatten()]
)

# Calculate without parallelization
t0 = timerpc()
fi.calculate_wake()
print("Time spent (non-parallelized): {:.3f} s".format(timerpc() - t0))
turbine_powers = fi.get_turbine_powers() / 1E3 # In kW

# Calculate with parallelization
t0 = timerpc()
fi.calculate_wake(num_tasks=4)
print("Time spent (parallelized): {:.3f} s".format(timerpc() - t0))
turbine_powers = fi.get_turbine_powers() / 1E3 # In kW
