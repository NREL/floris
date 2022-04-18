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


import matplotlib.pyplot as plt
import numpy as np

from floris.tools import FlorisInterface

# Initialize FLORIS with the given input file via FlorisInterface.
# For basic usage, FlorisInterface provides a simplified and expressive
# entry point to the simulation routines.
fi = FlorisInterface("inputs/gch.yaml")

# Create a 4-turbine layouts
fi.reinitialize( layout=( [0, 0., 500., 500.], [0., 300., 0., 300.] ) )

# Calculate wake
fi.calculate_wake()

# Collect the wind speed at all the turbine points
u_points = fi.floris.flow_field.u

print('U points is 1 wd x 1 ws x 4 turbines x 3 x 3 points (turbine_grid_points=3)')
print(u_points.shape)

# Collect the average wind speeds from each turbine
avg_vel = fi.get_turbine_average_velocities()

print('Avg vel is 1 wd x 1 ws x 4 turbines')
print(avg_vel.shape)

# Show that one is equivalent to the other following averaging
print('Avg Vel is determined by taking the cube root of mean of the cubed value across the points')
print('Average velocity: ', avg_vel)
print('Recomputed:       ', np.cbrt(np.mean(u_points**3, axis=(3,4))))
