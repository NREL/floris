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


# Show the grid points in hetergenous flow calculation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import floris.tools as wfct


fi = wfct.floris_interface.FlorisInterface("../example_input.json")

# Set layout to 2 turbines
fi.reinitialize_flow_field(layout_array=[[0, 0], [100, 400]])
fi.calculate_wake()

# Introduce variation in wind speed
fi.reinitialize_flow_field(wind_speed=[6, 9], wind_layout=[[-100, 300], [0, 500]])
fi.calculate_wake()

# Show the grid points (note only on turbines, not on wind measurements)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
xs = fi.floris.farm.flow_field.x
ys = fi.floris.farm.flow_field.y
zs = fi.floris.farm.flow_field.z
ax.scatter(xs, ys, zs, marker=".")
ax.set_xlim([0, 1000])
ax.set_ylim([0, 1000])
ax.set_zlim([0, 300])
plt.show()
