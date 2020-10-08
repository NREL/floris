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
fi.reinitialize_flow_field(layout_array=[[0, 500], [0, 0]])
fi.calculate_wake()

# Show the grid points (note only on turbines, not on wind measurements)
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(131, projection="3d")
xs = fi.floris.farm.flow_field.x
ys = fi.floris.farm.flow_field.y
zs = fi.floris.farm.flow_field.z
ax.scatter(xs, ys, zs, marker=".")

# Show the turbine points in this case
for coord, turbine in fi.floris.farm.turbine_map.items:
    xt, yt, zt = turbine.return_grid_points(coord)
    ax.scatter(xt, yt, zt, marker="o", color="r", alpha=0.25)

ax.set_xlim([0, 600])
ax.set_ylim([-150, 150])
ax.set_zlim([0, 300])
ax.set_title("Initial Rotor Points")
ax.set_xlabel("Streamwise [m]")
ax.set_ylabel("Spanwise [m]")
ax.set_zlabel("Vertical [m]")

# Raise the hub height of the second turbine and show
fi.change_turbine([0], {"hub_height": 150})
fi.calculate_wake()
ax = fig.add_subplot(132, projection="3d")
xs = fi.floris.farm.flow_field.x
ys = fi.floris.farm.flow_field.y
zs = fi.floris.farm.flow_field.z
ax.scatter(xs, ys, zs, marker=".")

# Show the turbine points in this case
for coord, turbine in fi.floris.farm.turbine_map.items:
    xt, yt, zt = turbine.return_grid_points(coord)
    ax.scatter(xt, yt, zt, marker="o", color="r", alpha=0.25)

ax.set_xlim([0, 600])
ax.set_ylim([-150, 150])
ax.set_zlim([0, 300])
ax.set_title("Raise First Hub Height")
ax.set_xlabel("Streamwise [m]")
ax.set_ylabel("Spanwise [m]")
ax.set_zlabel("Vertical [m]")

# Increase the first turbine rotor_diameter and plot
fi.change_turbine([0], {"rotor_diameter": 250})
fi.calculate_wake()
ax = fig.add_subplot(133, projection="3d")
xs = fi.floris.farm.flow_field.x
ys = fi.floris.farm.flow_field.y
zs = fi.floris.farm.flow_field.z
ax.scatter(xs, ys, zs, marker=".")

# Show the turbine points in this case
for coord, turbine in fi.floris.farm.turbine_map.items:
    xt, yt, zt = turbine.return_grid_points(coord)
    ax.scatter(xt, yt, zt, marker="o", color="r", alpha=0.25)

ax.set_xlim([0, 600])
ax.set_ylim([-150, 150])
ax.set_zlim([0, 300])
ax.set_title("Increase First Diameter")
ax.set_xlabel("Streamwise [m]")
ax.set_ylabel("Spanwise [m]")
ax.set_zlabel("Vertical [m]")

plt.show()
