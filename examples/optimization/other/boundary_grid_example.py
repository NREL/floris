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
import pandas as pd
import matplotlib.pyplot as plt

import floris.tools as wfct
import floris.tools.optimization.other.boundary_grid as BG


# Instantiate the FLORIS object
file_dir = os.path.dirname(os.path.abspath(__file__))
fi = wfct.floris_interface.FlorisInterface(
    os.path.join(file_dir, "../../example_input.json")
)

bg = BG.BoundaryGrid(fi)
n_boundary_turbs = 30
start = 0.0
nrows = 20
ncols = 10
farm_width = 10000
farm_height = 10000
shear = np.deg2rad(10.0)
rotation = np.deg2rad(30.0)
center_x = 0.0
center_y = 0.0
shrink_boundary = 500.0
boundary_x = np.array([-5000.0, -5000.0, 5000.0, 5000.0, -5000.0])
boundary_y = np.array([-5000.0, 5000.0, 7000.0, -5000.0, -5000.0])


bg.reinitialize_bg(
    n_boundary_turbs=n_boundary_turbs,
    start=start,
    nrows=nrows,
    ncols=ncols,
    farm_width=farm_width,
    farm_height=farm_height,
    shear=shear,
    rotation=rotation,
    center_x=center_x,
    center_y=center_y,
    shrink_boundary=shrink_boundary,
    boundary_x=boundary_x,
    boundary_y=boundary_y,
)

bg.reinitialize_xy()

wd = np.array([30.0])
ws = np.array([10.0])
wf = np.array([1.0])
AEP = fi.get_farm_AEP(wd, ws, wf)
print("AEP: ", AEP)

fi.reinitialize_flow_field(wind_direction=wd[0], wind_speed=ws[0])
fi.calculate_wake()
hor_plane = fi.get_hor_plane()

# Plot and show
# fig, ax = plt.subplots()
plt.figure(figsize=(6, 6))
ax = plt.gca()
ax.plot(boundary_x, boundary_y)
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)

plt.show()
