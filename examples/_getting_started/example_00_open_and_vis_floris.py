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
from floris.tools.visualization import visualize_cut_plane


# Initialize the FLORIS Interface, `fi`.
# For basic usage, the FLORIS Interface provides a simplified
# and expressive interface to the simulation routines.

fi = FlorisInterface("../example_input.yaml")

# Yaw the leading turbine 20 degrees
yaw_angles = np.zeros((1,1,3))
yaw_angles[:,:,0] = 20.0

# Using the FlorisInterface functions for generating plots,
# run FLORIS and extract 2D planes of data.
horizontal_plane = fi.get_hor_plane(yaw_angles=yaw_angles, x_resolution=200, y_resolution=100)
cross_plane = fi.get_cross_plane(yaw_angles=yaw_angles, y_resolution=100, z_resolution=100)
y_plane = fi.get_y_plane(yaw_angles=yaw_angles, x_resolution=200, z_resolution=100)

# Create the plots
fig, ax_list = plt.subplots(3, 1)
ax_list = ax_list.flatten()

visualize_cut_plane(horizontal_plane, ax=ax_list[0])
visualize_cut_plane(y_plane, ax=ax_list[1])
visualize_cut_plane(cross_plane, ax=ax_list[2])

plt.show()
