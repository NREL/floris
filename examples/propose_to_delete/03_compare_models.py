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
from floris.tools import visualize_cut_plane

"""
This example visually compares different models included in FLORIS.
We run the same simulation for each model and plot a horizontal slice
with and without yawed turbines for comparison.
"""

# Initialize FLORIS for 3 different models via FlorisInterface
fi_jensen = FlorisInterface("inputs/jensen.yaml")
fi_gch = FlorisInterface("inputs/gch.yaml")
fi_cc = FlorisInterface("inputs/cc.yaml")

# Create the plotting objects using matplotlib
fig, axarr = plt.subplots(2, 2, figsize=(16, 4))

# Iterate over the model-objects and create a plot of the flow fields
# for each model and configuration.
MIN_WS = 2.0
MAX_WS = 8.0
for idx, (fi, name) in enumerate(zip([fi_jensen, fi_gch], ["Jensen", "Gaussian"])):

    # Aligned
    ax = axarr[0, idx]
    horizontal_plane = fi.calculate_horizontal_plane()
    visualize_cut_plane(horizontal_plane, ax=ax, minSpeed=MIN_WS, maxSpeed=MAX_WS)
    ax.set_title(name)
    axarr[0, 0].set_ylabel("Aligned")

    # Yawed
    yaw_angles = np.zeros_like(fi.floris.farm.yaw_angles)
    yaw_angles[:,:,0] = 25.0
    ax = axarr[1, idx]
    horizontal_plane = fi.calculate_horizontal_plane(yaw_angles=yaw_angles)
    visualize_cut_plane(horizontal_plane, ax=ax, minSpeed=MIN_WS, maxSpeed=MAX_WS)
    axarr[1, 0].set_ylabel("Yawed")

plt.show()
