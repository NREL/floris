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

"""

# Initialize the FLORIS interface for 3 different models
fi_jensen = FlorisInterface("inputs/jensen.yaml")
fi_gch = FlorisInterface("inputs/gch.yaml")
fi_cc = FlorisInterface("inputs/cc.yaml")

fig, axarr = plt.subplots(2, 2, figsize=(16, 4))

# Iterate over the fi-objects plotting a horizontal slice of the flow fields
# for each model and configuration.
MIN_WS = 2.0
MAX_WS = 8.0
# for idx, (fi, name) in enumerate(zip([fi_jensen, fi_gch, fi_cc], ["Jensen", "Gaussian", "Cumulative"])):
for idx, (fi, name) in enumerate(zip([fi_jensen, fi_gch], ["Jensen", "Gaussian"])):

    # Aligned
    ax = axarr[0, idx]
    horizontal_plane = fi.get_hor_plane()
    visualize_cut_plane(horizontal_plane, ax=ax, minSpeed=MIN_WS, maxSpeed=MAX_WS)
    ax.set_title(name)
    axarr[0, 0].set_ylabel("Aligned")

    # Yawed
    yaw_angles = np.zeros_like(fi.floris.farm.yaw_angles)
    yaw_angles[:,:,0] = 25.0
    ax = axarr[1, idx]
    horizontal_plane = fi.get_hor_plane(yaw_angles=yaw_angles)
    visualize_cut_plane(horizontal_plane, ax=ax, minSpeed=MIN_WS, maxSpeed=MAX_WS)
    axarr[1, 0].set_ylabel("Yawed")

# Show the figure
plt.show()
