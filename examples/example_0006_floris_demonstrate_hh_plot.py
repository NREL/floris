# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# See read the https://floris.readthedocs.io for documentation

import matplotlib.pyplot as plt
import floris.tools as wfct
import floris.tools.visualization as vis
import floris.tools.cut_plane as cp
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import time

# Define speed limits
minSpeed = 4.
maxSpeed = 8.

# Initialize FLORIS model
fi = wfct.floris_utilities.FlorisInterface("example_input.json")

# Calculate the time to run wake
start = time.time()
fi.calculate_wake()
finish = time.time()
print('Time to calculate flow field', finish - start)

# Get a horizontal cut from default flow field
start = time.time()
hor_plane_1 = wfct.cut_plane.HorPlane(
    fi.get_flow_data(),
    fi.floris.farm.turbines[0].hub_height
)
fi.calculate_wake()
finish = time.time()
print('Time to extract default flow field', finish - start)

# Get a horizontal cut from horizontal methods
start = time.time()
hor_plane_2 = wfct.cut_plane.HorPlane(
    fi.get_hub_height_flow_data(),
    fi.floris.farm.turbines[0].hub_height
)
fi.calculate_wake()
finish = time.time()
print('Time to extract horizontal flow field', finish - start)

# Plot and show they are the same
fig, axarr = plt.subplots(1, 2)
wfct.visualization.visualize_cut_plane(
    hor_plane_1, ax=axarr[0], minSpeed=minSpeed, maxSpeed=maxSpeed)
wfct.visualization.visualize_cut_plane(
    hor_plane_2, ax=axarr[1], minSpeed=minSpeed, maxSpeed=maxSpeed)
plt.show()
