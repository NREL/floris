# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# See read the https://floris.readthedocs.io for documentation


# Short demo of how probe points currently added for visualization and demonstration of speed

import matplotlib.pyplot as plt
import floris.tools as wfct
import numpy as np
import time

# Initialize the FLORIS interface fi
fi = wfct.floris_interface.FlorisInterface("example_input.json")


## Parameters
d_space = 7
d = 126
dist = d * d_space

# Calculate wake
fi.calculate_wake()

# Declare plots
fig, axarr = plt.subplots(2,2,figsize=(10,10))

for n_row, ax in zip([5,6,7,8],axarr.flatten()):

    x_array = []
    y_array = []

    for x in np.arange(0,dist * n_row,dist):
        for y in np.arange(0,dist * n_row,dist):
            x_array.append(x)
            y_array.append(y)

    fi.reinitialize_flow_field(layout_array=(x_array,y_array))
    fi.calculate_wake()

    # Use the new method=====================================
    start = time.time()

    hor_plane = fi.get_hor_plane()
    wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
    end = time.time()
    ax.set_title('%d Turbs, time = %.1fs' % (n_row*n_row,end-start))

# Show the plot
plt.show()