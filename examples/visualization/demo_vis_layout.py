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

import numpy as np
import matplotlib.pyplot as plt

import floris.tools as wfct


# Initialize the FLORIS interface fi
fi = wfct.floris_interface.FlorisInterface("../example_input.json")

# Make a random 9-turbine layout
layout_x = np.random.uniform(low=0, high=126 * 10, size=4)
layout_y = np.random.uniform(low=0, high=126 * 10, size=4)
fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))
fi.calculate_wake()

# Show layout visualizations
fig, axarr = plt.subplots(1, 3, figsize=(15, 5))

ax = axarr[0]
hor_plane = fi.get_hor_plane()
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)

ax = axarr[1]
fi.vis_layout(ax=ax)

ax = axarr[2]
fi.vis_layout(ax=ax, show_wake_lines=True)

plt.show()
