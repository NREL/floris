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

from floris.tools import FlorisInterface
from floris.tools.layout_functions import visualize_layout


"""
This example visualizes a wind turbine layout
using the visualize_layout function
"""

# Declare a FLORIS interface
fi = FlorisInterface("inputs/gch.yaml")

# Assign a 6-turbine layout
fi.reinitialize(layout_x=[0, 100, 500, 1000, 1200,500], layout_y=[0, 800, 150, 500, 0,500])

# Give turbines specific names
turbine_names = ['T01', 'T02','T03','S01','X01', 'X02']

# Declare a 4-pane plot
fig, axarr = plt.subplots(2,2, sharex=True, sharey=True, figsize=(14,10))

# Show the layout with all defaults

# Default visualization
ax = axarr[0,0]
visualize_layout(fi, ax=ax)
ax.set_title('Default visualization')

# With wake lines
ax = axarr[0,1]
visualize_layout(fi, ax=ax, show_wake_lines=True)
ax.set_title('Show wake lines')

# Limit wake lines and use provided
ax = axarr[1,0]
visualize_layout(
    fi,
    ax=ax,
    show_wake_lines=True,
    lim_lines_per_turbine=2,
    turbine_names=turbine_names
)
ax.set_title('Show only nearest 2, use provided names')

# Show rotors and use black and white
ax = axarr[1,1]
visualize_layout(
    fi,
    ax=ax,
    show_wake_lines=True,
    lim_lines_per_turbine=2,
    plot_rotor=True,
    black_and_white=True
)
ax.set_title('Plot rotors and use black and white option')



plt.show()
