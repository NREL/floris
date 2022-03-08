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
from floris.tools.visualization import visualize_cut_plane


if __name__ == "__main__":
    # Data
    fi = FlorisInterface("inputs/gch.yaml")
    wind_directions = [225.0, 270.0, 315.0]
    layout = [[500.0, 1000.0, 1500.0], [0.0, 0.0, 0.0]]

    # For each wind direction
    for wd in wind_directions:

        # Compute wakes
        fi.reinitialize(layout=layout, wind_directions=[wd], wind_speeds=[7.0])
        fi.calculate_wake(yaw_angles=None)

        # Get horizontal plane
        hor_plane = fi.calculate_horizontal_plane(
            height=90.0,
            yaw_angles=None,
            north_up=True,
        )

        # Plot and save figure
        figure = plt.figure()
        render_ax = figure.add_subplot(111)
        im = visualize_cut_plane(hor_plane, ax=render_ax)
        render_ax.set_title("wind_direction={}°".format(wd))
        plt.show()
