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

import floris.tools.visualization as wakeviz
from floris.tools import cut_plane, FlorisInterface
from floris.tools.visualization import VelocityProfilesFigure


"""
This example illustrates how to plot velocity deficit profiles at
several location downstream of a turbine. Here we use the following definition:
    velocity_deficit = (homogeneous_wind_speed - u) / homogeneous_wind_speed
        , where u is the wake velocity obtained when the incoming wind speed is the
        same at all heights and equal to `homogeneous_wind_speed`.
"""

if __name__ == '__main__':
    D = 126.0 # Turbine diameter
    hub_height = 90.0
    fi = FlorisInterface("inputs/gch.yaml")
    fi.reinitialize(layout_x=[0.0], layout_y=[0.0])

    downstream_dists = D * np.array([3, 5, 7])
    homogeneous_wind_speed = 8.0

    # Sample three velocity deficit profiles along three corresponding lines that are all
    # parallel to the y-axis (cross-stream direction). The streamwise location of each line
    # is given in `downstream_dists`.
    profiles = fi.sample_velocity_deficit_profiles(
        direction='cross-stream',
        downstream_dists=downstream_dists,
        homogeneous_wind_speed=homogeneous_wind_speed,
    )

    horizontal_plane = fi.calculate_horizontal_plane(height=hub_height)
    fig, ax = plt.subplots(figsize=(6.4, 3))
    wakeviz.visualize_cut_plane(horizontal_plane, ax)
    colors = ['b', 'g', 'c']
    for i, df in enumerate(profiles):
        ax.plot(df['x'], df['y'], colors[i], label=f'x/D={downstream_dists[i] / D:.1f}')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Streamwise velocity in a horizontal plane: gauss velocity model')
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    ax.legend(bbox_to_anchor=[1.29, 1.04])

    # Initialize a VelocityProfilesFigure. The workflow is similar to a matplotlib Figure:
    # Initialize it, plot data, and then customize it further if needed.
    profiles_fig = VelocityProfilesFigure(
        downstream_dists_D=downstream_dists / D,
        layout=['cross-stream'],
        coordinate_labels=['x/D', 'y/D']
    )
    # Add profiles to the figure. This method automatically determines the direction and
    # streamwise location of each profile from the profile coordinates.
    profiles_fig.add_profiles(profiles, color='k')

    # Change velocity model to jensen, get the velocity deficit profiles,
    # and add them to the figure.
    floris_dict = fi.floris.as_dict()
    floris_dict['wake']['model_strings']['velocity_model'] = 'jensen'
    fi = FlorisInterface(floris_dict)
    profiles = fi.sample_velocity_deficit_profiles(
        direction='cross-stream',
        downstream_dists=downstream_dists,
        homogeneous_wind_speed=homogeneous_wind_speed,
        resolution=400,
    )
    profiles_fig.add_profiles(profiles, color='r')

    margin = 0.05
    profiles_fig.set_xlim([0.0 - margin, 0.6 + margin])
    profiles_fig.add_ref_lines_x2([-0.5, 0.5])

    profiles_fig.axs[0,0].legend(['gauss', 'jensen'], fontsize=11)
    profiles_fig.fig.suptitle(
        'Velocity deficit profiles from different velocity models',
        fontsize=14,
    )


    plt.show()
