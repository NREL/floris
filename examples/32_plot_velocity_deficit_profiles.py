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
from matplotlib import ticker

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
    homogeneous_wind_speed = 8.0

    fi = FlorisInterface("inputs/gch.yaml")
    fi.reinitialize(layout_x=[0.0], layout_y=[0.0])

    # ------------------------------ Single-turbine layout ------------------------------
    # We first show how to sample and plot velocity deficit profiles on a single-turbine layout.
    # Lines are drawn on a horizontal_plane to indicate were the velocity is sampled.
    downstream_dists = D * np.array([3, 5, 7])
    # Sample three profiles along three corresponding lines that are all parallel to the y-axis
    # (cross-stream direction). The streamwise location of each line is given in `downstream_dists`.
    profiles = fi.sample_velocity_deficit_profiles(
        direction='cross-stream',
        downstream_dists=downstream_dists,
        homogeneous_wind_speed=homogeneous_wind_speed,
    )

    horizontal_plane = fi.calculate_horizontal_plane(height=hub_height)
    fig, ax = plt.subplots(figsize=(6.4, 3))
    wakeviz.visualize_cut_plane(horizontal_plane, ax)
    colors = ['b', 'g', 'c']
    for i, profile in enumerate(profiles):
        ax.plot(profile['x'], profile['y'], colors[i], label=f'x/D={downstream_dists[i] / D:.1f}')
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
    # The dashed reference lines show the extent of the rotor
    profiles_fig.add_ref_lines_x2([-0.5, 0.5])
    for ax in profiles_fig.axs[0]:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))

    profiles_fig.axs[0,0].legend(['gauss', 'jensen'], fontsize=11)
    profiles_fig.fig.suptitle(
        'Velocity deficit profiles from different velocity models',
        fontsize=14,
    )

    # -------------------------------- Two-turbine layout --------------------------------
    # Show that the coordinates x1, x2, x3 returned by
    # sample_velocity_deficit_profiles are relative to the sampling starting point.
    # By default, this starting point is at (0.0, 0.0, fi.floris.flow_field.reference_wind_height).
    # It also rotates with the wind direction.
    downstream_dists = D * np.array([3, 5])
    floris_dict = fi.floris.as_dict()
    floris_dict['wake']['model_strings']['velocity_model'] = 'gauss'
    fi = FlorisInterface(floris_dict)
    fi.reinitialize(wind_directions=[315.0], layout_x=[0.0, 2 * D], layout_y=[0.0, -2 * D])

    cross_profiles = fi.sample_velocity_deficit_profiles(
        direction='cross-stream',
        downstream_dists=downstream_dists,
        homogeneous_wind_speed=homogeneous_wind_speed,
        x_start= 2 * D,
        y_start=-2 * D,
    )

    horizontal_plane = fi.calculate_horizontal_plane(height=hub_height, x_bounds=[-2 * D, 9 * D])
    ax = wakeviz.visualize_cut_plane(horizontal_plane)
    colors = ['b', 'g', 'c']
    for i, df in enumerate(cross_profiles):
        ax.plot(df['x'], df['y'], colors[i], label=f'$x_1/D={downstream_dists[i] / D:.1f}$')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Streamwise velocity in a horizontal plane')
    ax.legend()

    plt.quiver(
        [ 2 * D,  2 * D],
        [-2 * D, -2 * D],
        [ D, D],
        [-D, D],
        angles='xy',
        scale_units='xy',
        scale=1,
    )
    plt.text(3.2 * D, -2.7 * D, '$x_1$', bbox={'facecolor': 'white'})
    plt.text(3.25 * D, -0.95 * D, '$x_2$', bbox={'facecolor': 'white'})

    vertical_profiles = fi.sample_velocity_deficit_profiles(
        direction='vertical',
        profile_range=[-hub_height, 4 * D - hub_height],
        downstream_dists=downstream_dists,
        homogeneous_wind_speed=homogeneous_wind_speed,
        x_start= 2 * D,
        y_start=-2 * D,
    )

    profiles_fig = VelocityProfilesFigure(
        downstream_dists_D=downstream_dists / D,
        layout=['cross-stream', 'vertical'],
    )
    profiles_fig.add_profiles(cross_profiles + vertical_profiles, color='k')

    profiles_fig.add_ref_lines_x3([-hub_height / D], linestyle='-', linewidth=2.5)
    profiles_fig.set_xlim([0.0 - margin, 0.8 + margin])
    for ax in profiles_fig.axs[0]:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.4))

    profiles_fig.fig.suptitle(
        'Cross-stream profiles at hub-height, and\nvertical profiles at $x_2 = 0$',
        fontsize=14,
    )

    plt.show()
