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
from floris.tools.visualization import VelocityProfilesFigure


"""
The first part of this example illustrates how to plot velocity deficit profiles at
several location downstream of a turbine. Here we use the following definition:
    velocity_deficit = (homogeneous_wind_speed - u) / homogeneous_wind_speed
        , where u is the wake velocity obtained when the incoming wind speed is the
        same at all heights and equal to `homogeneous_wind_speed`.
The second part of the example shows a special case of how the profiles are affected
by a change in wind direction as well as a change in turbine location and sampling
starting point.
"""

def get_profiles(direction, resolution=100):
    return fi.sample_velocity_deficit_profiles(
        direction=direction,
        downstream_dists=downstream_dists,
        resolution=resolution,
        homogeneous_wind_speed=homogeneous_wind_speed,
    )

if __name__ == '__main__':
    D = 126.0 # Turbine diameter
    hub_height = 90.0
    fi = FlorisInterface("inputs/gch.yaml")
    fi.reinitialize(layout_x=[0.0], layout_y=[0.0])

    downstream_dists = D * np.array([3, 5, 7])
    homogeneous_wind_speed = 8.0

    # Below, `get_profiles('y')` returns three velocity deficit profiles. These are sampled along
    # three corresponding lines that are all parallel to the y-axis (cross-stream direction).
    # The streamwise location of each line is given in `downstream_dists`.
    # Similarly, `get_profiles('z')` samples profiles in the vertical direction (z) at the
    # same streamwise locations.
    profiles = get_profiles('y') + get_profiles('z')

    # Initialize a VelocityProfilesFigure. The workflow is similar to a matplotlib Figure:
    # Initialize it, plot data, and then customize it further if needed.
    # The provided value of `layout` puts y-profiles on the top row of the figure and
    # z-profiles on the bottom row.
    profiles_fig = VelocityProfilesFigure(
        downstream_dists_D=downstream_dists / D,
        layout=['y', 'z'],
    )
    # Add profiles to the figure. This method automatically determines the direction and
    # streamwise location of each profile from the profile coordinates.
    profiles_fig.add_profiles(profiles, color='k')

    # Change velocity model to jensen, get the velocity deficit profiles,
    # and add them to the figure.
    floris_dict = fi.floris.as_dict()
    floris_dict['wake']['model_strings']['velocity_model'] = 'jensen'
    fi = FlorisInterface(floris_dict)
    profiles_y = get_profiles('y', resolution=400)
    profiles_z = get_profiles('z', resolution=400)
    profiles_fig.add_profiles(profiles_y + profiles_z, color='r')

    margin = 0.05
    profiles_fig.set_xlim([0.0 - margin, 0.6 + margin])
    profiles_fig.add_ref_lines_y([-0.5, 0.5])
    profiles_fig.add_ref_lines_z([-0.5, 0.5])

    profiles_fig.axs[0,0].legend(['gauss', 'jensen'], fontsize=11)
    profiles_fig.fig.suptitle(
        'Velocity decifit profiles from different velocity models',
        fontsize=14,
    )

    # Second part of example:
    # Case 1: Show that, if we have a single turbine, then the profiles are independent of
    # the wind direction. This is because x is defined to be in the streamwise direction
    # in sample_velocity_deficit_profiles and VelocityProfilesFigure.
    # Case 2: Show that the coordinates x/D, y/D and z/D returned by
    # sample_velocity_deficit_profiles are relative to the sampling starting point.
    # By default, this starting point is at (0.0, 0.0, fi.floris.flow_field.reference_wind_height)
    # in inertial coordinates.
    downstream_dists = D * np.array([3])
    for case in [1, 2]:
        # The first added profile is a reference
        fi = FlorisInterface("inputs/gch.yaml")
        fi.reinitialize(layout_x=[0.0], layout_y=[0.0])
        profiles = get_profiles('y', resolution=400)
        profiles_fig = VelocityProfilesFigure(
            downstream_dists_D=downstream_dists / D,
            layout=['y'],
            ax_width=3.5,
            ax_height=3.5,
        )
        profiles_fig.add_profiles(profiles, color='k')

        if case == 1:
            # Change wind direction compared to reference
            wind_directions = [315.0]
            layout_x, layout_y = [0.0], [0.0]
            # Same as the default starting point but specified for completeness
            x_inertial_start, y_inertial_start = 0.0, 0.0
        elif case == 2:
            # Change turbine location compared to reference. Then, set the sampling starting
            # point to the new turbine location using the arguments
            # `x_inertial_start` and `y_inertial_start`.
            wind_directions = [270.0]
            layout_x, layout_y = [D], [D]
            x_inertial_start, y_inertial_start = D, D

        # Plot a second profile to show that it is equivalent to the reference
        fi.reinitialize(wind_directions=wind_directions, layout_x=layout_x, layout_y=layout_y)
        profiles = fi.sample_velocity_deficit_profiles(
            direction='y',
            downstream_dists=downstream_dists,
            resolution=21,
            homogeneous_wind_speed=homogeneous_wind_speed,
            x_inertial_start=x_inertial_start,
            y_inertial_start=y_inertial_start,
        )
        profiles_fig.add_profiles(
            profiles,
            linestyle='None',
            marker='o',
            color='b',
            markerfacecolor='None',
            markeredgewidth=1.3,
        )

        if case == 1:
            profiles_fig.axs[0,0].legend(['WD = 270 deg', 'WD = 315 deg'])
        elif case == 2:
            profiles_fig.fig.suptitle(
                'Legend (x, y) locations in inertial coordinates.\n'
                'x/D and y/D relative to sampling start point',
            )
            profiles_fig.axs[0,0].legend([
                'turbine location: (0, 0)\nsampling start point: (0, 0)',
                'turbine location: (D, D)\nsampling start point: (D, D)',
            ])

    plt.show()
