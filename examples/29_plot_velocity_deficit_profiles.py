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
from floris.tools import FlorisInterface
from floris.tools.visualization import VelocityProfilesFigure


"""
docstr
"""

def get_profiles(direction, resolution=100):
    return fi.sample_velocity_deficit_profiles(
        direction=direction,
        downstream_dists=downstream_dists,
        resolution=resolution,
        homogeneous_wind_speed=homogeneous_wind_speed,
    )

if __name__ == '__main__':
    D = 126.0
    hub_height = 90.0
    fi = FlorisInterface("inputs/gch.yaml")
    fi.reinitialize(layout_x=[0.0], layout_y=[0.0])

    downstream_dists = D * np.array([3, 5, 7])
    homogeneous_wind_speed = 8.0

    # Velocity deficit profiles can be obtained in either the cross-stream (y)
    # or the vertical direction (z). The default profile_range is [-2 * D, 2 * D] with
    # the default origin being (x, y, z) = (0.0, 0.0, reference_wind_height).
    # TODO: Explain difference between starting point and coordinates seen in fig
    profiles = get_profiles('y') + get_profiles('z')

    profiles_fig = VelocityProfilesFigure(
        downstream_dists_D=downstream_dists / D,
        layout=['y', 'z'],
    )
    profiles_fig.add_profiles(profiles, color='k')

    # Change velocity model to jensen, get the velocity deficit profiles,
    # and add them to the figure
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
        fontsize=14
    )

    # Show that profiles are independent of the wind direction for a single turbine.
    # This is because x is always in the streamwise direction.
    # Also show that the coordinates x / D, y / D and z / D returned by
    # sample_velocity_deficit_profiles are relative to the starting point mentioned above.
    fi = FlorisInterface("inputs/gch.yaml")
    fi.reinitialize(layout_x=[0.0], layout_y=[0.0])
    downstream_dists = D * np.array([3])
    profiles = get_profiles('y', resolution=400)
    profiles_fig = VelocityProfilesFigure(
        downstream_dists_D=downstream_dists / D,
        layout=['y']
    )
    profiles_fig.add_profiles(profiles, color='k')

    fi.reinitialize(wind_directions=[315.0], layout_x=[D], layout_y=[D])
    profiles = fi.sample_velocity_deficit_profiles(
        direction='y',
        downstream_dists=downstream_dists,
        resolution=21,
        homogeneous_wind_speed=homogeneous_wind_speed,
        x_inertial_start=D,
        y_inertial_start=D,
    )
    profiles_fig.add_profiles(
        profiles,
        linestyle='None',
        marker='o',
        color='b',
        markerfacecolor='None',
        markeredgewidth=1.3
    )
    profiles_fig.axs[0,0].legend(['WD = 270 deg', 'WD = 315 deg,\nand turbine\nmoved'])

    plt.show()
