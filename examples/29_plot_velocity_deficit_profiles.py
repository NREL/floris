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

def get_profiles(direction, profile_range=None, resolution=100):
    return fi.sample_velocity_deficit_profiles(
        direction=direction,
        downstream_dists=downstream_dists,
        profile_range=profile_range,
        resolution=resolution,
        homogeneous_wind_speed=homogeneous_wind_speed
    )

if __name__ == '__main__':
    D = 126.0
    hub_height = 90.0
    fi = FlorisInterface("inputs/gch.yaml")
    fi.reinitialize(layout_x=[0.0], layout_y=[0.0])

    downstream_dists = D * np.array([3, 5, 7])
    homogeneous_wind_speed = 8.0

    # Same range as y-profiles for an easier comparison
    profile_range_z = D * np.array([0, 4]) - hub_height
    profiles = get_profiles('y') + get_profiles('z', profile_range_z)

    profiles_fig = VelocityProfilesFigure(
        downstream_dists_D=downstream_dists / D,
        layout=['y', 'z'],
    )
    profiles_fig.add_profiles(profiles, color='k')

    # Change velocity model to jensen, get the velocity deficit profiles,
    # and plot them
    floris_dict = fi.floris.as_dict()
    floris_dict['wake']['model_strings']['velocity_model'] = 'jensen'
    fi = FlorisInterface(floris_dict)
    profiles_y = get_profiles('y', resolution=400)
    profiles_z = get_profiles('z', profile_range_z, resolution=400)
    profiles_fig.add_profiles(profiles_y + profiles_z, color='r')

    margin = 0.05
    profiles_fig.set_xlim([0.0 - margin, 0.6 + margin])
    profiles_fig.add_ref_lines_y_D([-0.5, 0.5])
    ref_lines_z_D = hub_height / D + np.array([-0.5, 0.5])
    profiles_fig.add_ref_lines_z_D(ref_lines_z_D)

    profiles_fig.axs[0,0].legend(['gauss', 'jensen'], fontsize=11)
    profiles_fig.fig.suptitle(
        'Velocity decifit profiles from different velocity models',
        fontsize=14
    )

    # Profiles are independent of wind direction for a single turbine.
    # This is because x is always in the streamwise direction
    fi = FlorisInterface("inputs/gch.yaml")
    fi.reinitialize(layout_x=[0.0], layout_y=[0.0])
    downstream_dists = D * np.array([3])
    profiles = get_profiles('y', resolution=400)
    profiles_fig = VelocityProfilesFigure(
        downstream_dists_D=downstream_dists / D,
        layout=['y']
    )
    profiles_fig.add_profiles(profiles, color='k')

    fi.reinitialize(wind_directions=[315.0])
    profiles = get_profiles('y', resolution=21)
    profiles_fig.add_profiles(
        profiles,
        linestyle='None',
        marker='o',
        color='b',
        markerfacecolor='None',
        markeredgewidth=1.3
    )
    profiles_fig.axs[0,0].legend(['WD = 270 deg', 'WD = 315 deg'])

    plt.show()
