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
D = 126.0
fi = FlorisInterface("inputs/gch.yaml")
fi.reinitialize(layout_x=[0.0], layout_y=[0.0])

downstream_dists = D * np.array([3, 5, 7])
homogeneous_wind_speed = 8.0

velocity_deficit_profiles_y = fi.sample_velocity_deficit_profiles(
    direction='y',
    downstream_dists=downstream_dists,
    homogeneous_wind_speed=homogeneous_wind_speed
)
# Same range as y-profiles for an easier comparison
profile_range_z = D * np.array([0, 4]) - 90.0
velocity_deficit_profiles_z = fi.sample_velocity_deficit_profiles(
    direction='z',
    downstream_dists=downstream_dists,
    profile_range=profile_range_z,
    homogeneous_wind_speed=homogeneous_wind_speed
)

profiles_fig = VelocityProfilesFigure(
    downstream_dists_D=downstream_dists / D,
    layout=['y', 'z']
)
profiles_fig.add_profiles(
    velocity_deficit_profiles_y + velocity_deficit_profiles_z,
    color='k'
)
margin = 0.05
profiles_fig.set_xlim([0.0 - margin, 0.6 + margin])
profiles_fig.add_ref_lines_y_D([-0.5, 0.5])
ref_lines_z_D = 90.0 / D + np.array([-0.5, 0.5])
profiles_fig.add_ref_lines_z_D(ref_lines_z_D)

plt.show()
