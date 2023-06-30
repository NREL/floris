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


"""
docstr
"""

fi = FlorisInterface("inputs/gch.yaml")
fi.reinitialize(layout_x=[0.0], layout_y=[0.0])

velocity_deficit_profiles = fi.sample_velocity_deficit_profiles(
    direction = 'y',
    resolution=10,
    homogeneous_wind_speed=7.0
)

for df in velocity_deficit_profiles:
    print(df)

velocity_deficit_profiles = fi.sample_velocity_deficit_profiles(
    direction = 'z',
    resolution=10,
    homogeneous_wind_speed=7.0
)

for df in velocity_deficit_profiles:
    print(df)

#horizontal_plane = fi.calculate_horizontal_plane(
#    x_resolution=200,
#    y_resolution=100,
#    height=90.0,
#)
#
#wakeviz.visualize_cut_plane(horizontal_plane, title="Horizontal plane")
#
#wakeviz.show_plots()
