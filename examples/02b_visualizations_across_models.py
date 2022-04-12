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
from floris.tools.visualization import plot_rotor_values

"""
Show how the new calculate_horizontal_plane_with_turbines can be used to visualize wake models for which
calculating a visualization grid of points is not yet possible
"""

# Initialize seperate FLORIS interface for each model
fi_jensen = FlorisInterface("inputs/jensen.yaml")
fi_gch = FlorisInterface("inputs/gch.yaml")
fi_cc = FlorisInterface("inputs/cc.yaml")
fi_turbopark = FlorisInterface("inputs/turbopark.yaml")

# For each model type that currently allows it, grab the horizontal plane
horizontal_plane_jensen = fi_jensen.calculate_horizontal_plane(x_resolution=200, y_resolution=100, height=90.0)
horizontal_plane_gch = fi_gch.calculate_horizontal_plane(x_resolution=200, y_resolution=100, height=90.0)

# Next for each model, calculate the horizontal u velocity using the turbine method
# Because this method is much slower, recommend a much coarser grid
t_res = 20
horizontal_plane_jensen_turbine = fi_jensen.calculate_horizontal_plane_with_turbines(x_resolution=t_res, y_resolution=t_res)
horizontal_plane_gch_turbine = fi_gch.calculate_horizontal_plane_with_turbines(x_resolution=t_res, y_resolution=t_res)
horizontal_plane_cc_turbine = fi_cc.calculate_horizontal_plane_with_turbines(x_resolution=t_res, y_resolution=t_res)
horizontal_plane_turbopark_turbine = fi_turbopark.calculate_horizontal_plane_with_turbines(x_resolution=t_res, y_resolution=t_res)

# Make a plot comparing the visualizations computed via the different methods
fig, axarr = plt.subplots(4,2, sharex=True, sharey=True)
visualize_cut_plane(horizontal_plane_jensen, ax=axarr[0,0], title="Jensen - Flowfield")
visualize_cut_plane(horizontal_plane_jensen_turbine, ax=axarr[0,1], title="Jensen - Turbine")
visualize_cut_plane(horizontal_plane_gch, ax=axarr[1,0], title="GCH - Flowfield")
visualize_cut_plane(horizontal_plane_gch_turbine, ax=axarr[1,1], title="GCH - Turbine")
visualize_cut_plane(horizontal_plane_cc_turbine, ax=axarr[2,1], title="CC - Turbine")
visualize_cut_plane(horizontal_plane_turbopark_turbine, ax=axarr[3,1], title="Turbopark - Turbine")

plt.show()
