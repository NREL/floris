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
import pandas as pd
import yaml
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from floris.tools import FlorisInterface
from floris.tools.visualization import (
    plot_rotor_values,
    VelocityProfilesFigure,
    visualize_cut_plane,
)


"""
The first part of this example shows a characteristic wake of a vertical-axis wind turbine (VAWT).
It is based on case 3 in :cite:``abkar2019theoretical. The super-Gaussian velocity model with
default coefficients is used, which allows the wake to have different characteristics in the
cross-stream (y) and vertical direction (z). The initial wake shape is closely related to
the turbine cross section, which is:
    rotor diameter * length of the vertical turbine blades.
When plotting the velocity deficit profiles, we use the following definition:
    velocity_deficit = (homogeneous_wind_speed - u) / homogeneous_wind_speed
        , where u is the wake velocity obtained when the incoming wind speed is the
        same at all heights and equal to `homogeneous_wind_speed`.
See example 29 for more details about how to sample and plot these kinds of profiles.

The second part of the example shows how turbine velocities and turbine powers are obtained in a
layout with two VAWTs.

References:
    .. bibliography:: /references.bib
        :style: unsrt
        :filter: docname in docnames
"""

D = 26.0 # Rotor diameter
vawt_blade_length = 48.0 # Length of vertical turbine blades
hub_height = 40.0
# Streamwise location of each velocity deficit profile
downstream_dists = D * np.array([1, 4, 7, 10])
homogeneous_wind_speed = 7.0

fi = FlorisInterface('inputs/super_gaussian_vawt.yaml')

# Velocity field in a horizontal plane
horizontal_plane = fi.calculate_horizontal_plane(
    x_resolution=200,
    y_resolution=100,
    height=hub_height,
)
horizontal_plane.df['x1'] /= D
horizontal_plane.df['x2'] /= D
fig, ax = plt.subplots()
visualize_cut_plane(
    horizontal_plane,
    ax,
    title='Streamwise velocity [m/s] in a horizontal plane at hub height',
)
# The figure's XAxis and YAxis are in inertial coordinates which are not affected by
# the wind direction
ax.set_xlabel('$x_{inertial} / D$', fontsize=12)
ax.set_ylabel('$y_{inertial} / D$', fontsize=12)
ax.tick_params('both', labelsize=12)
fig.set_size_inches(8, 4)

# Velocity field in a vertical plane
y_plane = fi.calculate_y_plane(
    x_resolution=200,
    z_resolution=100,
    crossstream_dist=0.0,
    z_bounds=np.array([-2, 2]) * D + hub_height,
)
y_plane.df['x1'] /= D
y_plane.df['x2'] /= D
fig, ax = plt.subplots()
visualize_cut_plane(
    y_plane,
    ax,
    title='Streamwise velocity [m/s] in a xz-plane going through\n'
          'the center of the turbine',
)
# The figure's XAxis and YAxis are in inertial coordinates which are not affected by
# the wind direction
ax.set_xlabel('$x_{inertial} / D$', fontsize=12)
ax.set_ylabel('$z_{inertial} / D$', fontsize=12)
ax.tick_params('both', labelsize=12)
fig.set_size_inches(8, 4)

# Velocity deficit profiles.
# The coordinates x/D, y/D and z/D returned by sample_velocity_deficit_profiles
# (and seen in the figure) are relative to the sampling starting point.
# Here, the following default starting point, in inertial coordinates, is used:
#     (0.0, 0.0, fi.floris.flow_field.reference_wind_height)
profiles_y = fi.sample_velocity_deficit_profiles(
    direction='y',
    downstream_dists=downstream_dists,
    homogeneous_wind_speed=homogeneous_wind_speed,
)
profiles_z = fi.sample_velocity_deficit_profiles(
    direction='z',
    downstream_dists=downstream_dists,
    homogeneous_wind_speed=homogeneous_wind_speed,
)

profiles_fig = VelocityProfilesFigure(
    downstream_dists_D=downstream_dists / D,
    layout=['y', 'z'],
    ax_width=1.8,
)
profiles_fig.add_profiles(profiles_y + profiles_z, color='k')

# Add dashed reference lines that show the extent of the turbine.
# Each line is defined by a coordinate that is normalized by `D`.
profiles_fig.add_ref_lines_y([-0.5, 0.5])
H_half = vawt_blade_length / 2
ref_lines_z_D = [-H_half / D, H_half / D]
profiles_fig.add_ref_lines_z(ref_lines_z_D)

profiles_fig.set_xlim([0.0 - 0.05, 0.6 + 0.05])
for ax in profiles_fig.axs[:,0]:
    ax.set_ylim([-2 - 0.05, 2 + 0.05])

for ax in profiles_fig.axs[-1]:
    ax.xaxis.set_major_locator(MultipleLocator(0.4))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))

profiles_fig.fig.suptitle('Velocity deficit profiles', fontsize=14)

# Switch to a two-turbine layout. Then, calculate the velocities at each turbine in a
# yz-grid centered on the hub of the turbine. The grids are based on the VAWTs' rectangular
# cross-section and have a resolution of 3x5 to make them reasonably equidistant.
solver_settings = {
    'type': 'turbine_grid',
    'turbine_grid_points': [3, 5]
}
fi.reinitialize(layout_x=[0.0, 78.0], layout_y=[0.0, 0.0], solver_settings=solver_settings)
fi.calculate_wake()

# Plot the velocities in each grid
fig, axes, _, _ = plot_rotor_values(
    fi.floris.flow_field.u,
    wd_index=0,
    ws_index=0,
    n_rows=1,
    n_cols=2,
    return_fig_objects=True,
)
fig.suptitle(
    'Streamwise velocity [m/s] in yz-grids centered on\n'
    'the hub of each turbine, in a two-turbine layout'
)

# Calculate the power generated by each turbine. The average velocity in each yz-grid
# is used in the calculation.
turbine_powers = fi.get_turbine_powers() / 1000
print(f'Turbine powers for a two-turbine layout [kW] =\n{turbine_powers}')

plt.show()
