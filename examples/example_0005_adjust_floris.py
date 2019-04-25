# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# See read the https://floris.readthedocs.io for documentation

import matplotlib.pyplot as plt
import floris.tools as wfct
import floris.tools.visualization as vis
import floris.tools.cut_plane as cp
from floris import Vec3
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

print('Running FLORIS...')
fi = wfct.floris_utilities.FlorisInterface("example_input.json")

# set turbine locations to 4 turbines in a row - demonstrate how to change coordinates
D = fi.floris.farm.turbines[0].rotor_diameter
layout_x = [0,7*D,0, 7*D]
layout_y = [0,0,5*D,5*D]
yaw_angles = [25.0,0,25.0,0]
fi.floris.farm.set_turbine_locations(layout_x, layout_y, calculate_wake=True)

# initial power output
power_initial = np.sum(fi.get_turbine_power())

# ================================================================================
print('Plotting the FLORIS flowfield...')
# ================================================================================

# Initialize the horizontal cut
hor_plane = wfct.cut_plane.HorPlane(
    fi.get_flow_field(),
    fi.floris.farm.turbines[0].hub_height
)

# Plot and show
fig, ax = plt.subplots()
wfct.visualization.visualize_cut_plane(hor_plane,ax=ax)

# ================================================================================
print('Changing wind direction and wind speed...')
# ================================================================================

ws = np.linspace(6,8,3)
wd = [45.0, 170.0, 270.]

# Plot and show
fig, ax = plt.subplots(3,3, figsize=(15,15))
power = np.zeros((len(ws),len(wd)))
for i,speed in enumerate(ws):
    for j,wdir in enumerate(wd):
        print('Calculating wake: wind direction = ', wdir, 'and wind speed = ', speed)

        fi.floris.farm.flow_field.reinitialize_flow_field(wind_speed=speed,
                                                                        wind_direction=wdir,

                                                                        # keep these the same
                                                                        wind_shear=fi.floris.farm.flow_field.wind_shear,
                                                                        wind_veer=fi.floris.farm.flow_field.wind_veer,
                                                                        turbulence_intensity=fi.floris.farm.flow_field.turbulence_intensity,
                                                                        air_density=fi.floris.farm.flow_field.air_density,
                                                                        wake=fi.floris.farm.flow_field.wake,
                                                                        turbine_map=fi.floris.farm.flow_field.turbine_map)
        # recalculate the wake
        fi.run_floris()

        # record powers
        power[i,j] = np.sum(fi.get_turbine_power())

        # ============================================
        # not necessary if you only want the powers
        # ============================================
        # Visualize the changes
        # Initialize the horizontal cut
        hor_plane = wfct.cut_plane.HorPlane(
            fi.get_flow_field(),
            fi.floris.farm.turbines[0].hub_height
        )
        im = wfct.visualization.visualize_cut_plane(hor_plane,ax=ax[i,j])
        strTitle = 'Wind Dir = ' + str(wdir) + 'deg' + ' Speed = ' + str(speed) + 'm/s'
        ax[i,j].set_title(strTitle)
        fig.colorbar(im,ax=ax[i,j],fraction=0.025, pad=0.04)

# ================================================================================
print('Set yaw angles...')
# ================================================================================

# assign yaw angles to turbines and calculate wake at 270
fi.floris.farm.set_yaw_angles(yaw_angles, calculate_wake=True)
power_yaw = np.sum(fi.get_turbine_power())

# ================================================================================
print('Plotting the FLORIS flowfield with yaw...')
# ================================================================================

# Initialize the horizontal cut
hor_plane = wfct.cut_plane.HorPlane(
    fi.get_flow_field(),
    fi.floris.farm.turbines[0].hub_height
)

# Plot and show
fig, ax = plt.subplots()
wfct.visualization.visualize_cut_plane(hor_plane,ax=ax)
ax.set_title('Flow with yawed front turbines')
plt.show()
