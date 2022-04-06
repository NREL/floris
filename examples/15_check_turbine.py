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
import os

from floris.tools import FlorisInterface
from floris.tools import visualize_cut_plane #, plot_turbines_with_fi

"""
For each turbine in the turbine library, make a small figure showing that its power curve and power loss to yaw are reasonable and 
reasonably smooth
"""
ws_array = np.arange(0.1,30,0.2)
yaw_angles = np.linspace(-30,30,60)
wind_speed_to_test_yaw = 11

# Grab the gch model
fi = FlorisInterface("inputs/gch.yaml")

# Make one turbine sim
fi.reinitialize(layout=[[0],[0]])

# Apply wind speeds
fi.reinitialize(wind_speeds=ws_array)

# Get a list of available turbine models
turbines = os.listdir('../floris/turbine_library')
turbines = [t.strip('.yaml') for t in turbines]

# Declare a set of figures for comparing cp and ct across models
fig_cp_ct, axarr_cp_ct = plt.subplots(2,1,sharex=True,figsize=(10,10))

# For each turbine model available plot the basic info
for t in turbines:

    # Set t as the turbine
    fi.reinitialize(turbine_type=[t])

    # Since we are changing the turbine type, make a matching change to the reference wind height
    fi.assign_hub_height_to_ref_height()

    # Plot cp and ct onto the fig_cp_ct plot
    axarr_cp_ct[0].plot(fi.floris.farm.turbine_map[0].power_thrust_table.wind_speed, fi.floris.farm.turbine_map[0].power_thrust_table.power,label=t )
    axarr_cp_ct[0].grid(True)
    axarr_cp_ct[0].legend()
    axarr_cp_ct[0].set_ylabel('Cp')
    axarr_cp_ct[1].plot(fi.floris.farm.turbine_map[0].power_thrust_table.wind_speed, fi.floris.farm.turbine_map[0].power_thrust_table.thrust,label=t )
    axarr_cp_ct[1].grid(True)
    axarr_cp_ct[1].legend()
    axarr_cp_ct[1].set_ylabel('Ct')
    axarr_cp_ct[1].set_xlabel('Wind Speed (m/s)')

    # Create a figure
    fig, axarr = plt.subplots(1,2,figsize=(10,5))

    # Try a few density
    for density in [1.15,1.225,1.3]:
        
        fi.reinitialize(air_density=density)

        # POWER CURVE
        ax = axarr[0]
        fi.reinitialize(wind_speeds=ws_array)
        fi.calculate_wake()
        turbine_powers = fi.get_turbine_powers().flatten() / 1e3
        if density == 1.225:
            ax.plot(ws_array,turbine_powers,label='Air Density = %.3f' % density, lw=2, color='k')
        else:
            ax.plot(ws_array,turbine_powers,label='Air Density = %.3f' % density, lw=1)
        ax.grid(True)
        ax.legend()
        ax.set_xlabel('Wind Speed (m/s)')
        ax.set_ylabel('Power (kW)')

        # Power loss to yaw, try a range of yaw angles
        ax = axarr[1]

        fi.reinitialize(wind_speeds=[wind_speed_to_test_yaw])
        yaw_result = []
        for yaw in yaw_angles:
            fi.calculate_wake(yaw_angles=np.array([[[yaw]]]))
            turbine_powers = fi.get_turbine_powers().flatten() / 1e3
            yaw_result.append(turbine_powers[0])
        if density == 1.225:
            ax.plot(yaw_angles,yaw_result,label='Air Density = %.3f' % density, lw=2, color='k')
        else:
            ax.plot(yaw_angles,yaw_result,label='Air Density = %.3f' % density, lw=1)
        # ax.plot(yaw_angles,yaw_result,label='Air Density = %.3f' % density)
        ax.grid(True)
        ax.legend()
        ax.set_xlabel('Yaw Error (deg)')
        ax.set_ylabel('Power (kW)')
        ax.set_title('Wind Speed = %.1f' % wind_speed_to_test_yaw )

    # Give a suptitle
    fig.suptitle(t)



plt.show()


