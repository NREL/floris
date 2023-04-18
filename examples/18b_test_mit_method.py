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


import os

import matplotlib.pyplot as plt
import numpy as np

from floris.tools import FlorisInterface


"""
For each turbine in the turbine library, make a small figure showing that its power
curve and power loss to yaw are reasonable and  reasonably smooth
"""
ws_array = np.arange(0.1,30,0.2)
yaw_angles = np.linspace(-30,30,60)
wind_speed_to_test_yaw = 8

# Grab the gch model
fi = FlorisInterface("inputs/gch.yaml")

# Make one turbine sim
fi.reinitialize(layout_x=[0], layout_y=[0])

# Apply wind speeds
fi.reinitialize(wind_speeds=ws_array)

# Get a list of available turbine models
turbines = ['nrel_5MW.yaml','nrel_5MW_mit.yaml' ]


# Create a figure
fig, axarr = plt.subplots(1,3,figsize=(10,5))


# For each turbine model available plot the basic info
for t in turbines:

    # Set t as the turbine
    fi.reinitialize(turbine_type=[t])

    # Since we are changing the turbine type, make a matching change to the reference wind height
    fi.assign_hub_height_to_ref_height()




    # POWER CURVE
    ax = axarr[0]
    fi.reinitialize(wind_speeds=ws_array)
    fi.calculate_wake()
    turbine_powers = fi.get_turbine_powers().flatten() / 1e3
    ax.plot(ws_array,turbine_powers,label=t, lw=2)
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('Wind Speed (m/s)')
    ax.set_ylabel('Power (kW)')

    # Power loss to yaw, try a range of yaw angles
    ax = axarr[1]

    fi.reinitialize(wind_speeds=[8.0])
    yaw_result = []
    for yaw in yaw_angles:
        fi.calculate_wake(yaw_angles=np.array([[[yaw]]]))
        turbine_powers = fi.get_turbine_powers().flatten() / 1e3
        yaw_result.append(turbine_powers[0])

    ax.plot(yaw_angles,yaw_result)
    # ax.plot(yaw_angles,yaw_result,label='Air Density = %.3f' % density)
    ax.grid(True)
    # ax.legend()
    ax.set_xlabel('Yaw Error (deg)')
    ax.set_ylabel('Power (kW)')
    ax.set_title('Wind Speed = %.1f' % 8 )


    # Power loss to yaw, try a range of yaw angles
    ax = axarr[2]

    fi.reinitialize(wind_speeds=[12.0])
    yaw_result = []
    for yaw in yaw_angles:
        fi.calculate_wake(yaw_angles=np.array([[[yaw]]]))
        turbine_powers = fi.get_turbine_powers().flatten() / 1e3
        yaw_result.append(turbine_powers[0])

    ax.plot(yaw_angles,yaw_result)
    # ax.plot(yaw_angles,yaw_result,label='Air Density = %.3f' % density)
    ax.grid(True)
    # ax.legend()
    ax.set_xlabel('Yaw Error (deg)')
    ax.set_ylabel('Power (kW)')
    ax.set_title('Wind Speed = %.1f' % 12 )
    # Give a suptitle
    fig.suptitle(t)

plt.show()
