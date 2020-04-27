# Copyright 2020 NREL
 
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0
 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
 
# See https://floris.readthedocs.io for documentation
 

# Compare 5 turbine results to SOWFA in 8 m/s, higher TI case

import matplotlib.pyplot as plt
import floris.tools as wfct
import numpy as np
import pandas as pd
import copy
import pickle

## Parameters
dist_downstream = 7 # Diameters, 

# Define some helper functions
def power_cross_sweep(fi_in,D,dist_downstream,yaw_angle=0):
    fi = copy.deepcopy(fi_in)

    sweep_locations = np.arange(-2,2.25,0.25)
    sweep_locations = np.arange(-1,1.25,0.25)
    power_out = np.zeros_like(sweep_locations)

    for y_idx, y_loc in enumerate(sweep_locations):

        fi.reinitialize_flow_field(
            layout_array=([0,dist_downstream*D], [0,y_loc*D])
        )
        fi.calculate_wake([yaw_angle,0])
        power_out[y_idx] = fi.get_turbine_power()[1]/1000.

    return sweep_locations, power_out

def power_cross_sweep_gain(fi_in,D,dist_downstream,yaw_angle=0):
    fi = copy.deepcopy(fi_in)

    sweep_locations = np.arange(-2,2.25,0.25)
    sweep_locations = np.arange(-1,1.25,0.25)
    power_out = np.zeros_like(sweep_locations)

    for y_idx, y_loc in enumerate(sweep_locations):

        fi.reinitialize_flow_field(
            layout_array=([0,dist_downstream*D], [0,y_loc*D])
        )
        fi.calculate_wake([0,0])
        base_power = fi.get_turbine_power()[1]/1000.
        fi.calculate_wake([yaw_angle,0])
        power_out[y_idx] = \
            100 * (fi.get_turbine_power()[1]/1000. - base_power) / base_power

    return sweep_locations, power_out

# Load the saved FLORIS interfaces
fi_dict = pickle.load( open( "floris_models.p", "rb" ) )

# Get HH and D

# Make a plot of comparisons
fig, axarr = plt.subplots(3,3,sharex=True, sharey=False,figsize=(14,9))

# Do the absolutes
for d_idx, dist_downstream in enumerate([10, 6, 3]):
    for y_idx, yaw in enumerate([0 , 20]):
        ax = axarr[d_idx, y_idx]
        ax.set_title('%d D downstream, yaw = %d' % (dist_downstream,yaw))

        for floris_label in fi_dict:
            (fi, floris_color, floris_marker) = fi_dict[floris_label]

            HH = fi.floris.farm.flow_field.turbine_map.turbines[0].hub_height
            D = fi.floris.farm.turbines[0].rotor_diameter
            
            sweep_locations, ps = power_cross_sweep(
                fi,
                D,
                dist_downstream,
                yaw
            )
            ax.plot(
                sweep_locations,
                ps,
                color=floris_color,
                marker=floris_marker,
                label=floris_label
            )
            ax.set_ylim([0,2000])

# Check the gains
for d_idx, dist_downstream in enumerate([10, 6, 3]):
    ax = axarr[d_idx, -1]
    ax.set_title('%d D downstream, Gain' % (dist_downstream))

    for floris_label in fi_dict:
        (fi, floris_color, floris_marker) = fi_dict[floris_label]
        sweep_locations, ps = power_cross_sweep_gain(
            fi,
            D,
            dist_downstream,
            yaw_angle=20
        )
        ax.plot(
                sweep_locations,
                ps,
                color=floris_color,
                marker=floris_marker,
                label=floris_label
            )
        ax.set_ylim([-100,100])

axarr[0,0].legend()
axarr[-1,0].set_xlabel('Lateral Offset (D)')
axarr[-1,1].set_xlabel('Lateral Offset (D)')
axarr[-1,2].set_xlabel('Lateral Offset (D)')
plt.show()
