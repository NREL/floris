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
from floris.tools.energy_ratio import plot_energy_ratio, plot_energy_ratio_ws 
import floris.tools.cut_plane as cp
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

# Parameters
n_sim = 1000 # number of simulations to use
wd_var_scale = 10
wd_bins = np.arange(265,365,1.)

# Set up a demonstration wind farm in a n L
# Initialize FLORIS model
fi = wfct.floris_utilities.FlorisInterface("example_input.json")

# set turbine locations to 4 turbines in a row - demonstrate how to change coordinates
D = fi.floris.farm.flow_field.turbine_map.turbines[0].rotor_diameter
layout_x = [0, 0, 7*D]
layout_y = [0, 5*D, 0, 0]
fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))

# Calculate wake
fi.calculate_wake()

# Show the farm
# Initialize the horizontal cut
hor_plane = wfct.cut_plane.HorPlane(
    fi.get_flow_data(),
    fi.floris.farm.turbines[0].hub_height
)

# Plot and show
fig, ax = plt.subplots()
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
# Annotate
ax.annotate('Control Turbine',(0,0),(0,200),arrowprops=dict(arrowstyle="->"))
ax.annotate('Test Turbine',(7*D,0),(7*D,200),arrowprops=dict(arrowstyle="->"))
ax.annotate('Reference Turbine',(0,5*D),(0,5*D-200),arrowprops=dict(arrowstyle="->"))

# Initialize the arrays for baseline
ref_pow_base = np.zeros(n_sim)
test_pow_base = np.zeros(n_sim)
ws_base = np.random.uniform(low=5,high=15.,size=n_sim)
wd_base = wd_var_scale * (np.random.rand(n_sim) - 0.5) + 270.

# Run the baseline
for i, (ws,wd) in enumerate(zip(ws_base,wd_base)):
    fi.reinitialize_flow_field(wind_speed=ws, wind_direction=wd)
    fi.calculate_wake()
    turb_powers = fi.get_turbine_power()
    ref_pow_base[i] = turb_powers[1] # The unaffacted second turbine
    test_pow_base[i] = turb_powers[2] # The downstream turbine


# Initialize the arrays for control
ref_pow_con = np.zeros(n_sim)
test_pow_con = np.zeros(n_sim)
ws_con = np.random.uniform(low=5,high=15.,size=n_sim)
wd_con = wd_var_scale * (np.random.rand(n_sim) - 0.5) + 270.

# Run the control
for i, (ws,wd) in enumerate(zip(ws_con,wd_con)):
    fi.reinitialize_flow_field(wind_speed=ws, wind_direction=wd)
    fi.calculate_wake(yaw_angles=[20,0,0]) # the control strategy is just to set the control turbine's yaw fixed at 20 degrees
    turb_powers = fi.get_turbine_power()
    ref_pow_con[i] = turb_powers[1] # The unaffacted second turbine
    test_pow_con[i] = turb_powers[2] # The downstream turbine


# Visualize the energy ratio wind direction

fig, axarr = plt.subplots(3,1,sharex=True,figsize=(10,10))

plot_energy_ratio(ref_pow_base,test_pow_base,ws_base,wd_base,
                    ref_pow_con,test_pow_con,ws_con,wd_con,
                    wd_bins, plot_simple=False, axarr=axarr, label_array=['Field Baseline', 'Field Controlled'],
                    label_pchange='Field Gain' )



for ax in axarr:
    ax.legend()


plt.show()
