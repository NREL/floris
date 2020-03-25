# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import matplotlib.pyplot as plt
import floris.tools as wfct
import numpy as np
import pandas as pd
import copy
import seaborn as sns

"""
Example Declare Gauss Legacy Two Ways
Declare Gauss Legacy through the specific JSON, or by modifying the default JSON
and confirm that they are identical
"""

# Define some helper functions
def power_cross_sweep(fi_in,D,dist_downstream,yaw_angle=0):
    fi = copy.deepcopy(fi_in)

    sweep_locations = np.arange(-2,2.25,0.25)
    sweep_locations = np.arange(-1,1.25,0.25)
    power_out = np.zeros_like(sweep_locations)

    for y_idx, y_loc in enumerate(sweep_locations):

        fi.reinitialize_flow_field(layout_array=([0,dist_downstream*D],[0,y_loc*D]))
        fi.calculate_wake([yaw_angle,0])
        power_out[y_idx] = fi.get_turbine_power()[1]/1000.

    return sweep_locations, power_out



# Set up the models ....
# ======================
fi_dict = dict()
color_dict = dict()
label_dict = dict()

# Declare from JSON
print('JSON========================')
fi_json = wfct.floris_interface.FlorisInterface("../example_input_gauss_legacy.json")
fi_dict['fi_json'] = fi_json
color_dict['fi_json'] = 'r^-'
label_dict['fi_json'] = 'from JSON'
print(fi_json.get_model_parameters())

# quit()

# Gauss_Legacy Class with GCH disabled and deflection multiplier = 1.2
print('SOFTWARE========================')
fi_software = wfct.floris_interface.FlorisInterface("../example_input.json")
fi_software.floris.farm.set_wake_model('gauss_legacy')
fi_software.set_gch(False) # Disable GCH
fi_software.floris.farm.wake._deflection_model.dm = 1.2 # Deflection multiplier to 1.2
fi_dict['fi_software'] = fi_software
color_dict['fi_software'] = 'bo--'
label_dict['fi_software'] = 'from softoware'
print(fi_software.get_model_parameters())



# Get HH and D
HH = fi_json.floris.farm.flow_field.turbine_map.turbines[0].hub_height
D = fi_json.floris.farm.turbines[0].rotor_diameter

# Make a plot of comparisons
fig, axarr = plt.subplots(3,2,sharex=True, sharey=False,figsize=(14,9))

# Do the absolutes
for d_idx, dist_downstream in enumerate([10, 6, 3]):
    for y_idx, yaw in enumerate([0 , 20]):
        ax = axarr[d_idx, y_idx]
        ax.set_title('%d D downstream, yaw = %d' % (dist_downstream,yaw))

        for fi_key in fi_dict.keys():
            sweep_locations, ps = power_cross_sweep(fi_dict[fi_key],D,dist_downstream,yaw)
            ax.plot(sweep_locations,ps,color_dict[fi_key] ,label=label_dict[fi_key])
            ax.set_ylim([0,2000])


axarr[0,0].legend()
axarr[-1,0].set_xlabel('Lateral Offset (D)')
axarr[-1,1].set_xlabel('Lateral Offset (D)')
# axarr[-1,2].set_xlabel('Lateral Offset (D)')
plt.show()

