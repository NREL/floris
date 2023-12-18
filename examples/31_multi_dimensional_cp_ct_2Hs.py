# Copyright 2023 NREL

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

from floris.tools import FlorisInterface


"""
This example follows after example 30 but shows the effect of changing the Hs setting.

NOTE: The multi-dimensional Cp/Ct data used in this example is fictional for the purposes of
facilitating this example. The Cp/Ct values for the different wave conditions are scaled
values of the original Cp/Ct data for the IEA 15MW turbine.
"""

# Initialize FLORIS with the given input file via FlorisInterface.
fi = FlorisInterface("inputs/gch_multi_dim_cp_ct.yaml")

# Make a second FLORIS interface with a different setting for Hs.
# Note the multi-cp-ct file (iea_15MW_multi_dim_Tp_Hs.csv)
# for the turbine model iea_15MW_floating_multi_dim_cp_ct.yaml
# Defines Hs at 1 and 5.
# The value in gch_multi_dim_cp_ct.yaml is 3.01 which will map
# to 5 as the nearer value, so we set the other case to 1
# for contrast.
fi_dict_mod = fi.floris.as_dict()
fi_dict_mod['flow_field']['multidim_conditions']['Hs'] = 1.0
fi_hs_1 = FlorisInterface(fi_dict_mod)

# Set both cases to 3 turbine layout
fi.reinitialize(layout_x=[0., 500., 1000.], layout_y=[0., 0., 0.])
fi_hs_1.reinitialize(layout_x=[0., 500., 1000.], layout_y=[0., 0., 0.])

# Use a sweep of wind speeds
wind_speeds = np.arange(5, 20, 1.0)
wind_directions = 270.0 * np.ones_like(wind_speeds)
fi.reinitialize(wind_directions=wind_directions, wind_speeds=wind_speeds)
fi_hs_1.reinitialize(wind_directions=wind_directions, wind_speeds=wind_speeds)

# Calculate wakes with baseline yaw
fi.calculate_wake()
fi_hs_1.calculate_wake()

# Collect the turbine powers in kW
turbine_powers = fi.get_turbine_powers_multidim()/1000.
turbine_powers_hs_1 = fi_hs_1.get_turbine_powers_multidim()/1000.

# Plot the power in each case and the difference in power
fig, axarr = plt.subplots(1,3,sharex=True,figsize=(12,4))

for t_idx in range(3):
    ax = axarr[t_idx]
    ax.plot(wind_speeds, turbine_powers[:,t_idx], color='k', label='Hs=3.1 (5)')
    ax.plot(wind_speeds, turbine_powers_hs_1[:,t_idx], color='r', label='Hs=1.0')
    ax.grid(True)
    ax.set_xlabel('Wind Speed (m/s)')
    ax.set_title(f'Turbine {t_idx}')

axarr[0].set_ylabel('Power (kW)')
axarr[0].legend()
fig.suptitle('Power of each turbine')

plt.show()
