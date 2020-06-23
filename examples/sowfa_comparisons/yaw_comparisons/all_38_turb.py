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

# Parameters
num_turbines = 38
sowfa_U0 = 8.0
sowfa_TI = 0.1 # High = 0.1, low = 0.06
layout_x = (1000.0, 1756.0, 2512.0, 3268.0, 4024.0)
layout_y = (1000.0, 1000.0, 1000.0, 1000.0, 1000.0)
# yaw_cases_to_select = [
#     np.array([0.,0.,0.,0.,0.]),
#     np.array([25.,0.,0.,0.,0.]),
#     np.array([25.,25.,0.,0.,0.])
# ]

## Grab certain hi-TI five simulations from saved SOWFA data set
df_sowfa = pd.read_pickle('../sowfa_data_set/sowfa_data_set.p')

# Limit number of turbines 
df_sowfa = df_sowfa[df_sowfa.num_turbines == num_turbines]

# Limit to wind speed
df_sowfa = df_sowfa[df_sowfa.sowfa_U0 == sowfa_U0]

# Limit to turbulence
df_sowfa = df_sowfa[df_sowfa.sowfa_TI == sowfa_TI]

# Sort by total sowfa power
df_sowfa['total_sowfa_power'] = df_sowfa.power.apply(np.sum)
df_sowfa = df_sowfa.sort_values('total_sowfa_power')

# Load the saved FLORIS interfaces
fi_dict = pickle.load( open( "../floris_models.p", "rb" ) )

# Resimulate the SOWFA cases
for floris_label in fi_dict:
    (fi, floris_color, floris_marker) = fi_dict[floris_label]

    df_sowfa[floris_label] = 0
    df_sowfa[floris_label] = df_sowfa[floris_label].astype(object)
    for i, row in df_sowfa.iterrows():

        # Match the layout, wind_speed and TI
        fi.reinitialize_flow_field(
            layout_array=[row.layout_x,row.layout_y],
            wind_speed=[row.floris_U0],
            turbulence_intensity=[row.floris_TI]
        )

        # Calculate wake with certain yaw
        fi.calculate_wake(yaw_angles = row.yaw)

        # Save the result
        df_sowfa.at[i,floris_label] = np.round(
            np.array(fi.get_turbine_power())/1000.,2
        )

# Compare the turbine powers by case
num_cases = df_sowfa.shape[0]
num_col = np.min([4,num_cases])
num_row = int(np.ceil(num_cases/num_col))
# fig, axarr = plt.subplots(
#     num_row,
#     num_col,
#     figsize=(10,5),
#     sharex=True,
#     sharey=True
# )
# axarr = axarr.flatten()
# for idx, (i, row) in enumerate(df_sowfa.iterrows()):
#     ax = axarr[idx]

#     # Plot the sowfa result
#     ax.plot(row.power,'ks-',label='SOWFA')

#     # Plot the FLORIS results
#     for floris_label in fi_dict:
#         (fi, floris_color, floris_marker) = fi_dict[floris_label]
#         ax.plot(
#             row[floris_label],
#             color=floris_color,
#             marker=floris_marker,
#             label=floris_label
#         )

#     ax.set_title(row.yaw)
#     ax.grid(True)
#     ax.set_xlabel('Turbine')
#     ax.set_ylabel('Power (kW)')
# axarr[0].legend()

# Compare the change in total power
fig, ax = plt.subplots(figsize=(7,4))
case_names = df_sowfa.yaw.apply(lambda x: '/'.join(x.astype(int).astype(str)))
sowfa_total = df_sowfa.power.apply(np.sum)
ax.plot(sowfa_total,case_names,'ks-',label='SOWFA')
# Plot the FLORIS results
for floris_label in fi_dict:
    (fi, floris_color, floris_marker) = fi_dict[floris_label]
    total = df_sowfa[floris_label].apply(np.sum)
    ax.plot(
        total,
        case_names,
        color=floris_color,
        marker=floris_marker,
        label=floris_label
    )

ax.grid(True)
ax.set_xlabel('Total Power (kW)')
ax.set_ylabel('Case')
ax.legend()
fig.tight_layout()

# Compare the change in normalized power
df_baseline = df_sowfa[df_sowfa.yaw.apply(lambda x: np.max(np.abs(x)))==0.0]
fig, ax = plt.subplots(figsize=(7,4))
case_names = df_sowfa.yaw.apply(lambda x: '/'.join(x.astype(int).astype(str)))
sowfa_total = df_sowfa.power.apply(np.sum)

# Normalized vetsion
# Assign a unique case to TI/layout pairs
df_sowfa['sim_case'] = df_sowfa.apply(lambda x: 'case_' + str(int(np.floor(x.d_spacing))+1),axis=1)

# Do the norms
df_norm = df_sowfa.copy(deep=True)
# Switch to totals
df_norm['power'] = df_norm['power'].apply(np.sum)
df_norm['default'] = df_norm['default'].apply(np.sum)
df_norm['merge'] = df_norm['merge'].apply(np.sum)
df_norm['legacy'] = df_norm['legacy'].apply(np.sum)
# df_norm.head()
df_norm['power']  = df_norm['power'] / df_norm.groupby('sim_case').transform(np.min)['power']
df_norm['default']  = df_norm['default'] / df_norm.groupby('sim_case').transform(np.min)['default']
df_norm['merge']  = df_norm['merge'] / df_norm.groupby('sim_case').transform(np.min)['merge']
df_norm['legacy']  = df_norm['legacy'] / df_norm.groupby('sim_case').transform(np.min)['legacy']

# Compare the change in total power
fig, axarr = plt.subplots(2,1,figsize=(7,4),sharex=True)

for idx in range(2):
    ax = axarr[idx]
    
    df_sub = df_norm[df_norm.sim_case == 'case_%d' % (idx+1)]
    
    case_names = df_sub.yaw.apply(lambda x: '/'.join(x.astype(int).astype(str)))
    sowfa_total = df_sub.power.values
    ax.plot(sowfa_total,case_names,'ks-',label='SOWFA')
    # Plot the FLORIS results
    for floris_label in fi_dict:
        (fi, floris_color, floris_marker) = fi_dict[floris_label]
        total = df_sub[floris_label]#.apply(np.sum)
        ax.plot(
            total,
            case_names,
            color=floris_color,
            marker=floris_marker,
            label=floris_label
        )

    ax.grid(True)
    ax.set_xlabel('Total Power (kW)')
    ax.set_ylabel('Case')
    ax.legend()
    ax.set_title('case_%d' % (idx+1))
    fig.tight_layout()

plt.show()