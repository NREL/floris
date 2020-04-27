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
num_turbines = 3
sowfa_U0 = 8.0
sowfa_TI = 0.1 # High = 0.1, low = 0.06

# Can chose between two x layouts
# layout_x = (1000.0, 1756.0, 2512.0) 
layout_x = (1000.0, 1882.0, 2764.0)
layout_y = (1000.0, 1000.0, 1000.0)

## Grab certain hi-TI five simulations from saved SOWFA data set
df_sowfa = pd.read_pickle('../sowfa_data_set/sowfa_data_set.p')

# Limit number of turbines 
df_sowfa = df_sowfa[df_sowfa.num_turbines == num_turbines]

# Limit to wind speed
df_sowfa = df_sowfa[df_sowfa.sowfa_U0 == sowfa_U0]

# Limit to turbulence
df_sowfa = df_sowfa[df_sowfa.sowfa_TI == sowfa_TI]

# Limit to particular layout
df_sowfa = df_sowfa[df_sowfa.layout_x == layout_x]
df_sowfa = df_sowfa[df_sowfa.layout_y == layout_y]

# Limit to positive yaw angles
df_sowfa = df_sowfa[df_sowfa.yaw.apply(np.min) >= 0.0]

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
fig, axarr = plt.subplots(
    num_row,
    num_col,
    figsize=(10,5),
    sharex=True,
    sharey=True
)
axarr = axarr.flatten()
for idx, (i, row) in enumerate(df_sowfa.iterrows()):
    ax = axarr[idx]

    # Plot the sowfa result
    ax.plot(row.power,'ks-',label='SOWFA')

    # Plot the FLORIS results
    for floris_label in fi_dict:
        (fi, floris_color, floris_marker) = fi_dict[floris_label]
        ax.plot(
            row[floris_label],
            color=floris_color,
            marker=floris_marker,
            label=floris_label
        )

    ax.set_title(row.yaw)
    ax.grid(True)
    ax.set_xlabel('Turbine')
    ax.set_ylabel('Power (kW)')
axarr[0].legend()

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

# Normalize
base_total = df_baseline.power.apply(np.sum).values[0]
sowfa_total = sowfa_total / base_total
ax.plot(sowfa_total,case_names,'ks-',label='SOWFA')

# Plot the FLORIS results
for floris_label in fi_dict:
    (fi, floris_color, floris_marker) = fi_dict[floris_label]
    total = df_sowfa[floris_label].apply(np.sum)

    # Normalize
    base_total = df_baseline[floris_label].apply(np.sum).values[0]
    total = total / base_total
    ax.plot(
        total,
        case_names,
        color=floris_color,
        marker=floris_marker,
        label=floris_label
    )

ax.grid(True)
ax.set_xlabel('Normalized Power')
ax.set_ylabel('Case')
ax.legend()
fig.tight_layout()
plt.show()

# Write out SOWFA results

sowfa_results = np.array([
    [1940,843.9,856.9,893.1,926.2,0,0,0,0,0],
    [1575.3,1247.3,1008.4,955.4,887.1,25,0,0,0,0],
    [1576.4,1065,1147.5,1185.2,1198.5,25,20,15,10,0],
    [1577,986.9,1338.7,1089.4,999.8,25,25,0,0,0],
    [1941.1,918.6,945.3,948,968.2,0,0,0,0,0]
])
df_sowfa = pd.DataFrame(
    sowfa_results, 
    columns = ['p0','p1','p2','p3','p4','y0','y1','y2','y3','y4']
)

# ## SET UP FLORIS AND MATCH TO BASE CASE
# wind_speed = 8.38
# TI = 0.09

# # Initialize the FLORIS interface fi, use default model
# fi = wfct.floris_interface.FlorisInterface("../../example_input.json")
# fi.reinitialize_flow_field(wind_speed=[wind_speed],turbulence_intensity=[TI],layout_array=(layout_x, layout_y))

# # Setup alternative with gch off
# fi_b = copy.deepcopy(fi)
# fi_b.set_gch(False)

# # Setup the previous defaul
# fi_gl = wfct.floris_interface.FlorisInterface("../../other_jsons/input_legacy.json")
# fi_gl.reinitialize_flow_field(wind_speed=[wind_speed],turbulence_intensity=[TI],layout_array=(layout_x, layout_y))

# # Compare yaw combinations
# yaw_combinations = [
#     (0,0,0,0,0), (25,0,0,0,0), (25,25,0,0,0)
# ]
# yaw_names = ['%d/%d/%d/%d/%d' % yc for yc in yaw_combinations]

# # Plot individual turbine powers
# fig, axarr = plt.subplots(1,3,sharex=True,sharey=True,figsize=(12,5))

# total_sowfa = []
# total_gch_on = []
# total_gch_off = []
# total_legacy = []

# for y_idx, yc in enumerate(yaw_combinations):

#     # Collect SOWFA DATA
#     s_data = df_sowfa[(df_sowfa.y0==yc[0]) & (df_sowfa.y1==yc[1]) & (df_sowfa.y2==yc[2]) & (df_sowfa.y2==yc[3]) & (df_sowfa.y2==yc[4])]
#     s_data = [s_data.p0.values[0], s_data.p1.values[0],s_data.p2.values[0],s_data.p3.values[0],s_data.p4.values[0]]
#     total_sowfa.append(np.sum(s_data))

#     # Collect GCH ON data
#     fi.calculate_wake(yaw_angles=yc)
#     g_data = np.array(fi.get_turbine_power())/ 1000. 
#     total_gch_on.append(np.sum(g_data))

#     # Collect GCH OFF data
#     fi_b.calculate_wake(yaw_angles=yc)
#     b_data = np.array(fi_b.get_turbine_power())/ 1000. 
#     total_gch_off.append(np.sum(b_data))

#     # Collect Legacy data
#     fi_b.calculate_wake(yaw_angles=yc)
#     b_data = np.array(fi_b.get_turbine_power())/ 1000. 
#     total_gch_off.append(np.sum(b_data))

#     ax = axarr[y_idx]
#     ax.set_title(yc)
#     ax.plot(['T0','T1','T2','T3','T4'], s_data,'k',marker='s',label='SOWFA')
#     ax.plot(['T0','T1','T2','T3','T4'], g_data,'g',marker='o',label='GCH ON')
#     ax.plot(['T0','T1','T2','T3','T4'], b_data,'b',marker='*',label='GCH OFF')

# axarr[-1].legend()

# # Calculate totals and normalized totals
# total_sowfa = np.array(total_sowfa)
# nom_sowfa = total_sowfa/total_sowfa[0]

# total_gch_on = np.array(total_gch_on)
# nom_gch_on = total_gch_on/total_gch_on[0]

# total_gch_off = np.array(total_gch_off)
# nom_gch_off = total_gch_off/total_gch_off[0]

# fig, axarr = plt.subplots(1,2,sharex=True,sharey=False,figsize=(8,5))

# # Show results
# ax  = axarr[0]
# ax.set_title("Total Power")
# ax.plot(yaw_names,total_sowfa,'k',marker='s',label='SOWFA',ls='None')
# ax.axhline(total_sowfa[0],color='k',ls='--')
# ax.plot(yaw_names,total_gch_on,'g',marker='o',label='GCH ON',ls='None')
# ax.axhline(total_gch_on[0],color='g',ls='--')
# ax.plot(yaw_names,total_gch_off,'b',marker='*',label='GCH OFF',ls='None')
# ax.axhline(total_gch_off[0],color='b',ls='--')
# ax.legend()

# # Normalized results
# ax  = axarr[1]
# ax.set_title("Normalized Power")
# ax.plot(yaw_names,nom_sowfa,'k',marker='s',label='SOWFA',ls='None')
# ax.axhline(nom_sowfa[0],color='k',ls='--')
# ax.plot(yaw_names,nom_gch_on,'g',marker='o',label='GCH ON',ls='None')
# ax.axhline(nom_gch_on[0],color='g',ls='--')
# ax.plot(yaw_names,nom_gch_off,'b',marker='*',label='GCH OFF',ls='None')
# ax.axhline(nom_gch_off[0],color='b',ls='--')

# plt.show()
