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

## Grab certain hi-TI five simulations from saved SOWFA data set
df_sowfa = pd.read_pickle('../sowfa_data_set/sowfa_data_set.p')

# Limit to aligned
df_sowfa = df_sowfa[df_sowfa.yaw.apply(lambda x: np.max(np.abs(x)))==0.0]

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

# Compute an error term
for floris_label in fi_dict:
    (fi, floris_color, floris_marker) = fi_dict[floris_label]
    df_sowfa['error_' + floris_label] = df_sowfa[floris_label] - df_sowfa.power

# List the error by number of wakes and by floris interface
for num_wakes in range(1,6):
    print('%d Wakes Impinging ================' % num_wakes)
    print('Model\t\tRSME\tMean Error')
    for floris_label in fi_dict:
        (fi, floris_color, floris_marker) = fi_dict[floris_label]
        result = []
        for i, row in df_sowfa.iterrows():
            for e in row['error_' + floris_label][row.wake_table==num_wakes]:
                result.append(e)

        result = np.array(result)
        print(
            '%s\t\t%.1f\t%.1f' %
            (
                floris_label,np.sqrt(np.sum(result**2)/len(result)),
                np.mean(result)
            )
        )

# downstream_power = lambda x: np.mean(x[2:])

# # Compare the downstream power production
# fig, ax = plt.subplots()
# df_sowfa['downstream_sowfa_power'] = df_sowfa.power.apply(downstream_power)
# ax.plot(df_sowfa.d_spacing,df_sowfa.downstream_sowfa_power,'ks-',label='SOWFA')
# # Plot the FLORIS results
# for floris_label in fi_dict:
#     (fi, floris_color, floris_marker) = fi_dict[floris_label]
#     df_sowfa['floris_downstream'] = df_sowfa[floris_label].apply(downstream_power)
#     ax.plot(df_sowfa.d_spacing,df_sowfa.floris_downstream,color=floris_color,marker=floris_marker,label=floris_label)
# ax.grid(True)
# ax.set_xlabel('Distance Downstream (D)')
# ax.set_ylabel('Power of downstream turbine (kW')
# ax.legend()

# plt.show()
