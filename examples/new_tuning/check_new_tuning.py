# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See read the https://floris.readthedocs.io for documentation

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

## Grab certain hi-TI five simulations from saved SOWFA data set
df = pd.read_pickle('../sowfa_comparisons/sowfa_data_set/sowfa_data_set.p')

## Limit data
df = df[df.sowfa_U0 == sowfa_U0]
df = df[df.num_turbines == num_turbines]

## Add an aligned flag
df['is_aligned'] = False
df.loc[df.yaw.apply(lambda x: np.max(np.abs(x))==0),'is_aligned'] = True

#df = df[df.is_aligned == True]

## Define the FLORIS cases
fi_legacy = wfct.floris_interface.FlorisInterface("../other_jsons/input_legacy.json")
fi_default = wfct.floris_interface.FlorisInterface("../example_input.json")

## New tuning cases
fi_tune =  wfct.floris_interface.FlorisInterface("../example_input.json")
fi_tune.floris.farm.wake.velocity_model.wake_rotation = False


# case_list = ['C_0014_0_0_7_0_0',
#                 'yaw_sweep_10',
#                 'ttmw_012',
#                 'ttmw_002',
#                 'C_0015_0_0_7_0_1',
#                 'C_0051_20_0_7_0_1']
df_result = pd.DataFrame()
df = df.reset_index(drop=True)
for i in range(df.shape[0]):
# for c in df.case_name.values:
#     row = df[df.case_name==c]
# for i, row in df.iterrows():
    row = df.loc[i]

    # Get the values
    layout_x = row.layout_x# .values[0]
    layout_y = row.layout_y# .values[0]
    yaw = row.yaw# .values[0]
    floris_U0 = row.floris_U0#.values[0]
    floris_TI = row.floris_TI#.values[0]

    if not (layout_x[0]==1000.):
        continue
    if not (layout_y[0]==1000.):
        continue

    
    # layout to D
    row['layout_x'] = '_'.join(['%.1f' % ((x-1000.)/126) for x in layout_x[1:]])
    row['layout_y'] = '_'.join(['%.1f' % ((x-1000.)/126) for x in layout_y[1:]])


    # Limit row to a few values
    row = row[['floris_TI','layout_x','layout_y','is_aligned','yaw','power']]
    
    # Seperate power by turbine
    sowfa_power = row.power#.values[0]
    for t, p in enumerate(sowfa_power):
        row['t_%d' % t] = p
    row['t_total'] = np.sum(sowfa_power)
    row = row.drop('power')#,axis=1)
    

    # Resimulate the FLORIS cases
    for fi, fi_name in zip([fi_legacy, fi_default, fi_tune],['leg', 'newdef', 'wr_off']):
        fi.reinitialize_flow_field(
            layout_array=[layout_x,layout_y],
            wind_speed=[floris_U0],
            turbulence_intensity=[floris_TI]
        )
        fi.calculate_wake(yaw)
        floris_power = np.array(fi.get_turbine_power())/1000.
        for t, p in enumerate(floris_power):
            #row['%s_%d' % (fi_name,t)] = np.round(p,1) #/ sowfa_power[t]
            row['%s_%d' % (fi_name,t)] = np.round(p,2)#/ sowfa_power[t],2)

            # Compute error to sowfa
            sp = row['t_%d' % t] 
            row['%s_%d_err' % (fi_name,t)] = np.round(100 * (p-sp)/sp,1)
        row['%s_total' % fi_name] = np.sum(floris_power) #/ np.sum(sowfa_power)
        row['%s_total' % fi_name] = np.round(100.*(np.sum(floris_power)-np.sum(sowfa_power)) / np.sum(sowfa_power),2)

    #row = row.reset_index(drop=True)
    df_result = df_result.append(row)
    #break

    # Add the normalized row 
# print(row)
# print(df_result.head())

# Handle the aligned data first
df_aligned = df_result[df_result.is_aligned == True]

# Resort the columns
cols = df_aligned.columns.values.tolist()
cols_final = []
for t in range(40):
    cols_final = cols_final + [c for c in cols if str(t) in c and 'err' in c]
cols_final = cols_final + [c for c in cols if 'total' in c]
print(cols_final)
print(cols[:5])
# cols_final = [c for c in cols if not (c in cols_final)] + cols_final
cols_final = ['floris_TI','layout_x','layout_y','is_aligned','yaw'] + cols_final
print(cols_final)
df_aligned = df_aligned[cols_final]
df_aligned = df_aligned.sort_values(['floris_TI','layout_x','layout_y'])
df_aligned.to_excel('result_aligned_%d.xlsx' % num_turbines)

# Limit to aligned data



  
# # Resort the columns
# cols = df_result.columns
# cols_final = []
# for t in range(40):
#     cols_final = cols_final + [c for c in cols if str(t) in c]
# cols_final = cols_final + [c for c in cols if 'total' in c]
# print(cols_final)
# cols_final = [c for c in cols if not (c in cols_final)] + cols_final
# print(cols_final)
# df_result = df_result[cols_final]
# df_result = df_result.sort_values(['floris_TI','layout_x','layout_y'])