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
# num_turbines = 38
sowfa_U0 = 8.0

for num_turbines in [2,5,3,38]:

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
    fi_legacy_2 = wfct.floris_interface.FlorisInterface("../other_jsons/input_legacy_2.json")
    fi_default = wfct.floris_interface.FlorisInterface("../example_input.json")

    ## New tuning cases
    fi_tune =  wfct.floris_interface.FlorisInterface("../example_input.json")
    # fi_tune.floris.farm.wake.velocity_model.wake_rotation = False
    fi_tune.floris.farm.wake.velocity_model.wake_rotation = True
    fi_tune.floris.farm.wake.velocity_model.gamma_scale = 1.5
    fi_tune.floris.farm.wake.velocity_model.gamma_rotation_scale = 0.1

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

        if num_turbines < 10:
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
        for fi, fi_name in zip([fi_legacy,fi_legacy_2, fi_default, fi_tune],['leg','leg_2', 'newdef', 'wr_off']):
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
            row['%s_total_err' % fi_name] = np.round(100.*(np.sum(floris_power)-np.sum(sowfa_power)) / np.sum(sowfa_power),2)

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
    cols_final = cols_final + [c for c in cols if 'total' in c and 'err' in c]
    # print(cols_final)
    # print(cols[:5])
    # cols_final = [c for c in cols if not (c in cols_final)] + cols_final
    cols_final = ['floris_TI','layout_x','layout_y','is_aligned','yaw'] + cols_final
    # print(cols_final)
    df_aligned = df_aligned[cols_final]
    df_aligned = df_aligned.sort_values(['floris_TI','layout_x','layout_y'])
    df_aligned.to_excel('result_aligned_%d.xlsx' % num_turbines)



    # Reset df_aligned
    df_aligned = df_result[df_result.is_aligned == True]
    df_result = df_result[df_result.is_aligned == False].reset_index(drop=True)


    df_gain = pd.DataFrame()
    # Loop through df_result and convert to gains:
    for i in range(df_result.shape[0]):
        row = df_result.loc[i]

        # Find the aligned row to match
        aligned_row = df_aligned[(df_aligned.floris_TI==row.floris_TI)  & 
                                (df_aligned.layout_x==row.layout_x) & 
                                (df_aligned.layout_y==row.layout_y)    ]#.iloc[0]

        if aligned_row.shape[0] == 0:
            continue
        else:
            aligned_row = aligned_row.iloc[0]

            for model in ['t','leg','leg_2' ,'newdef', 'wr_off']:
                for t in range(num_turbines):
                

                    # Compute the gain
                    row['%s_%d' % (model,t)] = 100 * (row['%s_%d' % (model,t)] - aligned_row['%s_%d' % (model,t)] ) / aligned_row['%s_%d' % (model,t)]

                    # Update the error
                    row['%s_%d_err' % (model,t)] = np.round(100 * (row['%s_%d' % (model,t)]-row['%s_%d' % ('t',t)])/row['%s_%d' % ('t',t)],1)



                # Compute the total gain
                row['%s_total' % (model)] = 100 * (row['%s_total' % (model)] - aligned_row['%s_total' % (model)] ) / aligned_row['%s_total' % (model)]

            df_gain = df_gain.append(row)


    # Rename the SOWFA columns to a for better sorting
    for t in range(num_turbines):
        df_gain=df_gain.rename(columns={'t_%d'%t:'a_%d'%t})

    df_gain=df_gain.rename(columns={'t_total':'a_total'})
    df_gain = df_gain[sorted(df_gain.columns.values)]


    # Resort the columns
    cols = df_gain.columns
    cols_final = []
    for t in range(40):
        cols_final = cols_final + [c for c in cols if str(t) in c and 'err' not in c]
    cols_final = cols_final + [c for c in cols if 'total' in c and 'err' not in c]
    print(cols_final)
    # cols_final = [c for c in cols if not (c in cols_final)] + cols_final
    cols_final = ['floris_TI','layout_x','layout_y','is_aligned','yaw'] + cols_final
    print(cols_final)
    df_gain = df_gain[cols_final]
    df_gain = df_gain.sort_values(['floris_TI','layout_x','layout_y'])

    df_gain.to_excel('result_gain_%d.xlsx' % num_turbines)