#
# Copyright 2019 NREL
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
#

# See read the https://floris.readthedocs.io for documentation

import floris.tools as wfct
import floris.tools.visualization as vis
import floris.tools.cut_plane as cp
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


D = 126

fig, axarr = plt.subplots(2,3,sharex=True,sharey=True)


# Load in results
df_power_full = pd.read_pickle('data_sowfa.p')

# List simulation options
ti_vals = ['low','hi']
yaw_values = [0,10,20]
x_locations_unique = sorted(df_power_full.layout_x.unique())

# These dont change
y_locations = np.array(df_power_full.layout_y.values[0])


# Iniitialize FLORIS
fi = wfct.floris_interface.FlorisInterface("example_input.json")
fi_b = wfct.floris_interface.FlorisInterface("example_input.json")
fi_b.floris.farm.set_wake_model('blondel')



for ti_idx, ti in enumerate(ti_vals):
    if ti == 'hi':
        df_power_ti = df_power_full[df_power_full.TI > 0.07]
        ti_val = 0.09
        wind_speed = 8.25
    else:
        df_power_ti = df_power_full[df_power_full.TI < 0.07]
        ti_val = 0.065
        wind_speed = 8.33

    for yaw_idx, yaw in enumerate(yaw_values):

        df_power_yaw = df_power_ti[df_power_ti.yaw_0==yaw]
        d_array = []
        results_sowfa_1 = []
        results_sowfa_2 = []
        gauss = []
        blondel = []

        for x_locations in x_locations_unique:

        
            df_inner = df_power_yaw[df_power_yaw.layout_x==x_locations]
            
            yaw_array = np.array([df_inner.yaw_0.values,df_inner.yaw_1.values,df_inner.yaw_2.values,df_inner.yaw_3.values])
            sowfa_power_array = np.array([df_inner.sowfa_power_0.values,df_inner.sowfa_power_1.values,df_inner.sowfa_power_2.values,df_inner.sowfa_power_3.values])

            # Set up FLORIS and get powers
            fi.reinitialize_flow_field(wind_speed=[wind_speed],turbulence_intensity=[ti_val],layout_array =[x_locations,y_locations])
            fi.calculate_wake(yaw_angles=yaw_array)
            floris_power_array = np.array([p[0]/1000. for p in fi.get_turbine_power()])

            # Repeat BLONDEL
            fi_b.reinitialize_flow_field(wind_speed=[wind_speed],turbulence_intensity=[ti_val],layout_array =[x_locations,y_locations])
            fi_b.calculate_wake(yaw_angles=yaw_array)
            floris_b_power_array = np.array([p[0]/1000. for p in fi_b.get_turbine_power()])


            # Save all the results
            d_loc = (x_locations[2] - x_locations[0])/D
            d_array.append(d_loc)
            results_sowfa_1.append(sowfa_power_array[2])
            results_sowfa_2.append(sowfa_power_array[3])
            gauss.append(floris_power_array[3]) # Identical, just pick one
            blondel.append(floris_b_power_array[3]) # Identical, just pick one

        ax = axarr[ti_idx, yaw_idx]
        ax.plot(d_array,results_sowfa_1, color='k',marker='o',ls='None',label='SOWFA 1')
        ax.plot(d_array,results_sowfa_2, color='k',marker='x',ls='None',label='SOWFA 2')
        ax.plot(d_array,gauss, color='g',marker='.',label="gauss")
        ax.plot(d_array,blondel, color='violet',marker='.',label="blondel")
        ax.set_title('%s TI, yaw=%d' % (ti,yaw))
        ax.grid(True)
        if (ti_idx==0) and (yaw_idx==0):
            ax.legend()





plt.show()

