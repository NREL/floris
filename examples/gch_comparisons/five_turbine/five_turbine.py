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
 

"""
Compare to the 5 turbine cases of 
https://www.wind-energ-sci-discuss.net/wes-2020-3/
"""

import matplotlib.pyplot as plt
import floris.tools as wfct
import numpy as np
import pandas as pd
import copy

# Parameter
use_nominal_values = True

# Load the paper results
df_results = pd.read_csv('paper_results_5_turbine.csv')

# # Initialize the FLORIS interface fi
fi_gl = wfct.floris_interface.FlorisInterface("../../example_input.json")
fi_new = wfct.floris_interface.FlorisInterface(
    "../../other_jsons/input_sowfa_tuning.json"
)

# Select gauss legacy with GCH off
# fi_gl.floris.farm.set_wake_model('gauss_legacy')
fi_gl.set_gch(False) # Disable GCH
# fi_gl.floris.farm.set_wake_model('gauss_merge')

# # Match the layout
x_layout = tuple(map(
    float,
    df_results.layout_x.values[0].replace('(','').replace(')','').split(',')
))
y_layout = tuple(map(
    float,
    df_results.layout_y.values[0].replace('(','').replace(')','').split(',')
))
fi_gl.reinitialize_flow_field(layout_array=[x_layout,y_layout])
fi_new.reinitialize_flow_field(layout_array=[x_layout,y_layout])

# Match the inflow
U0 = df_results.floris_U0.values[0]
TI = df_results.floris_TI.values[0]
fi_gl.reinitialize_flow_field(wind_speed=U0,turbulence_intensity=TI)
fi_new.reinitialize_flow_field(wind_speed=U0,turbulence_intensity=TI)

# Set up the yar model
fi_yar = copy.deepcopy(fi_gl)
fi_yar.set_gch_yaw_added_recovery(True)

# Set up the ss model
fi_ss = copy.deepcopy(fi_gl)
fi_ss.set_gch_secondary_steering(True)

# Set up the gch model
fi_gch = copy.deepcopy(fi_gl)
fi_gch.set_gch(True)

# Produce restuls for each floris model
number_cases = 4
power_matrix_gl = np.zeros((number_cases,5))
power_matrix_yar = np.zeros((number_cases,5))
power_matrix_ss = np.zeros((number_cases,5))
power_matrix_gch = np.zeros((number_cases,5))
power_matrix_gch_new = np.zeros((number_cases,5))

for r_idx, row in df_results.iterrows():
    yaw_angles = row.loc[['yaw_0','yaw_1','yaw_2','yaw_3','yaw_4']].values

    # Gauss Legacy
    fi_gl.calculate_wake(yaw_angles=yaw_angles)
    power_matrix_gl[r_idx, :] = np.array(fi_gl.get_turbine_power()) / 1E3

    # YAR
    fi_yar.calculate_wake(yaw_angles=yaw_angles)
    power_matrix_yar[r_idx, :] = np.array(fi_yar.get_turbine_power()) / 1E3

    # SS
    fi_ss.calculate_wake(yaw_angles=yaw_angles)
    power_matrix_ss[r_idx, :] = np.array(fi_ss.get_turbine_power()) / 1E3

    # GCH
    fi_gch.calculate_wake(yaw_angles=yaw_angles)
    power_matrix_gch[r_idx, :] = np.array(fi_gch.get_turbine_power()) / 1E3

    # GCH (new model)
    fi_new.calculate_wake(yaw_angles=yaw_angles)
    power_matrix_gch_new[r_idx, :] = np.array(fi_new.get_turbine_power()) / 1E3

# Loop over the cases and plot results
fig, axarr = plt.subplots(1,5,figsize=(20,9),sharex=True)
x_ticks = ['Base','25','25/25','25/25/25']
for t_idx, t in enumerate([1,2,3,4,'total']):

    ax = axarr[t_idx]

    # Get the paper results
    results_sowfa = df_results['sowfa_power_%s' % str(t)].values
    results_gauss = df_results['gauss_power_%s' % str(t)].values
    results_yar = df_results['yar_power_%s' % str(t)].values
    results_ss = df_results['ss_power_%s' % str(t)].values
    results_gch = df_results['gch_power_%s' % str(t)].values

    if use_nominal_values:
        results_sowfa = results_sowfa/results_sowfa[0]
        results_gauss = results_gauss/results_gauss[0]
        results_yar = results_yar/results_yar[0]
        results_ss = results_ss/results_ss[0]
        results_gch = results_gch/results_gch[0]

    # Plot the paper resuls
    ax.plot(x_ticks,results_sowfa,'ks-',label='SOWFA')
    ax.plot(x_ticks,results_gauss,'s-',color='orange',label='Paper Gauss')
    ax.plot(x_ticks,results_yar,'gs-',label='Paper YAR')
    ax.plot(x_ticks,results_ss,'bs-',label='Paper SS')
    ax.plot(x_ticks,results_gch,'ms-',label='Paper GCH')

    # Plot the new result

    # # GL
    if t == 'total':
        result = power_matrix_gl.sum(axis=1)
    else:
        result =power_matrix_gl[:,t]
    if use_nominal_values:
        result = result / result[0]
    ax.plot(x_ticks,result,'o--',color='orange',label='Gauss')

    # # YAR
    if t == 'total':
        result = power_matrix_yar.sum(axis=1)
    else:
        result =power_matrix_yar[:,t]
    if use_nominal_values:
        result = result / result[0]
    ax.plot(x_ticks,result,'go--',label='YAR')
    
    # # SS
    if t == 'total':
        result = power_matrix_ss.sum(axis=1)
    else:
        result =power_matrix_ss[:,t]
    if use_nominal_values:
        result = result / result[0]
    ax.plot(x_ticks,result,'bo--',label='SS')

    # GCH
    if t == 'total':
        result = power_matrix_gch.sum(axis=1)
    else:
        result =power_matrix_gch[:,t]
    if use_nominal_values:
        result = result / result[0]
    ax.plot(x_ticks,result,'mo--',label='GCH')

    # GCH
    if t == 'total':
        result = power_matrix_gch_new.sum(axis=1)
    else:
        result =power_matrix_gch_new[:,t]
    if use_nominal_values:
        result = result / result[0]
    ax.plot(x_ticks,result,'co--',label='GCH (New Model)',lw=3)

    # Make the plot title
    if t == 'total':
        ax.set_title('Total Power')
        ax.legend()
    else:
        ax.set_title('T %d' % t)
    
    ax.grid(True)

plt.show()
