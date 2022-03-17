# Copyright 2022 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

import numpy as np
import matplotlib.pyplot as plt
from floris.tools import FlorisInterface, UncertaintyInterface
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import (
    YawOptimizationSR,
)

"""
This example demonstrates how to perform a yaw optimization for multiple wind directions and 1 wind speed.

Yaw angles are computed with and without (default) uncertainity and improves are likewise computed with and without uncertaintiy
"""

wd_array = np.arange(250, 290.0, 1.0)

# Load the default example floris object
fi = FlorisInterface("inputs/gch.yaml") 
fi_unc = UncertaintyInterface("inputs/gch.yaml")

# Reinitialize as a 3-turbine farm with range of WDs and 1 WS
D = 126.0 # Rotor diameter for the NREL 5 MW
fi.reinitialize(
    layout=[[0.0, 5 * D, 10 * D], [0.0, 0.0, 0.0]],
    wind_directions=wd_array, 
    wind_speeds=[8.0],
)
fi_unc.reinitialize(
    layout=[[0.0, 5 * D, 10 * D], [0.0, 0.0, 0.0]],
    wind_directions=wd_array,
    wind_speeds=[8.0],
)

# Initialize optimizer object and run optimization using the Serial-Refine method
yaw_opt = YawOptimizationSR(fi,minimum_yaw_angle=-25,maximum_yaw_angle=25)
yaw_opt_unc = YawOptimizationSR(fi_unc,minimum_yaw_angle=-25,maximum_yaw_angle=25)

df_opt = yaw_opt.optimize()
df_opt_unc = yaw_opt_unc.optimize()

# Extract the optimal yaw angles from each result and arrange into a num_wd x num_ws x num_turbine matrix
yaw_angles = np.zeros([len(wd_array),1,3])
yaw_angles[:,0,:] = np.vstack(df_opt.yaw_angles_opt.values)
yaw_angles_unc = np.zeros([len(wd_array),1,3])
yaw_angles_unc[:,0,:] = np.vstack(df_opt_unc.yaw_angles_opt.values)

# Now recompute the power without uncertainty using the angles computed with uncertainty and vise versa
fi.calculate_wake(yaw_angles=yaw_angles_unc)
pow_base_yaw_unc = fi.get_farm_power().flatten()

fi_unc.calculate_wake(yaw_angles=yaw_angles)
pow_unc_yaw_base = fi_unc.get_farm_power().flatten()

# Split out the turbine results
for t in range(3):
    df_opt['t%d' % t] = df_opt.yaw_angles_opt.apply(lambda x: x[t])
for t in range(3):
    df_opt_unc['t%d' % t] = df_opt_unc.yaw_angles_opt.apply(lambda x: x[t])

# Show the results
fig, axarr = plt.subplots(3,1,sharex=True,sharey=False,figsize=(8,8))

# Define a fixed color dictionary
color_dict = {0:'k',1:'b',2:'r'}

# Yaw results 
ax = axarr[0]
for t in range(3):
    ax.plot(df_opt.wind_direction,df_opt['t%d' % t],label='t%d' % t,ls='-',color=color_dict[t])
    ax.plot(df_opt_unc.wind_direction,df_opt_unc['t%d' % t],label='t%d (unc)' % t,ls='--',color=color_dict[t])
ax.set_ylabel('Yaw Offset (deg')
ax.legend()
ax.grid(True)

# Power results (No Uncertainity)
ax = axarr[1]
ax.plot(df_opt.wind_direction,df_opt.farm_power_baseline,color='k',label='Baseline Farm Power')
ax.plot(df_opt.wind_direction,df_opt.farm_power_opt,color='r',label='Optimized Farm Power')
ax.plot(df_opt.wind_direction,pow_base_yaw_unc,color='r',ls='--',label='Optimized Farm Power (Unc Yaw)')
ax.set_ylabel('Power (W)')
ax.set_xlabel('Wind Direction (deg)')
ax.legend()
ax.grid(True)
ax.set_title('Power computed without uncertainty')

# Power results (No Uncertainity)
ax = axarr[2]
ax.plot(df_opt_unc.wind_direction,df_opt_unc.farm_power_baseline,color='k',label='Baseline Farm Power')
ax.plot(df_opt_unc.wind_direction,pow_unc_yaw_base,color='r',label='Optimized Farm Power')
ax.plot(df_opt_unc.wind_direction,df_opt_unc.farm_power_opt,color='r',ls='--',label='Optimized Farm Power (Unc Yaw)')

ax.set_ylabel('Power (W)')
ax.set_xlabel('Wind Direction (deg)')
ax.legend()
ax.grid(True)
ax.set_title('Power computed with uncertainty')

plt.show()
