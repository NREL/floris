# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# See read the https://floris.readthedocs.io for documentation


import matplotlib.pyplot as plt
import floris.tools as wfct
import floris.tools.visualization as vis
import floris.tools.cut_plane as cp
from floris.tools.optimization import YawOptimizationWindRose
import floris.tools.wind_rose as rose
import floris.tools.power_rose as pr
import numpy as np
import pandas as pd

# Instantiate the FLORIS object
fi = wfct.floris_interface.FlorisInterface("example_input.json")

# Define wind farm coordinates and layout
wf_coordinate = [39.8283, -98.5795]

# Set wind farm to N_row x N_row grid with constant spacing 
# (2 x 2 grid, 5 D spacing)
D = fi.floris.farm.turbines[0].rotor_diameter
N_row = 2
spc = 5
layout_x = []
layout_y = []
for i in range(N_row):
	for k in range(N_row):
		layout_x.append(i*spc*D)
		layout_y.append(k*spc*D)
N_turb = len(layout_x)

fi.reinitialize_flow_field(layout_array=(layout_x, layout_y),wind_direction=[270.0],wind_speed=[8.0])
fi.calculate_wake()

# set min and max yaw offsets for optimization
min_yaw = 0.0
max_yaw = 25.0

# Define minimum and maximum wind speed for optimizing power. 
# Below minimum wind speed, assumes power is zero.
# Above maximum_ws, assume optimal yaw offsets are 0 degrees
minimum_ws = 8.0
maximum_ws = 9.0

# ================================================================================
print('Plotting the FLORIS flowfield...')
# ================================================================================

# Initialize the horizontal cut
hor_plane = fi.get_hor_plane()

# Plot and show
fig, ax = plt.subplots()
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
ax.set_title('Baseline flow for U = 8 m/s, Wind Direction = 270$^\circ$')

# ================================================================================
print('Importing wind rose data...')
# ================================================================================

# Create wind rose object and import wind rose dataframe using WIND Toolkit HSDS API.
# Alternatively, load existing .csv file with wind rose information.
calculate_new_wind_rose = False

wind_rose = rose.WindRose()

if calculate_new_wind_rose:

	wd_list = np.arange(0,360,5)
	ws_list = np.arange(0,26,1)

	df = wind_rose.import_from_wind_toolkit_hsds(wf_coordinate[0],
	                                                    wf_coordinate[1],
	                                                    ht = 100,
	                                                    wd = wd_list,
	                                                    ws = ws_list,
	                                                    limit_month = None,
	                                                    st_date = None,
	                                                    en_date = None)

else:
	df = wind_rose.load('windtoolkit_geo_center_us.p')

# plot wind rose
wind_rose.plot_wind_rose()

# =============================================================================
print('Finding baseline and optimal yaw angles in FLORIS...')
# =============================================================================

# Instantiate the Optimization object
yaw_opt = YawOptimizationWindRose(fi, df.wd, df.ws,
                               minimum_yaw_angle=min_yaw,
                               maximum_yaw_angle=max_yaw,
                               minimum_ws=minimum_ws,
                               maximum_ws=maximum_ws)

# Determine baseline power with and without wakes
df_base = yaw_opt.calc_baseline_power()


# combine wind farm-level power into one dataframe
df_power = pd.DataFrame({'ws':df.ws,'wd':df.wd, \
    'freq_val':df.freq_val,'power_no_wake':df_base.power_no_wake, \
    'power_baseline':df_base.power_baseline})

# Set up the power rose
df_turbine_power_no_wake = pd.DataFrame([list(row) for row in df_base['turbine_power_no_wake']],columns=[str(i) for i in range(1,N_turb+1)])
df_turbine_power_no_wake['ws'] = df.ws
df_turbine_power_no_wake['wd'] = df.wd
df_turbine_power_baseline = pd.DataFrame([list(row) for row in df_base['turbine_power_baseline']],columns=[str(i) for i in range(1,N_turb+1)])
df_turbine_power_baseline['ws'] = df.ws
df_turbine_power_baseline['wd'] = df.wd

case_name = 'Example '+str(N_row)+' x '+str(N_row)+ ' Wind Farm'
power_rose = pr.PowerRose(case_name, df_power, df_turbine_power_no_wake, df_turbine_power_baseline)

# Display AEP analysis
fig, axarr = plt.subplots(2, 1, sharex=True, figsize=(6.4, 6.5))
power_rose.plot_by_direction(axarr)
power_rose.report()

plt.show()
