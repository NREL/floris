# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# See read the https://floris.readthedocs.io for documentation

# Compare 3 turbine results to SOWFA in 8 m/s, higher TI case

import matplotlib.pyplot as plt
import floris.tools as wfct
import numpy as np
import pandas as pd



# Write out SOWFA results
layout_x = (1000.0, 1882.0, 2764.0) # 7D Spacing
layout_y = (1000.0, 1000.0, 1000.0)
sowfa_results = np.array([
[1702.1,849.7,1196.7,-20,-20,0],
[1699.9,960.3,1035.1,-20,-10,0],
[1699.4,1018,898.1,-20,0,0],
[1701,1019.6,800.6,-20,10,0],
[1700.7,968.7,774.6,-20,20,0],
[1883.4,672.2,1035.1,-10,-20,0],
[1881,748.1,945.2,-10,-10,0],
[1879.9,783.7,865.5,-10,0,0],
[1881.7,775.5,835.5,-10,10,0],
[1882.7,726.6,831.9,-10,20,0],
[1949.5,667.1,906.9,0,-20,0],
[1947.5,722.1,878.3,0,-10,0],
[1946.8,738.8,869.4,0,0,0],
[1947.6,714,891.9,0,10,0],
[1949.2,651.6,927,0,20,0],
[1883.9,819,842.7,10,-20,0],
[1882.7,869.6,856.1,10,-10,0],
[1880.5,874.3,905.9,10,0,0],
[1882.1,829.3,993.6,10,10,0],
[1884.2,739.7,1093.1,10,20,0],
[1702.3,1103.7,833.4,20,-20,0],
[1701.9,1164.3,899.1,20,-10,0],
[1700.9,1162.7,1030.6,20,0,0],
[1702.3,1096.5,1192,20,10,0],
[1703.8,970.8,1349.4,20,20,0]
])
df_sowfa = pd.DataFrame(sowfa_results, 
                        columns = ['p0','p1','p2','y0','y1','y2'] )

## SET UP FLORIS AND MATCH TO BASE CASE
wind_speed = 8.39
TI = 0.065


# Initialize the FLORIS interface fi, use default gauss model
fi = wfct.floris_interface.FlorisInterface("example_input.json")
fi.reinitialize_flow_field(wind_speed=[wind_speed],turbulence_intensity=[TI],layout_array=(layout_x, layout_y))

# Setup blonel
fi_b = wfct.floris_interface.FlorisInterface("example_input.json")
fi_b.floris.farm.set_wake_model('blondel')
fi_b.reinitialize_flow_field(wind_speed=[wind_speed],turbulence_intensity=[TI],layout_array=(layout_x, layout_y))

# Compare yaw combinations
yaw_combinations = [
    (0,0,0), (20,0,0), (0,20,0), (20,20,0)
]
yaw_names = ['%d/%d/%d' % yc for yc in yaw_combinations]

# Plot individual turbine powers
fig, axarr = plt.subplots(1,4,sharex=True,sharey=True,figsize=(12,5))

total_sowfa = []
total_gauss = []
total_blondel = []

for y_idx, yc in enumerate(yaw_combinations):

    # Collect SOWFA DATA
    s_data = df_sowfa[(df_sowfa.y0==yc[0]) & (df_sowfa.y1==yc[1]) & (df_sowfa.y2==yc[2])]
    s_data = [s_data.p0.values[0], s_data.p1.values[0],s_data.p2.values[0]]
    total_sowfa.append(np.sum(s_data))

    # Collect Gauss data
    fi.calculate_wake(yaw_angles=yc)
    g_data = np.array(fi.get_turbine_power())/ 1000. 
    total_gauss.append(np.sum(g_data))

    # Collect Blondel data
    fi_b.calculate_wake(yaw_angles=yc)
    b_data = np.array(fi_b.get_turbine_power())/ 1000. 
    total_blondel.append(np.sum(b_data))

    ax = axarr[y_idx]
    ax.set_title(yc)
    ax.plot(['T0','T1','T2'], s_data,'k',marker='s',label='SOWFA')
    ax.plot(['T0','T1','T2'], g_data,'g',marker='o',label='Gauss')
    ax.plot(['T0','T1','T2'], b_data,'b',marker='*',label='Blondel')

axarr[-1].legend()

# Calculate totals and normalized totals
total_sowfa = np.array(total_sowfa)
nom_sowfa = total_sowfa/total_sowfa[0]

total_gauss = np.array(total_gauss)
nom_gauss = total_gauss/total_gauss[0]

total_blondel = np.array(total_blondel)
nom_blondel = total_blondel/total_blondel[0]

fig, axarr = plt.subplots(1,2,sharex=True,sharey=False,figsize=(8,5))

# Show results
ax  = axarr[0]
ax.set_title("Total Power")
ax.plot(yaw_names,total_sowfa,'k',marker='s',label='SOWFA',ls='None')
ax.axhline(total_sowfa[0],color='k',ls='--')
ax.plot(yaw_names,total_gauss,'g',marker='o',label='Gauss',ls='None')
ax.axhline(total_gauss[0],color='g',ls='--')
ax.plot(yaw_names,total_blondel,'b',marker='*',label='Blondel',ls='None')
ax.axhline(total_blondel[0],color='b',ls='--')
ax.legend()

# Normalized results
ax  = axarr[1]
ax.set_title("Normalized Power")
ax.plot(yaw_names,nom_sowfa,'k',marker='s',label='SOWFA',ls='None')
ax.axhline(nom_sowfa[0],color='k',ls='--')
ax.plot(yaw_names,nom_gauss,'g',marker='o',label='Gauss',ls='None')
ax.axhline(nom_gauss[0],color='g',ls='--')
ax.plot(yaw_names,nom_blondel,'b',marker='*',label='Blondel',ls='None')
ax.axhline(nom_blondel[0],color='b',ls='--')



plt.show()

