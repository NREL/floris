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
layout_x = (1000.0, 1882.0, 2764.0) # 7D spacing
layout_y = (1000.0, 1000.0, 1000.0)
sowfa_results = np.array([
[1703,891.2,1087.6,-20,-20,0],
[1701.8,994.4,1013.1,-20,-10,0],
[1701.2,1043.4,947.5,-20,0,0],
[1702.1,1034.7,900.4,-20,10,0],
[1702.6,972.2,910.3,-20,20,0],
[1880.3,799.6,1035,-10,-20,0],
[1878.7,881.3,1001.7,-10,-10,0],
[1878.2,915.8,964.9,-10,0,0],
[1879.3,899.1,954.5,-10,10,0],
[1880,836,973.4,-10,20,0],
[1943,827,971.6,0,-20,0],
[1941.2,896.7,943.7,0,-10,0],
[1941.1,918.7,947.9,0,0,0],
[1942.1,889.6,992.2,0,10,0],
[1942.8,814.2,1055.1,0,20,0],
[1874.1,951.8,980.9,10,-20,0],
[1872.3,1031.8,1037.8,10,0,0],
[1873.5,987.5,1096.8,10,10,0],
[1874.3,891.7,1184.2,10,20,0],
[1691.4,1132.2,949.6,20,-20,0],
[1690.2,1207.2,1005.9,20,-10,0],
[1690.1,1212.3,1015,20,-9,0],
[1690,1215.2,1023.8,20,-8,0],
[1689.9,1218.3,1033.4,20,-7,0],
[1689.8,1220,1039.5,20,-6,0],
[1689.8,1221.2,1049.2,20,-5,0],
[1689.8,1221.7,1055.8,20,-4,0],
[1689.9,1221.2,1064.5,20,-3,0],
[1690,1220.7,1069.9,20,-2,0],
[1690,1218.9,1079.5,20,-1,0],
[1690.1,1216.9,1086.1,20,0,0],
[1690.3,1214.7,1093,20,1,0],
[1690.3,1210.4,1105.7,20,2,0],
[1690.4,1207.2,1115,20,3,0],
[1690.6,1201.4,1132.1,20,4,0],
[1690.7,1197,1142,20,5,0],
[1690.8,1189.7,1153.9,20,6,0],
[1690.9,1184.1,1163.6,20,7,0],
[1691.2,1174.9,1177.4,20,8,0],
[1691.2,1168.2,1186.2,20,9,0],
[1691.3,1157.5,1206.5,20,10,0],
[1692.2,1036.4,1330.7,20,20,0]
])
df_sowfa = pd.DataFrame(sowfa_results, 
                        columns = ['p0','p1','p2','y0','y1','y2'] )

## SET UP FLORIS AND MATCH TO BASE CASE
wind_speed = 8.38
TI = 0.09

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

