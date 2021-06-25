# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


import matplotlib.pyplot as plt

import floris.tools as wfct
import numpy as np

# Initialize the FLORIS interface fi
# For basic usage, the florice interface provides a simplified interface to
# the underlying classes

files = ['/Users/jannoni/Desktop/Desktop/Repos/FLORIS_RDHH/floris_sowfa_analysis/analysis/floris_models/nrel_5.json']
         # '/Users/jannoni/Desktop/Desktop/Repos/FLORIS_RDHH/floris_sowfa_analysis/analysis/floris_models/iea_10.json',
         # '/Users/jannoni/Desktop/Desktop/Repos/FLORIS_RDHH/floris_sowfa_analysis/analysis/floris_models/iea_15.json']

# HH = [69, 75.6, 82, 88.2, 94.5]
HH = [69, 94.5]

plt.figure()
colors = ['b','g','m','k','y']
alpha = [0.25,1.0]
for j in range(2):
    for i in range(len(HH)):
        fi = wfct.floris_interface.FlorisInterface(files[0])
        fi.change_turbine([0], {"hub_height": HH[i]})

        if j == 0:
            fi.floris.farm.wake.velocity_model.flag_orig_vel_def = False
        elif j == 1:
            fi.floris.farm.wake.velocity_model.flag_orig_vel_def = True

        # Calculate wake
        fi.calculate_wake()

        # Get horizontal plane at default height (hub-height)
        hor_plane = fi.get_hor_plane()

        # Get horizontal plane at default height (hub-height)
        h = HH[i]
        vert_plane = fi.get_y_plane(y_loc=0.0,z_bounds=(0.1,300))

        # Plot and show
        # fig, ax = plt.subplots()
        # wfct.visualization.visualize_cut_plane(vert_plane, ax=ax)

        x_locs = np.unique(vert_plane.df['x1'])
        D = fi.floris.farm.turbines[0].rotor_diameter

        idx_shear = np.min(np.where(x_locs > -1 * D))
        idx1 = np.where(vert_plane.df['x1'] == x_locs[idx_shear])[0]
        shear = vert_plane.df['u'][idx1]

        idx = np.min(np.where(x_locs > 5 * D))
        idx1 = np.where(vert_plane.df['x1'] == x_locs[idx])[0]

        if j == 0:
            strHH = 'With Ground Hub Height = ' + str(h)
            plt.plot(np.array(vert_plane.df['u'][idx1]),vert_plane.df['x2'][idx1],label=strHH,c=colors[i],alpha=alpha[j])
        if j == 1:
            strHH = 'Without Ground Hub Height = ' + str(h)
            plt.plot(np.array(vert_plane.df['u'][idx1]),vert_plane.df['x2'][idx1],label=strHH,c=colors[i],alpha=alpha[j])
            # plt.plot([np.min(vert_plane.df['u'][idx1]),np.min(vert_plane.df['u'][idx1])], [h - D / 2, h + D / 2],c=colors[i])

idx_shear = np.min(np.where(x_locs > -1 * D))
idx1 = np.where(vert_plane.df['x1'] == x_locs[idx_shear])[0]
plt.plot(vert_plane.df['u'][idx1],vert_plane.df['x2'][idx1],'k--')
plt.grid()
plt.legend(fontsize=12)
plt.tick_params(which='both',labelsize=12)
plt.xlabel('u (m/s)', fontsize=12)
plt.ylabel('z (m)', fontsize=12)
plt.show()
