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
 

import matplotlib.pyplot as plt
import floris.tools as wfct
import numpy as np


# Side by Side, adjust T0 and T1 heights
fi = wfct.floris_interface.FlorisInterface('../example_input.json')
fi.reinitialize_flow_field(layout_array=[[0,0],[0,1000]])

# Calculate wake
fi.calculate_wake()
init_power = np.array(fi.get_turbine_power())/1000.

fig, axarr = plt.subplots(1,3,sharex=False,sharey=False,figsize=(15,5))

# Show the hub-height slice in the 3rd pane
hor_plane = fi.get_hor_plane()
wfct.visualization.visualize_cut_plane(hor_plane, ax=axarr[2])


for t in range(2):

    ax = axarr[t]

    # Now sweep the heights for this turbine
    heights =  np.arange(70,120,1.)
    powers = np.zeros_like(heights)

    for h_idx, h in enumerate(heights):
        fi.change_turbine([t],{'hub_height':h})
        fi.calculate_wake()
        powers[h_idx] = fi.get_turbine_power()[t]/1000.



    ax.plot(heights, powers, 'k')
    ax.axhline(init_power[t],color='r',ls=':')
    ax.axvline(90,color='r',ls=':')
    ax.set_title('T%d' % t)
    ax.set_xlim([70,120])
    ax.set_ylim([1000,2000])
    ax.set_xlabel('Hub Height T%d' % t)
    ax.set_ylabel('Power')

plt.suptitle('Adjusting Both Turbine Heights')


# Waked, adjust T0 height
fi = wfct.floris_interface.FlorisInterface('../example_input.json')
fi.reinitialize_flow_field(layout_array=[[0,500],[0,0]])

# Calculate wake
fi.calculate_wake()
init_power = np.array(fi.get_turbine_power())/1000.

fig, axarr = plt.subplots(1,3,sharex=False,sharey=False,figsize=(15,5))

# Show the hub-height slice in the 3rd pane
hor_plane = fi.get_hor_plane()
wfct.visualization.visualize_cut_plane(hor_plane, ax=axarr[2])


for t in range(2):

    ax = axarr[t]

    # Now sweep the heights for this turbine
    heights =  np.arange(70,120,1.)
    powers = np.zeros_like(heights)

    for h_idx, h in enumerate(heights):
        fi.change_turbine([0],{'hub_height':h})
        fi.calculate_wake()
        powers[h_idx] = fi.get_turbine_power()[t]/1000.

    ax.plot(heights, powers, 'k')
    ax.axhline(init_power[t],color='r',ls=':')
    ax.axvline(90,color='r',ls=':')
    ax.set_title('T%d' % t)
    ax.set_xlim([50,120])
    ax.set_xlabel('Hub Height T0')
    ax.set_ylabel('Power T%d' % t)

plt.suptitle('Adjusting T0 Height')

# Waked, adjust T1 height
fi = wfct.floris_interface.FlorisInterface('../example_input.json')
fi.reinitialize_flow_field(layout_array=[[0,500],[0,0]])

# Calculate wake
fi.calculate_wake()
init_power = np.array(fi.get_turbine_power())/1000.

fig, axarr = plt.subplots(1,3,sharex=False,sharey=False,figsize=(15,5))

# Show the hub-height slice in the 3rd pane
hor_plane = fi.get_hor_plane()
wfct.visualization.visualize_cut_plane(hor_plane, ax=axarr[2])


for t in range(2):

    ax = axarr[t]

    # Now sweep the heights for this turbine
    heights =  np.arange(70,120,1.)
    powers = np.zeros_like(heights)

    for h_idx, h in enumerate(heights):
        fi.change_turbine([1],{'hub_height':h})
        fi.calculate_wake()
        powers[h_idx] = fi.get_turbine_power()[t]/1000.

    ax.plot(heights, powers, 'k')
    ax.axhline(init_power[t],color='r',ls=':')
    ax.axvline(90,color='r',ls=':')
    ax.set_title('T%d' % t)
    ax.set_xlim([70,120])
    ax.set_xlabel('Hub Height T1')
    ax.set_ylabel('Power T%d' % t)

plt.suptitle('Adjusting T1 Height')

plt.show()
