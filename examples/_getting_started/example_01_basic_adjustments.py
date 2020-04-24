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

"""
This example reviews two essential functions of the FLORIS interface
reinitialize_flow_field and calculate_wake

reinitialize_flow_field is used to change the layout and inflow of the farm 
while calculate_wake computed the wake velocities, deflections and combinations

Both functions provide a simpler interface to the underlying functions in the FLORIS class

Using them ensures that necessary recalcuations occur with changing certain variables

Note that it is typically necessary to call calculate_wake after reinitialize_flow_field,
but the two functions are seperated so that calculate_wake can be called repeatedly,
for example when optimizing yaw angles
"""

# Initialize the FLORIS interface fi
fi = wfct.floris_interface.FlorisInterface("../example_input.json")

# Declare a short-cut visualization function for brevity in this example
def plot_slice_shortcut(fi, ax, title):
    # Get horizontal plane at default height (hub-height)
    hor_plane = fi.get_hor_plane()
    wfct.visualization.visualize_cut_plane(hor_plane, ax=ax, minSpeed=4.0, maxSpeed=8.0)


# Define a plot
fig, axarr = plt.subplots(3,3,sharex=True,figsize=(12,5))
axarr = axarr.flatten()

# Plot the initial setup
fi.calculate_wake()
plot_slice_shortcut(fi, axarr[0], 'Initial')

# Change the wind speed
fi.reinitialize_flow_field(wind_speed=7.0)
fi.calculate_wake()
plot_slice_shortcut(fi, axarr[1], 'WS=7')

# Change the wind direction
fi.reinitialize_flow_field(wind_direction=320.)
fi.calculate_wake()
plot_slice_shortcut(fi, axarr[2], 'WD=280')

# Change the TI
fi.reinitialize_flow_field(turbulence_intensity=.15)
fi.calculate_wake()
plot_slice_shortcut(fi, axarr[3], 'TI=15%')

# Change the shear
fi.reinitialize_flow_field(wind_shear=.2)
fi.calculate_wake()
plot_slice_shortcut(fi, axarr[4], 'Shear=.2')

# Change the veer
fi.reinitialize_flow_field(wind_veer=5) #TODO IS THIS RIGHT?
fi.calculate_wake()
plot_slice_shortcut(fi, axarr[5], 'Veer=5')

# Change the air density
fi.reinitialize_flow_field(air_density=1.0) #TODO IS THIS RIGHT?
fi.calculate_wake()
plot_slice_shortcut(fi, axarr[6], 'Air Density=1.0')

# Change the farm layout
fi.reinitialize_flow_field(layout_array=[[0,500],[0,0]]) #TODO IS THIS RIGHT?
fi.calculate_wake()
plot_slice_shortcut(fi, axarr[7], 'Change layout')
wfct.visualization.plot_turbines_with_fi(axarr[7], fi)

# Changes the yaw angles
fi.calculate_wake(yaw_angles=[25,10])
plot_slice_shortcut(fi, axarr[8], 'Change yaw angles')
wfct.visualization.plot_turbines_with_fi(axarr[8], fi)


plt.show()
