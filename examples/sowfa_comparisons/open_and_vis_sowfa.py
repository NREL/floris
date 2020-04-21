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

sowfa_case = wfct.sowfa_utilities.SowfaInterface('sowfa_example')

# Summarize self
print(sowfa_case)

# Demonstrate flow field visualizations

# # Get the horizontal cut plane at 90 m
hor_plane = sowfa_case.get_hor_plane(90)

# Show the views in different permutations
fig, axarr = plt.subplots(3, 2, figsize=(10, 10))

# Original
ax = axarr[0, 0]
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
ax.set_title('Original')

# Set turbine location as 0,0
hor_plane = wfct.cut_plane.set_origin(hor_plane, 250., 200.)
ax = axarr[1, 0]
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
ax.set_title('Turbine at origin')

# Increase the resolution
hor_plane = wfct.cut_plane.change_resolution(
    hor_plane, resolution=(1000, 1000))
ax = axarr[2, 0]
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
ax.set_title('Increased Resolution (Interpolation)')

# # Interpolate onto new array
x1_array = np.linspace(-50, 300)
x2_array = np.linspace(-100, 100)
hor_plane = wfct.cut_plane.interpolate_onto_array(
    hor_plane, x1_array, x2_array)
ax = axarr[0, 1]
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
ax.set_title('Provided Grid')

# Express axis in terms of D
D = 126.  # m
hor_plane = wfct.cut_plane.rescale_axis(hor_plane, x1_factor=D, x2_factor=D)
ax = axarr[1, 1]
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
ax.set_title('Axis in D')

# Invert x1
ax = axarr[2, 1]
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
wfct.visualization.reverse_cut_plane_x_axis_in_plot(ax=ax)
ax.set_title('Invert x axis')

# Access and plot SOWFA outputs

# Local copy of output dataframe
df_out = sowfa_case.turbine_output

# Limit to first turbine (not needed but shown for example)
df_out = df_out[df_out.turbine == 0]

# Display available columns
print(df_out.columns)

fig, axarr = plt.subplots(2, 1, sharex=True)

ax = axarr[0]
ax.plot(df_out.time, df_out.powerGenerator/1E3)
ax.set_title('Generator Power (kW)')

ax = axarr[1]
ax.plot(df_out.time, df_out.rotSpeedFiltered)
ax.set_title('Rotor Speed (RPM)')
ax.set_xlabel('Time (s)')

plt.show()
