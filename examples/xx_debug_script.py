# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See read the https://floris.readthedocs.io for documentation

import matplotlib.pyplot as plt
import floris.tools as wfct
import floris.tools.cut_plane as cp
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Initialize the FLORIS interface fi
# For basic usage, the florice interface provides a simplified interface to
# the underlying classes
fi = wfct.floris_interface.FlorisInterface("example_input.json")

fi.reinitialize_flow_field(layout_array=[[0,10*126,20*126],[0,0,0]])

# Calculate wake
print('\n----- Turbine resolution -----')
fi.calculate_wake()
print('\n----- Turbine resolution -----')
fi.calculate_wake()
print('\n----- Turbine resolution -----')
fi.calculate_wake()
# fi.calculate_wake()
# fi.calculate_wake()
# print('--')
# fi.calculate_wake(yaw_angles=[0,0,0])

print('\n----- x/y resolution = 50 -----')
print('----- fi.get_hor_plane(x_resolution=50, y_resolution=50) -----')
tmp1 = fi.get_hor_plane(x_resolution=50, y_resolution=50)
print('\n----- x/y resolution = 100 -----')
print('----- fi.get_hor_plane(x_resolution=100, y_resolution=100) -----')
tmp2 = fi.get_hor_plane(x_resolution=100, y_resolution=100)
print('\n----- x/y resolution = 150 -----')
print('----- fi.get_hor_plane(x_resolution=150, y_resolution=150) -----')
tmp3 = fi.get_hor_plane(x_resolution=150, y_resolution=150)

# Get horizontal plane at default height (hub-height)
print('\n----- x/y resolution = 200 -----')
print('----- fi.get_hor_plane() ----- # 200 is the default')
hor_plane = fi.get_hor_plane()

fi_curl = wfct.floris_interface.FlorisInterface("example_input.json")
fi_curl.floris.farm.set_wake_model('curl')
fi_curl.reinitialize_flow_field(layout_array=[[0,10*126,20*126],[0,0,0]])
hor_plane_curl = fi_curl.get_hor_plane()

hor_plane_project = cp.project_onto(hor_plane_curl, hor_plane)

# hor_plane.df['u'] = hor_plane.df['u']/8
# hor_plane_project.df['u'] = hor_plane_project.df['u']/8

gaussMin = np.min(hor_plane.df['u'])
curlMin = np.min(hor_plane_project.df['u'])
gaussMax = np.max(hor_plane.df['u'])
curlMax = np.max(hor_plane_project.df['u'])

curlSpan = curlMax - curlMin
gaussSpan = gaussMax - gaussMin

# Convert the left range into a 0-1 range (float)
# hor_plane.df['u'] = [float(value - gaussMin) / float(gaussMax) for value in hor_plane.df['u']]
# hor_plane.df['u'] = hor_plane.df['u']/gaussMax
hor_plane_project.df['u'] = [
    (float(value - curlMin) * float(gaussSpan) / curlSpan) + gaussMin for value in hor_plane_project.df['u']
]
# hor_plane_project.df['u'] = hor_plane_project.df['u']/gaussMax
# valueScaled = float(value - leftMin) / float(leftSpan)

print('gauss min: ', np.min(hor_plane.df['u']))
print('curl min: ', np.min(hor_plane_curl.df['u']))
print('project min: ', np.min(hor_plane_project.df['u']))
print('gauss max: ', np.max(hor_plane.df['u']))
print('curl max: ', np.max(hor_plane_curl.df['u']))
print('project max: ', np.max(hor_plane_project.df['u']))

diff_plane = cp.subtract(hor_plane_project, hor_plane)
diff_plane.df['u'] = diff_plane.df['u']/gaussMax

print('diff min: ', np.min(diff_plane.df['u']))
print('diff max: ', np.max(diff_plane.df['u']))

# Plot and show
fig, axarr = plt.subplots(3, 1, figsize=(15,5))
ax0 = axarr[0]
ax1 = axarr[1]
ax2 = axarr[2]

levels = [2, 3, 4, 5, 6, 7, 8]

im = wfct.visualization.visualize_cut_plane(hor_plane, ax=ax0, levels=levels)
divider = make_axes_locatable(ax0)
cax0 = divider.append_axes("right", size="2.5%", pad=0.1)
ax0.set_title('Gauss')
fig.colorbar(im, cax=cax0)

im = wfct.visualization.visualize_cut_plane(hor_plane_project, ax=ax1, levels=levels)
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("right", size="2.5%", pad=0.1)
ax1.set_title('Curl Projected onto Gauss')
fig.colorbar(im, cax=cax1)

levels = [-.5, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .4, .5]

im = wfct.visualization.visualize_cut_plane(diff_plane, ax=ax2, minSpeed=-0.5, maxSpeed=0.5)
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("right", size="2.5%", pad=0.1)
ax2.set_title('((Curl Projected - Mapped onto Gauss Range) - (Gauss)) / GaussMax')
fig.colorbar(im, cax=cax2)
plt.show()
