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
 

## THIS IS A WORK IN PROGRESS

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import floris.tools as wfct

# Initialize the FLORIS interface fi
fi = wfct.floris_interface.FlorisInterface('../example_input.json')

# Just one turbine
fi.reinitialize_flow_field(layout_array=([100],[100]))

# Calculate wake
fi.calculate_wake()

# Show thoe points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs = fi.floris.farm.flow_field.x
ys = fi.floris.farm.flow_field.y
zs = fi.floris.farm.flow_field.z
ax.scatter(xs, ys, zs, marker='.')

# # print('yaw')
# # fi.calculate_wake(yaw_angles=[30])
# print('wd')
# fi.reinitialize_flow_field(wind_direction=280)
# fi.calculate_wake()






# n = 100

# # For each set of style and range settings, plot n random points in the box
# # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
# for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
#     xs = randrange(n, 23, 32)
#     ys = randrange(n, 0, 100)
#     zs = randrange(n, zlow, zhigh)
#     ax.scatter(xs, ys, zs, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
