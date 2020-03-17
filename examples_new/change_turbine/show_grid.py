# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# See read the https://floris.readthedocs.io for documentation

# Show the grid points in hetergenous flow calculation

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import floris.tools as wfct
import pandas as pd
import numpy as np

fi = wfct.floris_interface.FlorisInterface("../example_input.json")
fi.reinitialize_flow_field(layout_array=[[0,500],[0,0]])
fi.calculate_wake()

# Show the grid points (note only on turbines, not on wind measurements)
fig = plt.figure(figsize=(15,5))
ax = fig.add_subplot(131, projection='3d')
xs = fi.floris.farm.flow_field.x
ys = fi.floris.farm.flow_field.y
zs = fi.floris.farm.flow_field.z
ax.scatter(xs, ys, zs, marker='.')
ax.set_xlim([0,600])
ax.set_ylim([-300,300])
ax.set_zlim([0,200])
ax.set_title('Initial')


# Raise the hub height of the second turbine and show
fi.change_turbine([1],{'hub_height':150})
fi.reinitialize_flow_field()
fi.calculate_wake()
ax = fig.add_subplot(132, projection='3d')
xs = fi.floris.farm.flow_field.x
ys = fi.floris.farm.flow_field.y
zs = fi.floris.farm.flow_field.z
ax.scatter(xs, ys, zs, marker='.')
ax.set_xlim([0,600])
ax.set_ylim([-300,300])
ax.set_zlim([0,200])
ax.set_title('Raise Second HH')

# Increase the first turbine rotor_diameter and plot 
fi.change_turbine([0],{'rotor_diameter':250})
fi.reinitialize_flow_field()
fi.calculate_wake()
ax = fig.add_subplot(133, projection='3d')
xs = fi.floris.farm.flow_field.x
ys = fi.floris.farm.flow_field.y
zs = fi.floris.farm.flow_field.z
ax.scatter(xs, ys, zs, marker='.')
ax.set_xlim([0,600])
ax.set_ylim([-300,300])
ax.set_zlim([0,200])
ax.set_title('Increase first diameter')


plt.show()