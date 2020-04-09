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
import copy

def power_cross_sweep(fi_in,yaw_angle=0):
    fi = copy.deepcopy(fi_in)

    D = 126.

    sweep_locations = np.arange(-2,2.25,0.25)
    sweep_locations = np.arange(-1,1.25,0.25)
    power_out = np.zeros_like(sweep_locations)

    for y_idx, y_loc in enumerate(sweep_locations):

        fi.reinitialize_flow_field(
            layout_array=(
                [0,10*126,20*126],
                [0,0,y_loc*D]
            )
        )
        fi.calculate_wake([yaw_angle,0,0])
        power_out[y_idx] = fi.get_turbine_power()[2]/1000.

    return sweep_locations, power_out

# Initialize the FLORIS interface fi
# For basic usage, the florice interface provides a simplified interface to
# the underlying classes
fi = wfct.floris_interface.FlorisInterface("example_input.json")

fi.reinitialize_flow_field(layout_array=[[0,10*126,20*126],[0,0,0]])

fi_curl = wfct.floris_interface.FlorisInterface("example_input.json")
fi_curl.floris.farm.set_wake_model('curl')
fi_curl.reinitialize_flow_field(layout_array=[[0,10*126,20*126],[0,0,0]])

x1,y1 = power_cross_sweep(fi)
fi.floris.farm.wake.velocity_model.wake_rotation = False
x2,y2 = power_cross_sweep(fi)
fi.floris.farm.wake.velocity_model.wake_rotation = True
fi.floris.farm.wake.velocity_model.gamma_scale = 1.5
fi.floris.farm.wake.velocity_model.gamma_rotation_scale = 0.1
x3,y3 = power_cross_sweep(fi)
# x2,y2 = power_cross_sweep(fi_curl)

fig, ax = plt.subplots()
ax.plot(x1,y1,label='GCH')
ax.plot(x2,y2,label='GHC - No wake rotation')
ax.plot(x3,y3,label='GHC - Scaled')
ax.legend()
ax.axvline(0)

fi.floris.farm.wake.velocity_model.wake_rotation = True
fi.floris.farm.wake.velocity_model.gamma_scale = 1.0
fi.floris.farm.wake.velocity_model.gamma_rotation_scale = 0.5
x4,y4 = power_cross_sweep(fi, yaw_angle=25)
fi.floris.farm.wake.velocity_model.wake_rotation = False
x5,y5 = power_cross_sweep(fi, yaw_angle=25)
fi.floris.farm.wake.velocity_model.wake_rotation = True
fi.floris.farm.wake.velocity_model.gamma_scale = 1.5
fi.floris.farm.wake.velocity_model.gamma_rotation_scale = 0.1
x6,y6 = power_cross_sweep(fi, yaw_angle=25)

fig, ax = plt.subplots()
ax.plot(x4,y4,label='GCH')
ax.plot(x5,y5,label='GHC - No wake rotation')
ax.plot(x6,y6,label='GHC - Scaled')
ax.legend()
ax.axvline(0)

plt.show()
