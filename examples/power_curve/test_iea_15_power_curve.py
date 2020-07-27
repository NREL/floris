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


import numpy as np

import floris.tools as wfct
import matplotlib.pyplot as plt


# Initialize the FLORIS interface fi
# For basic usage, the florice interface provides a simplified interface to
# the underlying classes
fi = wfct.floris_interface.FlorisInterface("iea_15.json")

# Get the power curve
wind_speeds = np.arange(3, 30, 0.5)
power_array_current = fi.get_power_curve(wind_speeds) / 1e6  # In MW
power_array_new = fi.get_power_curve(wind_speeds, use_new=True) / 1e6  # In MW

# Show the power curve
fig, ax = plt.subplots()
ax.plot(wind_speeds, power_array_current, "ko-", label="Current Version")
ax.plot(wind_speeds, power_array_new, "r.-", label="New Version")
ax.set_xlabel("Wind Speed (m/s)")
ax.set_ylabel("Power (MW)")
ax.legend()
plt.show()
