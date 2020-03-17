# Copyright 2019 NREL

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
import numpy as np

# Initialize the FLORIS interface fi
fi = wfct.floris_interface.FlorisInterface("../example_input.json")

# Calculate wake
fi.calculate_wake()

# Show the powers
init_power = fi.get_turbine_power()[0]/1000.
print(init_power)

# Now sweep the heights and see what happens
heights = np.arange(100,5,-1.)
powers = np.zeros_like(heights)

for h_idx, h in enumerate(heights):
    fi.change_turbine([0],{'hub_height':h})
    fi.calculate_wake()
    powers[h_idx] = fi.get_turbine_power()[0]/1000.


fig, ax = plt.subplots()
ax.plot(heights, powers, 'k')
ax.axhline(init_power,color='r')
ax.axvline(90,color='r')
plt.show()
