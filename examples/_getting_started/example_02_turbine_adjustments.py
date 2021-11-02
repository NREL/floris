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


# This example illustrates changing the properties of some of the turbines
# This can be used to setup farms of different turbines

import matplotlib.pyplot as plt

from src.tools.floris_interface import FlorisInterface
from src.tools.visualization import visualize_cut_plane


# Initialize the FLORIS interface fi
fi = FlorisInterface("../example_input.json")

# Set to 2x2 farm
fi.floris.farm.layout_x = [0, 0, 600, 600]
fi.floris.farm.layout_y = [0, 300, 0, 300]

# Change turbine 0 and 3 to have a 35 m rotor diameter
fi.floris.farm.rotor_diameter[0] = 35
fi.floris.farm.rotor_diameter[3] = 35

# Calculate wake
fi.calculate_wake()

# Get horizontal plane at default height (hub-height)
hor_plane = fi.get_hor_plane()

# Plot and show
fig, ax = plt.subplots()
visualize_cut_plane(hor_plane, ax=ax)
plt.show()
