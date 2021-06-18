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


import matplotlib.pyplot as plt

import floris.tools as wfct


# Initialize the FLORIS interface fi
# For basic usage, the florice interface provides a simplified interface to
# the underlying classes
fi = wfct.floris_interface.FlorisInterface("../example_input.json")

layout_x = [0, 630]
layout_y = [0, 0]

low_ti = 0.06
high_ti = 0.09

# Calculate wake for low turbulence
fi.reinitialize_flow_field(layout_array=(layout_x,layout_y),turbulence_intensity=low_ti)
fi.calculate_wake(yaw_angles=[0,0])
power_base_low_ti = fi.get_farm_power()

fi.reinitialize_flow_field(layout_array=(layout_x,layout_y),turbulence_intensity=low_ti)
fi.calculate_wake(yaw_angles=[25,0])
power_yaw_low_ti = fi.get_farm_power()

print('Power Gain Low TI = ', 100*(power_yaw_low_ti - power_base_low_ti) / power_base_low_ti)

# Calculate wake for low turbulence
fi.reinitialize_flow_field(layout_array=(layout_x,layout_y),turbulence_intensity=high_ti)
fi.calculate_wake(yaw_angles=[0,0])
power_base_high_ti = fi.get_farm_power()

fi.reinitialize_flow_field(layout_array=(layout_x,layout_y),turbulence_intensity=high_ti)
fi.calculate_wake(yaw_angles=[25,0])
power_yaw_high_ti = fi.get_farm_power()

print('Power Gain High TI = ', 100*(power_yaw_high_ti - power_base_high_ti) / power_base_high_ti)


