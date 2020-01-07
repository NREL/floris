# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# See read the https://floris.readthedocs.io for documentation

import matplotlib.pyplot as plt
import floris.tools as wfct
import numpy as np
# from floris.utilities import Vec3

# Initialize the FLORIS interface fi, use default gauss model
fi = wfct.floris_interface.FlorisInterface("example_input.json")
fi_gch = wfct.floris_interface.FlorisInterface("example_input.json")

# Force dm to 1.0
fi.floris.farm.wake._deflection_model.deflection_multiplier = 1.0
fi_gch.floris.farm.wake._deflection_model.deflection_multiplier = 1.0

# Set up gch
fi_gch.floris.farm.wake.velocity_model = "gauss_curl_hybrid"
fi_gch.floris.farm.wake.deflection_model = "gauss_curl_hybrid"
fi_gch.floris.farm.wake.velocity_models["gauss_curl_hybrid"].use_yar = False
fi_gch.floris.farm.wake.deflection_models["gauss_curl_hybrid"].use_ss = False

# Change the layout
D = fi.floris.farm.flow_field.turbine_map.turbines[0].rotor_diameter
layout_x = [0, 7*D, 14*D]
layout_y = [0, 0, 0]
yaw_angles = [0, 0, 0]
fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))
fi_gch.reinitialize_flow_field(layout_array=(layout_x, layout_y))

# Calculate baseline wake
fi.calculate_wake(yaw_angles=yaw_angles)
fi_gch.calculate_wake(yaw_angles=yaw_angles)

# Print the turbine power
print("Power in baseline")
print('Gauss',np.array(fi.get_turbine_power())/1000.0)
print('GCH',np.array(fi_gch.get_turbine_power())/1000.0)


# # Calculate wake
# fi.calculate_wake(yaw_angles=yaw_angles)

# # Print the turbine power
# print(np.array(fi.get_turbine_power())/1000.0)
