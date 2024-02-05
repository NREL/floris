# Copyright 2024 NREL

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
import yaml

from floris.tools import FlorisInterface


"""
Example to test out derating of turbines
"""

# Grab model of FLORIS using de-rate style NREL 5MW turbines
fi = FlorisInterface("inputs/gch.yaml")

with open(str(
    fi.floris.as_dict()["farm"]["turbine_library_path"] /
    (fi.floris.as_dict()["farm"]["turbine_type"][0] + ".yaml")
)) as t:
    turbine_type = yaml.safe_load(t)
turbine_type["power_thrust_model"] = "simple-derating"

fi.reinitialize(turbine_type=[turbine_type])

# Convert to a simple two turbine layout
fi.reinitialize(layout_x=[0, 500.0], layout_y=[0.0, 0.0])

# Set the wind directions and speeds to be constant over n_findex = N time steps
N = 50
fi.reinitialize(wind_directions=270 * np.ones(N), wind_speeds=8.0 * np.ones(N))
fi.calculate_wake()
