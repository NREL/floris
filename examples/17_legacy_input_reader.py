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


import numpy as np
from floris.tools import FlorisInterfaceLegacyV2

"""
This example creates a FLORIS instance based on a v2.4 legacy input file.
"""

# Initialize FLORIS with a legacy input file for FLORIS v2.4 via
# 'FlorisInterfaceLegacyV2'.
fi = FlorisInterfaceLegacyV2("inputs/legacy_example_input.json")

# Example usage: calculate wind farm power for a three-turbine case
fi.reinitialize(
    layout=[[0.0, 600.0, 1200.0], [0.0, 0.0, 0.0]],
    wind_directions=np.array([260., 270., 280.]),
    wind_speeds=np.array([8.0, 9.0, 10.0]),
)
fi.calculate_wake()
turbine_powers = fi.get_turbine_powers() / 1000.
print(turbine_powers)
