# Copyright 2022 NREL

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
import numpy as np

from floris.tools import FlorisInterface


"""
04_sweep_wind_directions

This example demonstrates vectorization of wind direction.
A vector of wind directions is passed to the intialize function
and the powers of the two simulated turbines is computed for all
wind directions in one call

The power of both turbines for each wind direction is then plotted

"""

# Instantiate FLORIS using either the GCH or CC model
fi = FlorisInterface("inputs/gch.yaml") # GCH model matched to the default "legacy_gauss" of V2

# Option one:
ws_array = np.array([6.0, 7.0, 8.0, 9.0])
ti_array = np.array([0.10, 0.09, 0.08, 0.07])
farm_powers = []
for wsii, ws in enumerate(ws_array):
    fi.reinitialize(
        wind_directions=[270.0],
        wind_speeds=[ws],
        turbulence_intensity=ti_array[wsii]
    )
    fi.calculate_wake()
    farm_powers.append(float(fi.get_farm_power()))
print(f"Farm powers (old method): {np.round(farm_powers)}")

# Option two: as a table
fi.reinitialize(
    wind_directions=[270.0],
    wind_speeds=ws_array,
    turbulence_intensity=ti_array[None, :],
)
fi.calculate_wake()
farm_powers = fi.get_farm_power().flatten()
print(f"Farm powers (new method): {np.round(farm_powers)}")
