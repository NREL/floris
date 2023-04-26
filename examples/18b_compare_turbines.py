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

"""
Plots the primary turbine characteristics for each of the turbines in FLORIS `turbine_library`.
Additionally, this demonstrates how a user can further interact with turbine configuration files
in both the internal turbine library and a user provided turbine library.
"""

import matplotlib.pyplot as plt
import numpy as np

from floris.turbine_library import TurbineLibrary


ws_array = np.linspace(0, 30, 61)

# Create the turbine library interface
tl = TurbineLibrary()
tl.load_internal_library()
# tl.load_external_library("my_path")  # external libraries can also be loaded


# Plot all values together in a single window
tl.plot_comparison(wind_speed=ws_array, plot_kwargs={"linewidth": 1})

# Each of the subpllots can also be produced individually
# tl.plot_power_curves(wind_speed=ws_array, show=True)
# tl.plot_Cp_curves(wind_speed=ws_array, show=True)
# tl.plot_Ct_curves(wind_speed=ws_array, show=True)
# tl.plot_rotor_diameters(show=True)
# tl.plot_hub_heights(show=True)

plt.show()
