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
import numpy as np

from floris.tools import FlorisInterface

"""
This example creates a FLORIS instance
1) Makes a two-turbine layout
2) Demonstrates single ws/wd simulations
3) Demonstrates mulitple ws/wd simulations

Main concept is introduce FLORIS and illustrate essential structure of most-used FLORIS calls
"""

# Parameter
ws = 8.
num_t_per_row = 4

# Make a plot
fig, axarr = plt.subplots(1,3,figsize=(15,5))

for wd_idx, wd in enumerate([270, 275, 280]):

    ax = axarr[wd_idx]

    # Initialize FLORIS with the given input file via FlorisInterface.
    # For basic usage, FlorisInterface provides a simplified and expressive
    # entry point to the simulation routines.
    fi_gch = FlorisInterface("inputs/gch.yaml")
    fi_cc = FlorisInterface("inputs/cc.yaml")
    fi_turbopark = FlorisInterface("inputs/turbopark.yaml")

    # Make a box layout
    X = []
    Y = []

    for x_idx in range(num_t_per_row):
        for y_idx in range(num_t_per_row):
            X.append(126 * 6 * x_idx)
            Y.append(126 * 6 * y_idx)

    # Convert to a simple two turbine layout
    fi_gch.reinitialize( layout=( X,Y), wind_directions=[wd], wind_speeds=[ws] )
    fi_cc.reinitialize( layout=( X,Y), wind_directions=[wd], wind_speeds=[ws] )
    fi_turbopark.reinitialize( layout=( X,Y), wind_directions=[wd], wind_speeds=[ws] )

    # Perform wake calculations
    fi_gch.calculate_wake()
    fi_cc.calculate_wake()
    fi_turbopark.calculate_wake()

    # Get the turbine powers
    turbine_powers_gch = fi_gch.get_turbine_powers()/1000.
    turbine_powers_cc = fi_cc.get_turbine_powers()/1000.
    turbine_powers_turbopark = fi_turbopark.get_turbine_powers()/1000.


    ax.plot(turbine_powers_gch.flatten(),label='GCH')
    ax.plot(turbine_powers_cc.flatten(),label='CC')
    ax.plot(turbine_powers_turbopark.flatten(),label='TurbOPark')
    ax.set_xlabel('Turbine')
    ax.set_ylabel('Power (kW)')

    ax.grid(True)
    ax.legend()
    ax.set_title('Wind Direction = %.1f' % wd)

plt.show()