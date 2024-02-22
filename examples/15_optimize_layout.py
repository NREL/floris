
import os

import matplotlib.pyplot as plt
import numpy as np

from floris.tools import FlorisInterface, WindRose
from floris.tools.optimization.layout_optimization.layout_optimization_scipy import (
    LayoutOptimizationScipy,
)


"""
This example shows a simple layout optimization using the python module Scipy.

A 4 turbine array is optimized such that the layout of the turbine produces the
highest annual energy production (AEP) based on the given wind resource. The turbines
are constrained to a square boundary and a random wind resource is supplied. The results
of the optimization show that the turbines are pushed to the outer corners of the boundary,
which makes sense in order to maximize the energy production by minimizing wake interactions.
"""

# Initialize the FLORIS interface fi
file_dir = os.path.dirname(os.path.abspath(__file__))
fi = FlorisInterface('inputs/gch.yaml')

# Setup 72 wind directions with a 1 wind speed and frequency distribution
wind_directions = np.arange(0, 360.0, 5.0)
wind_speeds = np.array([8.0])

# Shape frequency distribution to match number of wind directions and wind speeds
freq_table = np.zeros((len(wind_directions), len(wind_speeds)))
np.random.seed(1)
freq_table[:,0] = (np.abs(np.sort(np.random.randn(len(wind_directions)))))
freq_table = freq_table / freq_table.sum()

# Establish a TimeSeries object
wind_rose = WindRose(wind_directions=wind_directions,
                     wind_speeds=wind_speeds,
                     freq_table=freq_table)

fi.reinitialize(wind_data=wind_rose)

# The boundaries for the turbines, specified as vertices
boundaries = [(0.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0), (1000.0, 0.0), (0.0, 0.0)]

# Set turbine locations to 4 turbines in a rectangle
D = 126.0 # rotor diameter for the NREL 5MW
layout_x = [0, 0, 6 * D, 6 * D]
layout_y = [0, 4 * D, 0, 4 * D]
fi.reinitialize(layout_x=layout_x, layout_y=layout_y)

# Setup the optimization problem
layout_opt = LayoutOptimizationScipy(fi, boundaries, wind_data=wind_rose)

# Run the optimization
sol = layout_opt.optimize()

# Get the resulting improvement in AEP
print('... calcuating improvement in AEP')
fi.calculate_wake()
base_aep = fi.get_farm_AEP_with_wind_data(wind_data=wind_rose) / 1e6
fi.reinitialize(layout_x=sol[0], layout_y=sol[1])
fi.calculate_wake()
opt_aep = fi.get_farm_AEP_with_wind_data(wind_data=wind_rose)  / 1e6
percent_gain = 100 * (opt_aep - base_aep) / base_aep

# Print and plot the results
print(f'Optimal layout: {sol}')
print(
    f'Optimal layout improves AEP by {percent_gain:.1f}% '
    f'from {base_aep:.1f} MWh to {opt_aep:.1f} MWh'
)
layout_opt.plot_layout_opt_results()

plt.show()
