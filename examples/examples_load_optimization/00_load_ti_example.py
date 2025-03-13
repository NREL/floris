
"""Example: Load turbulence intensity
Description TBD.
"""

import numpy as np

from floris import FlorisModel, TimeSeries
from floris.optimization.load_optimization.load_optimization import compute_load_ti


D = 126.0
wind_directions = np.arange(0, 360, 30.0)

# Declare a floris model with default configuration
fmodel = FlorisModel(configuration="defaults")

# Declare a time series
time_series = TimeSeries(
    wind_directions=wind_directions,
    wind_speeds=8.0,
    turbulence_intensities=0.06
)

# Set the turbine layout to be a simple two turbine layout using the
# time series object
fmodel.set(layout_x=[0, D * 7], layout_y=[0.0, 0.0], wind_data=time_series)
fmodel.run()

# Compute the load turbulence intensity
load_ti = compute_load_ti(fmodel, 0.10)
