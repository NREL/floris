"""Example: Load turbulence intensity

Demonstrate the behavior of the load turbulence intensity model.

"""

import matplotlib.pyplot as plt
import numpy as np

from floris import FlorisModel, TimeSeries
from floris.optimization.load_optimization.load_optimization import compute_load_ti


# Parameters
D = 126.0


# Declare a floris model with default configuration
fmodel = FlorisModel(configuration="defaults")

# WIND DIRECTION SWEEP
wind_directions = np.arange(0, 360, 1.0)

# Assume uniform load ambient turbulence intensities
load_ambient_tis = 0.1 * np.ones_like(wind_directions)

# Declare a time series
time_series = TimeSeries(
    wind_directions=wind_directions, wind_speeds=8.0, turbulence_intensities=0.06
)

# Set the turbine layout to be a simple two turbine layout using the
# time series object
fmodel.set(layout_x=[0, D * 7], layout_y=[0.0, 0.0], wind_data=time_series)
fmodel.run()

# Compute the load turbulence intensity
load_ti = compute_load_ti(fmodel, load_ambient_tis)

# Print the TI for each turbine
fig, ax = plt.subplots()
for t in range(fmodel.n_turbines):
    ax.plot(wind_directions, load_ti[:, t], label=f"Turbine {t}")
ax.set_ylabel("Load TI")
ax.set_xlabel("Wind Direction (deg)")
ax.legend()

# CT SWEEP
fmodel = FlorisModel(configuration="defaults")
N = 50
wind_directions = np.ones(N) * 270.0
load_ambient_tis = np.ones(N) * 0.1
power_setpoints = np.linspace(5e6, 1, N)
power_setpoints_grid = np.column_stack([power_setpoints, power_setpoints * 0 + 1])
time_series = TimeSeries(
    wind_directions=wind_directions, wind_speeds=8.0, turbulence_intensities=0.06
)
fmodel.set_operation_model("mixed")
fmodel.set(
    layout_x=[0, D * 7],
    layout_y=[0.0, 0.0],
    wind_data=time_series,
    power_setpoints=power_setpoints_grid,
)
fmodel.run()

# Compute the load turbulence intensity
load_ti = compute_load_ti(fmodel, load_ambient_tis)

# Print the TI for each turbine
fig, ax = plt.subplots()
for t in range(fmodel.n_turbines):
    ax.plot(power_setpoints, load_ti[:, t], label=f"Turbine {t}")
ax.set_ylabel("Load TI")
ax.set_xlabel("Power Setpoint (T0)")
ax.legend()


plt.show()
