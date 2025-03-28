"""Example: Variable Operating Cost

Demonstrate behavior of the Variable Operating Cost (VOC)

"""

import matplotlib.pyplot as plt
import numpy as np

from floris import FlorisModel, TimeSeries
from floris.optimization.load_optimization.load_optimization import (
    compute_net_revenue,
    compute_turbine_voc,
)


# Compare VOC and net revenue under different conditions


# Parameters
D = 126.0
N = 100


# Declare a floris model with default configuration
fmodel = FlorisModel(configuration="defaults")

# Sweep TI first==================================================================

wind_directions = np.ones(N) * 270.0

# Assume uniform load ambient turbulence intensities
load_ambient_tis = np.linspace(0.05, 0.25, N)

# Assume uniform values
values = 2e-5*np.ones(N) # $20/MWh

# Declare a time series
time_series = TimeSeries(
    wind_directions=wind_directions, wind_speeds=8.0, turbulence_intensities=0.06, values=values
)

# Set the turbine layout to be a simple two turbine layout using the
# time series object
fmodel.set(layout_x=[0, D * 7], layout_y=[0.0, 0.0], wind_data=time_series)
fmodel.run()

# Compute the load turbulence intensity
voc = compute_turbine_voc(fmodel, 2e-5, load_ambient_tis)

# Compute net revenue
net_revenue = compute_net_revenue(fmodel, 2e-5, load_ambient_tis)

# Plot the VOC vs. load TI for each turbine
fig, axarr = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)

ax = axarr[0]
ax.plot(load_ambient_tis, voc[:, 0], label="Turbine 0")
ax.plot(load_ambient_tis, voc[:, 1], label="Turbine 1")
ax.set_ylabel("VOC ($)")
ax.set_xlabel("Load Ambient TI (-)")
ax.legend()
ax.grid(True)
ax = axarr[1]
ax.plot(load_ambient_tis, net_revenue, label="Farm Net Revenue", color="k")
ax.set_ylabel("Net Revenue ($)")
ax.set_xlabel("Load Ambient TI (-)")
ax.grid(True)
fig.suptitle(
    "X, Y Turbine Coordinates: T0: (0, 0), T1: (7D, 0); Wind Direction = 270\u00B0; "
    "Value of Electricity = $20/MWh"
)


# Sweep values next==================================================================

wind_directions = np.ones(N) * 270.0

# Assume uniform load ambient turbulence intensities
load_ambient_tis = np.ones(N) * 0.1

# Assume uniform values
values = np.linspace(1e-5, 1e-4, N)

# Declare a time series
time_series = TimeSeries(
    wind_directions=wind_directions, wind_speeds=8.0, turbulence_intensities=0.06, values=values
)

# Set the turbine layout to be a simple two turbine layout using the
# time series object
fmodel.set(layout_x=[0, D * 7], layout_y=[0.0, 0.0], wind_data=time_series)
fmodel.run()

# Compute the load turbulence intensity
voc = compute_turbine_voc(fmodel, 2e-5, load_ambient_tis)

# Compute net revenue
net_revenue = compute_net_revenue(fmodel, 2e-5, load_ambient_tis)

# Plot the VOC vs. value for each turbine
fig, axarr = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)

ax = axarr[0]
ax.plot(1e6*values, voc[:, 0], label="Turbine 0")
ax.plot(1e6*values, voc[:, 1], label="Turbine 1")
ax.set_ylabel("VOC ($)")
ax.set_xlabel("Value of Electricity ($/MWh)")
ax.legend()
ax.grid(True)
ax = axarr[1]
ax.plot(1e6*values, net_revenue, label="Farm Net Revenue", color="k")
ax.set_ylabel("Net Revenue ($)")
ax.set_xlabel("Value of Electricity ($/MWh)")
ax.grid(True)
fig.suptitle("X, Y Turbine Coordinates: T0: (0, 0), T1: (7D, 0); Wind Direction = 270\u00B0")


plt.show()
