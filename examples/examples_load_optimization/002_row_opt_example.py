"""Example: Optimize a row of turbines

This example optimizes the derating of a row of three turbines to maximize net revenue for a
variety of combinations of wind direction, load ambient TI, and electricity values.
The row is aligned when the wind direction is 270 degrees.

"""

import matplotlib.pyplot as plt
import numpy as np

from floris import FlorisModel, TimeSeries
from floris.optimization.load_optimization.load_optimization import (
    compute_farm_revenue,
    compute_farm_voc,
    compute_net_revenue,
    find_A_to_satisfy_rev_voc_ratio,
    optimize_power_setpoints,
)


# Parameters
D = 126.0
d_spacing = 7.0
MIN_POWER_SETPOINT = 0.00000001
derating_levels = np.linspace(1.0, MIN_POWER_SETPOINT, 10)
n_turbines = 3
N_per_loop = 10 # Number of unique values for wind direction, value, and load ambient TI


# Declare a floris model with default configuration
fmodel = FlorisModel(configuration="defaults")
fmodel.set_operation_model("simple-derating")


# Set up a row of turbines
fmodel.set(
    layout_x=[i * D * d_spacing for i in range(n_turbines)],
    layout_y=[0.0 for i in range(n_turbines)],
)


# Set up input conditions
wind_directions = []
values = []
ambient_lti = []

for w_i in np.linspace(230, 270, N_per_loop):
    for v_i in np.linspace(1e-5, 5e-5, N_per_loop):
        for t_i in np.linspace(0.02, 0.2, N_per_loop):
            wind_directions.append(w_i)
            values.append(v_i)
            ambient_lti.append(t_i)
wind_directions = np.array(wind_directions)
values = np.array(values)
ambient_lti = np.array(ambient_lti)
N = len(wind_directions)

time_series = TimeSeries(
    wind_directions=wind_directions, wind_speeds=8.0, turbulence_intensities=0.06, values=values
)

# Run the FLORIS model
fmodel.set(wind_data=time_series)
fmodel.run()

# Set the initial power setpoints as no derating
initial_power_setpoint = np.ones((N, n_turbines)) * 5e6

# Calculate the A which would put the farm at a 4 revenue to VOC ratio
A_initial = find_A_to_satisfy_rev_voc_ratio(fmodel, 4.0, ambient_lti)

# Set these initial power setpoints
fmodel.set(power_setpoints=initial_power_setpoint)
fmodel.run()
net_revenue_initial = compute_net_revenue(fmodel, A_initial, ambient_lti)
farm_voc_initial = compute_farm_voc(fmodel, A_initial, ambient_lti)
farm_revenue_initial = compute_farm_revenue(fmodel)


# Compute the optimal derating levels given A_initial
opt_power_setpoints, opt_net_revenue = optimize_power_setpoints(
    fmodel,
    A_initial,
    ambient_lti,
    power_setpoint_initial=initial_power_setpoint,
    derating_levels=derating_levels,
)

# Compute final values
fmodel.set(power_setpoints=opt_power_setpoints)
fmodel.run()
net_revenue_opt = compute_net_revenue(fmodel, A_initial, ambient_lti)
farm_voc_opt = compute_farm_voc(fmodel, A_initial, ambient_lti)
farm_revenue_opt = compute_farm_revenue(fmodel)


# Show the results
fig, axarr = plt.subplots(7, 1, sharex=True, figsize=(10, 10))

# Plot the wind direction
ax = axarr[0]
ax.plot(wind_directions, color="k")
ax.set_ylabel("Wind\n Direction (deg)")
ax.set_title("X, Y Turbine Coordinates: T0: (0, 0), T1: (7D, 0), T2: (14D, 0); Wind Speed = 8 m/s")

# Plot the load TI
ax = axarr[1]
ax.plot(ambient_lti, color="k")
ax.set_ylabel("Load Ambient\n TI (-)")

# Plot the values
ax = axarr[2]
ax.plot(1e6*values, color="k")
ax.set_ylabel("Value of\n Electricity ($/MWh)")

# Plot the initial and final farm revenue
ax = axarr[3]
ax.plot(farm_revenue_initial, label="Initial", color="k")
ax.plot(farm_revenue_opt, label="Optimized", color="r")
ax.set_ylabel("Farm\n Revenue ($)")
ax.legend()

# Plot the initial and final farm VOC
ax = axarr[4]
ax.plot(farm_voc_initial, label="Initial", color="k")
ax.plot(farm_voc_opt, label="Optimized", color="r")
ax.set_ylabel("Farm VOC ($)")
ax.legend()

# Plot the initial and final farm net revenue
ax = axarr[5]
ax.plot(net_revenue_initial, label="Initial", color="k")
ax.plot(net_revenue_opt, label="Optimized", color="r")
ax.set_ylabel("Farm Net\nRevenue ($)")
ax.legend()

# Plot the turbine deratings
ax = axarr[6]
for i in range(n_turbines):
    ax.plot(opt_power_setpoints[:, i] / 1000.0, label=f"Turbine {i}", lw=3 * n_turbines / (i + 1))
ax.set_ylabel("Power\n Setpoint (kW)")
ax.set_xlabel("Wind Condition and Electricity Value Combination")
ax.legend()

for ax in axarr:
    ax.grid(True)

plt.show()
