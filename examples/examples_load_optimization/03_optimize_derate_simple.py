"""Example: Simple optimization of derating

Look at optimizing for a two turbine farm with constant direction and varying TI and price

"""

import matplotlib.pyplot as plt
import numpy as np

from floris import FlorisModel, TimeSeries
from floris.optimization.load_optimization.load_optimization import (
    compute_farm_revenue,
    compute_farm_voc,
    compute_net_revenue,
    compute_turbine_voc,
    find_A_to_satisfy_rev_voc_ratio,
    optimize_derate,
)


# Compare VOC and net revenue under different conditions


# Parameters
D = 126.0
N = 100
derating_levels = np.linspace(1.0, 0.001, 20)


# Declare a floris model with default configuration
fmodel = FlorisModel(configuration="defaults")
fmodel.set_operation_model("simple-derating")


# Set up a two turbine farm
fmodel.set(layout_x=[0, D * 5], layout_y=[0.0, 0.0])


# Set up the conditions
wind_directions = np.ones(N) * 270.0
values = np.ones(N)
load_ambient_tis = np.linspace(0.05, 0.25, N)
time_series = TimeSeries(
    wind_directions=wind_directions, wind_speeds=8.0, turbulence_intensities=0.06, values=values
)

fmodel.set(wind_data=time_series)
fmodel.run()

# Set the initial power setpoints as no derating
initial_power_setpoint = np.ones((N, 2)) * 5e6

# Calculate the A which would put the farm at a 10x revenue to VOC ratio
A_initial = find_A_to_satisfy_rev_voc_ratio(fmodel, 4.0, load_ambient_tis)

# Set these initial power setpoints
fmodel.set(power_setpoints=initial_power_setpoint)
fmodel.run()
net_revenue_initial = compute_net_revenue(fmodel, A_initial, load_ambient_tis)
farm_voc_initial = compute_farm_voc(fmodel, A_initial, load_ambient_tis)
farm_revenue_initial = compute_farm_revenue(fmodel)


# Compute the optimal derating levels given A_initial
opt_power_setpoints = optimize_derate(
    fmodel,
    A_initial,
    load_ambient_tis,
    initial_power_setpoint=initial_power_setpoint,
    derating_levels=derating_levels,
)

# Compute final values
fmodel.set(power_setpoints=opt_power_setpoints)
fmodel.run()
net_revenue_opt = compute_net_revenue(fmodel, A_initial, load_ambient_tis)
farm_voc_opt = compute_farm_voc(fmodel, A_initial, load_ambient_tis)
farm_revenue_opt = compute_farm_revenue(fmodel)


# Show the results
fig, axarr = plt.subplots(6, 1, sharex=True, figsize=(10, 12))

# Plot the load TI
ax = axarr[0]
ax.plot(load_ambient_tis, marker="s", color="k")
ax.set_ylabel("Load Ambient TI")

# Plot the values
ax = axarr[1]
ax.plot(values, color="k")
ax.set_ylabel("Value of Electricity (-)")

# Plot the initial and final farm revenue
ax = axarr[2]
ax.plot(farm_revenue_initial, label="Initial", color="k")
ax.plot(farm_revenue_opt, label="Optimized", color="r")
ax.set_ylabel("Farm Revenue ($)")
ax.legend()

# Plot the initial and final farm VOC
ax = axarr[3]
ax.plot(farm_voc_initial, label="Initial", color="k")
ax.plot(farm_voc_opt, label="Optimized", color="r")
ax.set_ylabel("Farm VOC")
ax.legend()

# Plot the initial and final farm net revenue
ax = axarr[4]
ax.plot(net_revenue_initial, label="Initial", color="k")
ax.plot(net_revenue_opt, label="Optimized", color="r")
ax.set_ylabel("Farm Net Revenue ($)")
ax.legend()

# Plot the turbine deratings
ax = axarr[5]
for i in range(2):
    ax.plot(opt_power_setpoints[:, i] / 1000.0, label=f"Turbine {i}")
ax.set_ylabel("Power Setpoint (kW)")
ax.set_xlabel("Time Step")
ax.legend()

for ax in axarr:
    ax.grid(True)

plt.show()
