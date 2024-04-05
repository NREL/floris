"""Example: Layout optimization with heterogeneous inflow
This example shows a layout optimization using the geometric yaw option. It
combines elements of layout optimization and heterogeneous
inflow for demonstrative purposes.

Heterogeneity in the inflow provides the necessary driver for coupled yaw
and layout optimization to be worthwhile. First, a layout optimization is
run without coupled yaw optimization; then a coupled optimization is run to
show the benefits of coupled optimization when flows are heterogeneous.
"""


import os

import matplotlib.pyplot as plt
import numpy as np

from floris import FlorisModel, WindRose
from floris.optimization.layout_optimization.layout_optimization_scipy import (
    LayoutOptimizationScipy,
)


# Initialize FLORIS
file_dir = os.path.dirname(os.path.abspath(__file__))
fmodel = FlorisModel("../inputs/gch.yaml")

# Setup 2 wind directions (due east and due west)
# and 1 wind speed with uniform probability
wind_directions = np.array([270.0, 90.0])
n_wds = len(wind_directions)
wind_speeds = [8.0] * np.ones_like(wind_directions)
turbulence_intensities = 0.06 * np.ones_like(wind_directions)
# Shape frequency distribution to match number of wind directions and wind speeds
freq_table = np.ones((len(wind_directions), len(wind_speeds)))
freq_table = freq_table / freq_table.sum()


# The boundaries for the turbines, specified as vertices
D = 126.0  # rotor diameter for the NREL 5MW
size_D = 12
boundaries = [(0.0, 0.0), (size_D * D, 0.0), (size_D * D, 0.1), (0.0, 0.1), (0.0, 0.0)]

# Set turbine locations to 4 turbines at corners of the rectangle
# (optimal without flow heterogeneity)
layout_x = [0.1, 0.3 * size_D * D, 0.6 * size_D * D]
layout_y = [0, 0, 0]

# Generate exaggerated heterogeneous inflow (same for all wind directions)
speed_multipliers = np.repeat(np.array([0.5, 1.0, 0.5, 1.0])[None, :], n_wds, axis=0)
x_locs = [0, size_D * D, 0, size_D * D]
y_locs = [-D, -D, D, D]

# Create the configuration dictionary to be used for the heterogeneous inflow.
heterogeneous_inflow_config_by_wd = {
    "speed_multipliers": speed_multipliers,
    "wind_directions": wind_directions,
    "x": x_locs,
    "y": y_locs,
}

# Establish a WindRose object
wind_rose = WindRose(
    wind_directions=wind_directions,
    wind_speeds=wind_speeds,
    freq_table=freq_table,
    ti_table=0.06,
    heterogeneous_inflow_config_by_wd=heterogeneous_inflow_config_by_wd,
)


fmodel.set(
    layout_x=layout_x,
    layout_y=layout_y,
    wind_data=wind_rose,
)

# Setup and solve the layout optimization problem without heterogeneity
maxiter = 100
layout_opt = LayoutOptimizationScipy(
    fmodel, boundaries, min_dist=2 * D, optOptions={"maxiter": maxiter}
)

# Run the optimization
np.random.seed(0)
sol = layout_opt.optimize()

# Get the resulting improvement in AEP
print("... calcuating improvement in AEP")

fmodel.run()
base_aep = fmodel.get_farm_AEP() / 1e6
fmodel.set(layout_x=sol[0], layout_y=sol[1])
fmodel.run()
opt_aep = fmodel.get_farm_AEP() / 1e6

percent_gain = 100 * (opt_aep - base_aep) / base_aep

# Print and plot the results
print(f"Optimal layout: {sol}")
print(
    f"Optimal layout improves AEP by {percent_gain:.1f}% "
    f"from {base_aep:.1f} MWh to {opt_aep:.1f} MWh"
)
layout_opt.plot_layout_opt_results()
ax = plt.gca()
fig = plt.gcf()
sm = ax.tricontourf(x_locs, y_locs, speed_multipliers[0], cmap="coolwarm")
fig.colorbar(sm, ax=ax, label="Speed multiplier")
ax.legend(["Initial layout", "Optimized layout", "Optimization boundary"])
ax.set_title("Geometric yaw disabled")


# Rerun the layout optimization with geometric yaw enabled
print("\nReoptimizing with geometric yaw enabled.")
fmodel.set(layout_x=layout_x, layout_y=layout_y)
layout_opt = LayoutOptimizationScipy(
    fmodel, boundaries, min_dist=2 * D, enable_geometric_yaw=True, optOptions={"maxiter": maxiter}
)

# Run the optimization
np.random.seed(0)
sol = layout_opt.optimize()

# Get the resulting improvement in AEP
print("... calcuating improvement in AEP")

fmodel.set(yaw_angles=np.zeros_like(layout_opt.yaw_angles))
fmodel.run()
base_aep = fmodel.get_farm_AEP() / 1e6
fmodel.set(layout_x=sol[0], layout_y=sol[1], yaw_angles=layout_opt.yaw_angles)
fmodel.run()
opt_aep = fmodel.get_farm_AEP() / 1e6

percent_gain = 100 * (opt_aep - base_aep) / base_aep

# Print and plot the results
print(f"Optimal layout: {sol}")
print(
    f"Optimal layout improves AEP by {percent_gain:.1f}% "
    f"from {base_aep:.1f} MWh to {opt_aep:.1f} MWh"
)
layout_opt.plot_layout_opt_results()
ax = plt.gca()
fig = plt.gcf()
sm = ax.tricontourf(x_locs, y_locs, speed_multipliers[0], cmap="coolwarm")
fig.colorbar(sm, ax=ax, label="Speed multiplier")
ax.legend(["Initial layout", "Optimized layout", "Optimization boundary"])
ax.set_title("Geometric yaw enabled")

print(
    "Turbine geometric yaw angles for wind direction {0:.2f}".format(wind_directions[1])
    + " and wind speed {0:.2f} m/s:".format(wind_speeds[0]),
    f"{layout_opt.yaw_angles[1, :]}",
)

plt.show()
