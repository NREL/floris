
import matplotlib.pyplot as plt
import numpy as np

from floris.tools import FlorisInterface
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR


"""
This example demonstrates how to perform a yaw optimization for multiple wind directions
and 1 wind speed.

First, we initialize our Floris Interface, and then generate a 3 turbine wind farm.
Next, we create the yaw optimization object `yaw_opt` and perform the optimization using the
SerialRefine method. Finally, we plot the results.
"""

# Load the default example floris object
fi = FlorisInterface("inputs/gch.yaml") # GCH model matched to the default "legacy_gauss" of V2
# fi = FlorisInterface("inputs/cc.yaml") # New CumulativeCurl model

# Reinitialize as a 3-turbine farm with range of WDs and 1 WS
wd_array = np.arange(0.0, 360.0, 3.0)
ws_array = 8.0 * np.ones_like(wd_array)
D = 126.0 # Rotor diameter for the NREL 5 MW
fi.reinitialize(
    layout_x=[0.0, 5 * D, 10 * D],
    layout_y=[0.0, 0.0, 0.0],
    wind_directions=wd_array,
    wind_speeds=ws_array,
)
print(fi.floris.farm.rotor_diameters)

# Initialize optimizer object and run optimization using the Serial-Refine method
yaw_opt = YawOptimizationSR(fi)
df_opt = yaw_opt.optimize()

print("Optimization results:")
print(df_opt)

# Split out the turbine results
for t in range(3):
    df_opt['t%d' % t] = df_opt.yaw_angles_opt.apply(lambda x: x[t])

# Show the results
fig, axarr = plt.subplots(2,1,sharex=True,sharey=False,figsize=(8,8))

# Yaw results
ax = axarr[0]
for t in range(3):
    ax.plot(df_opt.wind_direction,df_opt['t%d' % t],label='t%d' % t)
ax.set_ylabel('Yaw Offset (deg')
ax.legend()
ax.grid(True)

# Power results
ax = axarr[1]
ax.plot(df_opt.wind_direction,df_opt.farm_power_baseline,color='k',label='Baseline Farm Power')
ax.plot(df_opt.wind_direction,df_opt.farm_power_opt,color='r',label='Optimized Farm Power')
ax.set_ylabel('Power (W)')
ax.set_xlabel('Wind Direction (deg)')
ax.legend()
ax.grid(True)

plt.show()
