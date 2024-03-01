
import matplotlib.pyplot as plt
import numpy as np

from floris.tools import FlorisInterface


"""
05_sweep_wind_speeds

This example sweeps wind speeds while holding wind direction constant

The power of both turbines for each wind speed is then plotted

"""


# Instantiate FLORIS using either the GCH or CC model
fi = FlorisInterface("inputs/gch.yaml") # GCH model matched to the default "legacy_gauss" of V2

# Define a two turbine farm
D = 126.
layout_x = np.array([0, D*6])
layout_y = [0, 0]
fi.set(layout_x=layout_x, layout_y=layout_y)

# Sweep wind speeds but keep wind direction fixed
ws_array = np.arange(5,25,0.5)
wd_array = 270.0 * np.ones_like(ws_array)
fi.set(wind_directions=wd_array,wind_speeds=ws_array)

# Define a matrix of yaw angles to be all 0
# Note that yaw angles is now specified as a matrix whose dimensions are
# wd/ws/turbine
num_wd = len(wd_array)
num_ws = len(ws_array)
n_findex = num_wd  # Could be either num_wd or num_ws
num_turbine = len(layout_x)
yaw_angles = np.zeros((n_findex, num_turbine))
fi.set(yaw_angles=yaw_angles)

# Calculate
fi.run()

# Collect the turbine powers
turbine_powers = fi.get_turbine_powers() / 1E3 # In kW

# Pull out the power values per turbine
pow_t0 = turbine_powers[:,0].flatten()
pow_t1 = turbine_powers[:,1].flatten()

# Plot
fig, ax = plt.subplots()
ax.plot(ws_array,pow_t0,color='k',label='Upstream Turbine')
ax.plot(ws_array,pow_t1,color='r',label='Downstream Turbine')
ax.grid(True)
ax.legend()
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Power (kW)')
plt.show()
