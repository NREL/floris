import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from floris import FlorisModel, TimeSeries

### Look at the wake profile at a single downstream distance for a range of wind directions
# Load the original TurboPark implementation
fmodel_orig = FlorisModel("Case_TwinPark_TurbOPark.yaml")

# Set up and solve flows
wd_array = np.arange(225,315,0.1)
wind_data_wd_sweep = TimeSeries(
    wind_speeds=8.0,
    wind_directions=wd_array,
    turbulence_intensities=0.06
)
fmodel_orig.set(
    layout_x = [0.0, 600.0],
    layout_y = [0.0, 0.0],
    wind_data=wind_data_wd_sweep
)
fmodel_orig.run()

# Extract output velocities at downstream turbine
orig_vels_ds = fmodel_orig.turbine_average_velocities[:,1]
u0 = fmodel_orig.wind_speeds[0] # Get freestream wind speed for normalization

# Load the new implementation
fmodel_new = FlorisModel("Case_TwinPark_TurbOParkGauss.yaml")

# Set up and solve flows; extract velocities at downstream turbine
fmodel_new.set(
    layout_x = [0.0, 600.0],
    layout_y = [0.0, 0.0],
    wind_data=wind_data_wd_sweep
)
fmodel_new.run()
new_vels_ds = fmodel_new.turbine_average_velocities[:,1]

# Load comparison data
df_twinpark = pd.read_csv("WindDirection_Sweep_Orsted.csv")

# Plot the data and compare
fig, ax = plt.subplots(2, 1)
fig.set_size_inches(7, 10)
ax[0].plot(wd_array, orig_vels_ds/u0, label="Floris - TurbOPark")
ax[0].plot(wd_array, new_vels_ds/u0, label="Floris - TurbOPark-Gauss")
df_twinpark.plot("wd", "wws", ax=ax[0], linestyle="--", color="k", label="Orsted - TurbOPark")

ax[0].set_xlabel("Wind direction [deg]")
ax[0].set_ylabel("Normalized rotor averaged waked wind speed [-]")
ax[0].set_xlim(240,300)
ax[0].set_ylim(0.65,1.05)
ax[0].legend()
ax[0].grid()

### Now, look at velocities along a row of ten turbines aligned with the flow
layout_x = np.linspace(0.0, 5400.0, 10)
layout_y = np.zeros_like(layout_x)
turbines = range(len(layout_x))
wind_data_row = TimeSeries(
    wind_speeds=np.array([8.0]),
    wind_directions=270.0,
    turbulence_intensities=0.06
)
fmodel_orig.set(
    layout_x=layout_x,
    layout_y=layout_y,
    wind_data=wind_data_row
)
fmodel_new.set(
    layout_x=layout_x,
    layout_y=layout_y,
    wind_data=wind_data_row
)

# Run and extract flow velocities at the turbines
fmodel_orig.run()
orig_vels_row = fmodel_orig.turbine_average_velocities
fmodel_new.run()
new_vels_row = fmodel_new.turbine_average_velocities
u0 = fmodel_orig.wind_speeds[0] # Get freestream wind speed for normalization

# Load comparison data
df_rowpark = pd.read_csv("Rowpark_Orsted.csv")

# Plot the data and compare
ax[1].scatter(turbines, df_rowpark["wws"], s=80, marker="o", color="k", label="Orsted - TurbOPark")
ax[1].scatter(turbines, orig_vels_row/u0, s=20, marker="o", label="Floris - TurbOPark")
ax[1].scatter(turbines, new_vels_row/u0, s=20, marker="o", label="Floris - TurbOPark_Gauss")
ax[1].set_xlabel("Turbine number")
ax[1].set_ylabel("Normalized rotor averaged wind speed [-]")
ax[1].set_ylim(0.25, 1.05)
ax[1].legend()
ax[1].grid()

plt.show()
