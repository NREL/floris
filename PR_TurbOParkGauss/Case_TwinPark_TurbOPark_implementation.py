import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from floris import FlorisModel, TimeSeries

### Look at the wake profile at a single downstream distance for a range of wind directions
# Load the original TurboPark implementation
fmodel_orig = FlorisModel("Case_TwinPark_TurbOPark.yaml")

# Set up and solve flows
wd_array = np.arange(225,315,0.1)
fmodel_orig.set(
    layout_x = [0.0, 600.0],
    layout_y = [0.0, 0.0],
    wind_data=TimeSeries(wind_speeds=8.0, wind_directions=wd_array, turbulence_intensities=0.06)
)
fmodel_orig.run()

# Extract output velocities at downstream turbine
orig_vels_ds = fmodel_orig.turbine_average_velocities[:,1]
u0 = fmodel_orig.wind_speeds[0]

# Load the new implementation
fmodel_new = FlorisModel("Case_TwinPark_TurbOParkGauss.yaml")

# Set up and solve flows; extract velocities at downstream turbine
fmodel_new.set(
    layout_x = [0.0, 600.0],
    layout_y = [0.0, 0.0],
    wind_data=TimeSeries(wind_speeds=8.0, wind_directions=wd_array, turbulence_intensities=0.06)
)
fmodel_new.run()
new_vels_ds = fmodel_new.turbine_average_velocities[:,1]

# Load comparison data
df = pd.read_csv("WindDirection_Sweep_Orsted.csv")

# Plot the data and compare
fig, ax = plt.subplots()
ax.plot(wd_array, orig_vels_ds/u0, label="Floris 4 - TurbOPark")
ax.plot(wd_array, new_vels_ds/u0, label="Floris 4 - TurbOPark-Gauss")
df.plot("wd", "wws", ax=ax, linestyle="--", color="k", label="Orsted - TurbOPark")

ax.set_xlabel("Wind direction [deg]")
ax.set_ylabel("Normalized rotor averaged waked wind speed [-]")
ax.set_xlim(240,300)
ax.set_ylim(0.65,1.05)
ax.legend()
ax.grid()
plt.show()
