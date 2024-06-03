import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from floris import FlorisModel

# Data from Nygaard
tp_paper_vels = [
    1.0,
    0.709920677983239,
    0.615355749367675,
    0.551410465937128,
    0.502600655337247,
    0.463167556093190,
    0.430238792036599,
    0.402137593655074,
    0.377783142608699,
    0.356429516711137,
]

# First, run the current TurboPark implementation from FLORIS.
fmodel_tp_orig = FlorisModel("Case_RowPark_TurbOPark.yaml")
fmodel_tp_orig.run()
tp_orig_vels = fmodel_tp_orig.turbine_average_velocities

# Get free stream wind speed for normalizing
u0 = fmodel_tp_orig.wind_speeds[0]

# New implementation
fmodel_tp_new = FlorisModel("Case_RowPark_TurbOParkGauss.yaml")
fmodel_tp_new.run()
tp_new_vels = fmodel_tp_new.turbine_average_velocities

# Plot the data and compare to Nygaard's results
turbines = range(1,11)
fig, ax = plt.subplots()
ax.scatter(turbines, tp_paper_vels, s=80, marker="o", color="k", label="Orsted - TurbOPark")
ax.scatter(turbines, tp_orig_vels/u0, s=20, marker="o", label="Floris - TurbOPark")
ax.scatter(turbines, tp_new_vels/u0, s=20, marker="o", label="Floris - TurbOPark_Gauss")
ax.set_xlabel("Turbine number")
ax.set_ylabel("Normalized waked wind speed [-]")
ax.set_xlim(0, 11)
ax.set_ylim(0.25, 1.05)
ax.legend()
ax.grid()

plt.show()
