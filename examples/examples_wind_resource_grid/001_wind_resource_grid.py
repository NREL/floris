"""Example: Wind Resource Grid

Open the WRG file generated in the previous example using the `WindRoseWRG` object
and show that wind roses can be interpolated between grid points.

"""
import matplotlib.pyplot as plt
import numpy as np

from floris import WindRoseWRG


# Read the WRG file
wind_rose_wrg = WindRoseWRG("wrg_example.wrg")

# Print some basic information
print(wind_rose_wrg)

# The wind roses were set to ...
y_points_to_test = np.array([0, 500, 1000, 1500, 2000])

fig, axarr = plt.subplots(1, 5, figsize=(16, 5), subplot_kw={"polar": True})

for i in range(5):
    wind_rose = wind_rose_wrg.get_wind_rose_at_point(0, y_points_to_test[i])
    wind_rose.plot(ax=axarr[i], ws_step=5)
    if i %2 == 0:
        axarr[i].set_title(f"y = {y_points_to_test[i]}")
    else:
        axarr[i].set_title(f"y = {y_points_to_test[i]}\n(Interpolated)")

plt.show()
