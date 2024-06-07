"""Example: Use Wind Resource Grid

Open the WRG file generated in the previous example using the `WindResourceGrid` object
and show that wind roses can be interpolated between grid points.

"""
import matplotlib.pyplot as plt
import numpy as np

from floris import WindResourceGrid


# Read the WRG file
wrg = WindResourceGrid("wrg_example.wrg")

# Print some basic information
print(wrg)

# The wind roses were set to rotate at each movement along the y-direction,
# show that the points in between the grid points have wind roses derived
# by interpolating the provided sector probabilities, and Weibull parameters.
y_points_to_test = np.array([0, 500, 1000, 1500, 2000])

fig, axarr = plt.subplots(1, 5, figsize=(16, 5), subplot_kw={"polar": True})

for i in range(5):
    wind_rose = wrg.get_wind_rose_at_point(0, y_points_to_test[i])
    wind_rose.plot(ax=axarr[i], ws_step=5)
    if i %2 == 0:
        axarr[i].set_title(f"y = {y_points_to_test[i]}")
    else:
        axarr[i].set_title(f"y = {y_points_to_test[i]}\n(Interpolated)")

plt.show()
