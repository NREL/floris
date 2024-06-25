import matplotlib.pyplot as plt
import numpy as np

from floris import WindRose, WindTIRose
from floris.utilities import wrap_180


# Case 1, 4 wind directions with a 1 wind speed, uniform frequency distribution
wind_directions = np.array([0, 90, 180, 270])
wind_speeds = np.array([8.0])

# Create a wind rose
wind_rose = WindRose(
    wind_directions=wind_directions,
    wind_speeds=wind_speeds,
    ti_table=0.06,
)

# Show the wind rose
wind_rose.plot()

# Case 2, test this for the WindTIRose
turbulence_intensities = np.array([0.06])
wind_ti_rose = WindTIRose(
    wind_directions=wind_directions,
    wind_speeds=wind_speeds,
    turbulence_intensities=turbulence_intensities,
)

# Show the wind_ti_rose
wind_ti_rose.plot(wd_step = 90.)

# Case 3, double the frequency of the east direction
freq_table = np.array([[0.1], [0.2], [0.1], [0.1]])
freq_table = freq_table / freq_table.sum()

# Create a wind rose
wind_rose = WindRose(
    wind_directions=wind_directions, wind_speeds=wind_speeds, ti_table=0.06, freq_table=freq_table
)

# Show the wind rose
wind_rose.plot()

# Case 4, 1 wind directions with a 4 wind speeds, uniform frequency distribution
wind_directions = np.array([0.0])
wind_speeds = np.array([2.5, 7.5, 12.5, 17.5])

# Create a wind rose
wind_rose = WindRose(
    wind_directions=wind_directions,
    wind_speeds=wind_speeds,
    ti_table=0.06,
)

# Show the wind rose
wind_rose.plot()

# Case 5, 2 wind directions and 4 wind speeds, uniform frequency distribution,
# except for 17.5 m./s at 0 deg, which is 0
wind_directions = np.array([0.0, 180.0])
wind_speeds = np.array([2.5, 7.5, 12.5, 17.5])
freq_table = np.array([[0.25, 0.25, 0.25, 0.0], [0.25, 0.25, 0.25, 0.25]])
freq_table = freq_table / freq_table.sum()

# Create a wind rose
wind_rose = WindRose(
    wind_directions=wind_directions, wind_speeds=wind_speeds, ti_table=0.06, freq_table=freq_table
)

# Show the wind rose
wind_rose.plot()

# Case 6, same as case 4 but make 0 deg and 180 deg have equal probablility
freq_table = np.array([[1 + 1.0 / 3.0, 1 + 1.0 / 3.0, 1 + 1.0 / 3.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
freq_table = freq_table / freq_table.sum()

# Create a wind rose
wind_rose = WindRose(
    wind_directions=wind_directions, wind_speeds=wind_speeds, ti_table=0.06, freq_table=freq_table
)

# Show the wind rose
wind_rose.plot()

plt.show()
