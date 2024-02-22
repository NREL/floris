import matplotlib.pyplot as plt
import numpy as np

from floris.tools import FlorisInterface, UncertaintyInterface


"""
This example demonstrates how one can create an "UncertaintyInterface" object,
which adds uncertainty on the inflow wind direction on the FlorisInterface
class. The UncertaintyInterface class is interacted with in the exact same
manner as the FlorisInterface class is. This example demonstrates how the
wind farm power production is calculated with and without uncertainty.
Other use cases of UncertaintyInterface are, e.g., comparing FLORIS to
historical SCADA data and robust optimization.
"""

# Instantiate FLORIS using either the GCH or CC model
fi = FlorisInterface("inputs/gch.yaml")  # GCH model
fi_unc_3 = UncertaintyInterface(
    "inputs/gch.yaml", verbose=True, wd_std=3
)  # Add uncertainty with default settings
fi_unc_5 = UncertaintyInterface(
    "inputs/gch.yaml", verbose=True, wd_std=5
)  # Add uncertainty with default settings

# Define a two turbine farm
D = 126.0
layout_x = np.array([0, D * 6])
layout_y = [0, 0]
wd_array = np.arange(240.0, 300.0, 1.0)
wind_speeds = 8.0 * np.ones_like(wd_array)
fi.set(layout_x=layout_x, layout_y=layout_y, wind_directions=wd_array, wind_speeds=wind_speeds)
fi_unc_3.set(
    layout_x=layout_x, layout_y=layout_y, wind_directions=wd_array, wind_speeds=wind_speeds
)
fi_unc_5.set(
    layout_x=layout_x, layout_y=layout_y, wind_directions=wd_array, wind_speeds=wind_speeds
)


# Run both models
fi.run()
fi_unc_3.run()
fi_unc_5.run()

# Collect the nominal and uncertain farm power
turbine_powers_nom = fi.get_turbine_powers() / 1e3
turbine_powers_unc_3 = fi_unc_3.get_turbine_powers() / 1e3
turbine_powers_unc_5 = fi_unc_5.get_turbine_powers() / 1e3
farm_powers_nom = fi.get_farm_power() / 1e3
farm_powers_unc_3 = fi_unc_3.get_farm_power() / 1e3
farm_powers_unc_5 = fi_unc_5.get_farm_power() / 1e3

# Plot results
fig, axarr = plt.subplots(1, 3, figsize=(15, 5))
ax = axarr[0]
ax.plot(wd_array, turbine_powers_nom[:, 0].flatten(), color="k", label="Nominal power")
ax.plot(
    wd_array,
    turbine_powers_unc_3[:, 0].flatten(),
    color="r",
    label="Power with uncertainty = 3 deg",
)
ax.plot(
    wd_array, turbine_powers_unc_5[:, 0].flatten(), color="m", label="Power with uncertainty = 5deg"
)
ax.grid(True)
ax.legend()
ax.set_xlabel("Wind Direction (deg)")
ax.set_ylabel("Power (kW)")
ax.set_title("Upstream Turbine")

ax = axarr[1]
ax.plot(wd_array, turbine_powers_nom[:, 1].flatten(), color="k", label="Nominal power")
ax.plot(
    wd_array,
    turbine_powers_unc_3[:, 1].flatten(),
    color="r",
    label="Power with uncertainty = 3 deg",
)
ax.plot(
    wd_array,
    turbine_powers_unc_5[:, 1].flatten(),
    color="m",
    label="Power with uncertainty = 5 deg",
)
ax.set_title("Downstream Turbine")
ax.grid(True)
ax.legend()
ax.set_xlabel("Wind Direction (deg)")
ax.set_ylabel("Power (kW)")

ax = axarr[2]
ax.plot(wd_array, farm_powers_nom.flatten(), color="k", label="Nominal farm power")
ax.plot(
    wd_array, farm_powers_unc_3.flatten(), color="r", label="Farm power with uncertainty = 3 deg"
)
ax.plot(
    wd_array, farm_powers_unc_5.flatten(), color="m", label="Farm power with uncertainty = 5 deg"
)
ax.set_title("Farm Power")
ax.grid(True)
ax.legend()
ax.set_xlabel("Wind Direction (deg)")
ax.set_ylabel("Power (kW)")

# Compare the AEP calculation
freq = np.ones_like(wd_array)
freq = freq / freq.sum()

aep_nom = fi.get_farm_AEP(freq=freq)
aep_unc_3 = fi_unc_3.get_farm_AEP(freq=freq)
aep_unc_5 = fi_unc_5.get_farm_AEP(freq=freq)

print(f"AEP without uncertainty {aep_nom}")
print(f"AEP without uncertainty (3 deg) {aep_unc_3} ({100*aep_unc_3/aep_nom:.2f}%)")
print(f"AEP without uncertainty (5 deg) {aep_unc_5} ({100*aep_unc_5/aep_nom:.2f}%)")


plt.show()
