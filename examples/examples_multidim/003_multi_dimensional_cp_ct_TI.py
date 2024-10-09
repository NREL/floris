"""Example: Multi-dimensional Cp/Ct with 2 Hs values
This example follows the previous example but shows the effect of changing the Hs setting.

NOTE: The multi-dimensional Cp/Ct data used in this example is fictional for the purposes of
facilitating this example. The Cp/Ct values for the different wave conditions are scaled
values of the original Cp/Ct data for the IEA 15MW turbine.
"""


import matplotlib.pyplot as plt
import numpy as np

from floris import FlorisModel, TimeSeries


# Initialize FLORIS with the given input file.
fmodel = FlorisModel("../inputs/gch_multi_dim_cp_ct_TI.yaml")

# Make a second Floris instance with a different setting for Hs.
# Note the multi-cp-ct file (iea_15MW_multi_dim_Tp_Hs.csv)
# for the turbine model iea_15MW_floating_multi_dim_cp_ct.yaml
# Defines Hs at 1 and 5.
# The value in gch_multi_dim_cp_ct.yaml is 3.01 which will map
# to 5 as the nearer value, so we set the other case to 1
# for contrast.
# fmodel_dict_mod = fmodel.core.as_dict()
# fmodel_dict_mod["flow_field"]["multidim_conditions"]["TI"] = 0.10
# # TODO: create a set_multidim_conditions method on fmodel (which reinitializes)
# # Can check here if it is a valid key
# # OR: add to set() method. Each value in multidim_conditions dict should be a scalar value
# fmodel_hs_1 = FlorisModel(fmodel_dict_mod)

# Set both cases to 3 turbine layout
fmodel.set(layout_x=[0.0, 500.0, 1000.0], layout_y=[0.0, 0.0, 0.0])

# Use a sweep of wind speeds
wind_speeds = np.arange(5, 20, 1.0)
time_series = TimeSeries(
    wind_directions=270.0, wind_speeds=wind_speeds, turbulence_intensities=0.06
)
fmodel.set(wind_data=time_series)

fig, axarr = plt.subplots(1, 3, sharex=True, figsize=(12, 4))
for ti, col in zip([0.06, 0.10], ["k", "r"]):
    fmodel.set(multidim_conditions={"TI": ti})
    fmodel.run()
    turbine_powers = fmodel.get_turbine_powers() / 1000.0

    for t_idx in range(3):
        ax = axarr[t_idx]
        ax.plot(wind_speeds, turbine_powers[:, t_idx], color=col, label="TI={0:.2f}".format(ti))
for t_idx in range(3):
    axarr[t_idx].grid(True)
    axarr[t_idx].set_xlabel("Wind Speed (m/s)")
    axarr[t_idx].set_title(f"Turbine {t_idx}")
axarr[0].legend()

plt.show()
