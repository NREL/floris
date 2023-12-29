# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import NearestNDInterpolator

import floris.tools.visualization as wakeviz
from floris.tools import FlorisInterface


"""
This example demonstrates the impact of floating on turbine power and thurst
and wake behavior. A floating turbine in FLORIS is defined by including a
`floating_tilt_table` in the turbine input yaml which sets the steady tilt
angle of the turbine based on wind speed.  This tilt angle is computed for each
turbine based on effective velocity.  This tilt angle is then passed on
to the respective wake model.

The value of the parameter ref_tilt is the value of tilt at which the
ct/cp curves have been defined.

With `correct_cp_ct_for_tilt` True, the difference between the current
tilt as interpolated from the floating tilt table is used to scale the turbine
power and thrust.

In the example below, a 20-turbine, gridded wind farm is simulated using
the Empirical Gaussian wake model to show the effects of floating turbines on
both turbine power and wake development.

fi_fixed: Fixed bottom turbine (no tilt variation with wind speed)
fi_floating: Floating turbine (tilt varies with wind speed)
"""

# Declare the Floris Interface for fixed bottom, provide layout
fi_fixed = FlorisInterface("inputs_floating/emgauss_fixed.yaml")
fi_floating = FlorisInterface("inputs_floating/emgauss_floating.yaml")
x, y = np.meshgrid(np.linspace(0, 4*630., 5), np.linspace(0, 3*630., 4))
x = x.flatten()
y = y.flatten()
for fi in [fi_fixed, fi_floating]:
    fi.reinitialize(layout_x=x, layout_y=y)

# Compute a single wind speed and direction, power and wakes
for fi in [fi_fixed, fi_floating]:
    fi.reinitialize(
        layout_x=x,
        layout_y=y,
        wind_speeds=[10],
        wind_directions=[270]
    )
    fi.calculate_wake()

powers_fixed = fi_fixed.get_turbine_powers()
powers_floating = fi_floating.get_turbine_powers()
power_difference = powers_floating - powers_fixed

# Show the power differences
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
sc = ax.scatter(
    x,
    y,
    c=power_difference.flatten()/1000,
    cmap="PuOr",
    vmin=-30,
    vmax=30,
    s=200,
)
ax.set_xlabel("x coordinate [m]")
ax.set_ylabel("y coordinate [m]")
ax.set_title("Power increase due to floating for each turbine.")
plt.colorbar(sc, label="Increase (kW)")

print("Power increase from floating over farm (10m/s, 270deg winds): {0:.2f} kW".\
    format(power_difference.sum()/1000))

# Visualize flows (see also 02_visualizations.py)
horizontal_planes = []
y_planes = []
for fi in [fi_fixed, fi_floating]:
    horizontal_planes.append(
        fi.calculate_horizontal_plane(
            x_resolution=200,
            y_resolution=100,
            height=90.0,
        )
    )
    y_planes.append(
        fi.calculate_y_plane(
            x_resolution=200,
            z_resolution=100,
            crossstream_dist=0.0,
        )
    )

# Create the plots
fig, ax_list = plt.subplots(2, 1, figsize=(10, 8))
ax_list = ax_list.flatten()
wakeviz.visualize_cut_plane(horizontal_planes[0], ax=ax_list[0], title="Horizontal")
wakeviz.visualize_cut_plane(y_planes[0], ax=ax_list[1], title="Streamwise profile")
fig.suptitle("Fixed-bottom farm")

fig, ax_list = plt.subplots(2, 1, figsize=(10, 8))
ax_list = ax_list.flatten()
wakeviz.visualize_cut_plane(horizontal_planes[1], ax=ax_list[0], title="Horizontal")
wakeviz.visualize_cut_plane(y_planes[1], ax=ax_list[1], title="Streamwise profile")
fig.suptitle("Floating farm")

# Compute AEP (see 07_calc_aep_from_rose.py for details)
df_wr = pd.read_csv("inputs/wind_rose.csv")
wd_grid, ws_grid = np.meshgrid(
    np.array(df_wr["wd"].unique(), dtype=float),
    np.array(df_wr["ws"].unique(), dtype=float),
    indexing="ij"
)
freq_interp = NearestNDInterpolator(df_wr[["wd", "ws"]], df_wr["freq_val"])
freq = freq_interp(wd_grid, ws_grid).flatten()
freq = freq / np.sum(freq)

for fi in [fi_fixed, fi_floating]:
    fi.reinitialize(
        wind_directions=wd_grid.flatten(),
        wind_speeds= ws_grid.flatten(),
    )

# Compute the AEP
aep_fixed = fi_fixed.get_farm_AEP(freq=freq)
aep_floating = fi_floating.get_farm_AEP(freq=freq)
print("Farm AEP (fixed bottom): {:.3f} GWh".format(aep_fixed / 1.0e9))
print("Farm AEP (floating): {:.3f} GWh".format(aep_floating / 1.0e9))
print(
    "Floating AEP increase: {0:.3f} GWh ({1:.2f}%)".\
    format((aep_floating - aep_fixed) / 1.0e9, (aep_floating - aep_fixed)/aep_fixed*100)
)

plt.show()
