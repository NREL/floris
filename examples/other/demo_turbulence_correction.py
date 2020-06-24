import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import floris.tools as wfct


print("========== visualizing flow field with one turbine =========")
fi = wfct.floris_interface.FlorisInterface("../example_input.json")
fi.reinitialize_flow_field(layout_array=([0], [0]), turbulence_intensity=[0.15])
fi.calculate_wake()

# calculate turbine power output with and without turbulence correction
_power = fi.get_turbine_power()
turbulent_power = fi.get_turbine_power(use_turbulence_correction=True)


hor_plane = fi.get_hor_plane()


txt = (
    str("Turbine Power Output: \n       With Turbulence Correction: ")
    + str(turbulent_power[0] * 10 ** -6)
    + (" MW\n       Without Turbulence Correction: ")
    + str(_power[0] * 10 ** -6)
    + str(" MW")
)

fig = plt.figure(figsize=(10, 7))
ax = plt.subplot()
im = wfct.visualization.visualize_cut_plane(hor_plane, ax)
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
cbar.set_label("Wind Speed (m/s)", labelpad=+10)
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
plt.text(0, -0.5, txt, transform=ax.transAxes)
plt.show(im)


print(
    "========== calculating power curve at each wind speed and turbulence intensity =========="
)
sp = [
    0.1,
    3.5,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
    11.0,
    12.0,
    13.0,
    14.0,
    15.0,
    16.0,
    17.0,
    18.0,
    19.0,
    20.0,
    21.0,
    22.0,
    23.0,
    24.0,
    30.0,
]
power = np.zeros((10, 24))
ti = np.linspace(0.0, 0.5, num=10)
for i in range(10):
    TI = [ti[i]]
    powers = []
    for j in range(24):
        speed = [sp[j]]
        fi.reinitialize_flow_field(wind_speed=speed, turbulence_intensity=TI)
        fi.calculate_wake()
        p = fi.get_turbine_power(use_turbulence_correction=True)
        powers.append(p[0])
    power[i] = powers


print("========== plotting adjusted power curve ==========")
color = [
    "navy",
    "#014d4e",
    "teal",
    "g",
    "yellowgreen",
    "y",
    "orange",
    "red",
    "crimson",
    "maroon",
]
fig = plt.subplots(figsize=(10, 7))
for i in range(10):
    plt.plot(sp, list(power[i, :]), c=color[i], label=str(int(ti[i] * 100)) + "%")
plt.legend()
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Power (W)")
plt.show()
