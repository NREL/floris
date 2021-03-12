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


import numpy as np
import matplotlib.pyplot as plt

import floris.tools as wfct


# User inputs
D = 126.0
nturbs_x = 5
nturbs_y = 5
x_spacing = 5 * D
y_spacing = 4 * D
ws = 8.0
wd = 270.0

# Generate layout
layout_x = [i * x_spacing for j in range(nturbs_y) for i in range(nturbs_x)]
layout_y = [j * y_spacing for j in range(nturbs_y) for i in range(nturbs_x)]
# layout_x = [0.0, 6*126.0, 1*126.0, 7*126.0]
# layout_y = [0.0, 0.0, 6*126.0, 6*126.0]
layout_array = [layout_x, layout_y]

fi_gauss_cumulative = wfct.floris_interface.FlorisInterface("../example_input.json")
fi_gauss_legacy = wfct.floris_interface.FlorisInterface("../example_input.json")

fi_gauss_cumulative.floris.farm.set_wake_model("gauss_cumulative")
fi_gc_params = {
    "Wake Turbulence Parameters": {
        "ti_ai": 0.83,
        "ti_constant": 0.66,
        "ti_downstream": -0.32,
        "ti_initial": 0.03,
    }
}
fi_gauss_cumulative.set_model_parameters(fi_gc_params)
fi_gauss_cumulative.set_gch(enable=False)
fi_gauss_cumulative.floris.farm.flow_field.solver = "gauss_cumulative"
fi_gauss_cumulative.reinitialize_flow_field(
    wind_speed=ws, wind_direction=wd, layout_array=layout_array
)
# Calculate wake
fi_gauss_cumulative.calculate_wake()

fi_gauss_legacy.floris.farm.set_wake_model("gauss_legacy")
fi_gauss_legacy.floris.farm.flow_field.solver = "floris"
fi_gauss_legacy.reinitialize_flow_field(
    wind_speed=ws, wind_direction=wd, layout_array=layout_array
)
# Calculate wake
fi_gauss_legacy.calculate_wake()

# Get horizontal plane at default height (hub-height)
hor_plane_gauss_cumulative = fi_gauss_cumulative.get_hor_plane()
hor_plane_gauss_legacy = fi_gauss_legacy.get_hor_plane()

# Plot and show
fig, axarr = plt.subplots(2, 1)
wfct.visualization.visualize_cut_plane(hor_plane_gauss_cumulative, ax=axarr[0])
axarr[0].set_title("Gauss Cumulative")
wfct.visualization.visualize_cut_plane(hor_plane_gauss_legacy, ax=axarr[1])
axarr[1].set_title("Gauss Legacy")

gauss_cumulative_turb_power = np.array(fi_gauss_cumulative.get_turbine_power()) / 1e6
gauss_legacy_turb_power = np.array(fi_gauss_legacy.get_turbine_power()) / 1e6

labels = ["T" + str(i) for i in range(len(fi_gauss_cumulative.layout_x))]

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, gauss_cumulative_turb_power, width, label="Cumulative")
rects2 = ax.bar(x, gauss_legacy_turb_power, width, label="Legacy")

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("Turbine Powers [MW]")
ax.set_xlabel("Turbine")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            "{:.2f}".format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 1),  # 1 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)

fig.tight_layout()

plt.show()
