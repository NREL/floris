# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

# In the second case, simply time calls to calculate_wake

import time
import pickle

import numpy as np

import floris.tools as wfct
import matplotlib.pyplot as plt


# PARAMETERS
recompute_baseline = False
show_layout = False
repeats = 4  # Number of times to repeat the analysis
N = 10  # Number of iterations in timing loop
num_turbine = 100
turn_off_gch = False


# Initialize the FLORIS interface fi
# For basic usage, the florice interface provides a simplified interface to
# the underlying classes
fi = wfct.floris_interface.FlorisInterface("../../example_input.json")

if turn_off_gch:
    fi.set_gch(False)

# Set to a 5 turbine case
D = 126.0
fi.reinitialize_flow_field(
    layout_array=[
        [D * 6 * i for i in range(num_turbine)],
        [0 for i in range(num_turbine)],
    ]
)

# Calculate wake
fi.calculate_wake()

# Now check the timing
print("===START TEST===")
timing_result = []
for r in range(repeats):
    start = time.perf_counter()
    for i in range(N):
        fi.calculate_wake()
    end = time.perf_counter()
    elapsed_time = (end - start) / N
    timing_result.append(elapsed_time)

timing_result = np.array(timing_result)

# Collect the turbine powers
fi.calculate_wake()
turbine_powers = np.array(fi.get_turbine_power())

# Report the timing
print("====RESULT====")
print(
    "*** calculate wake takes on average %.1f ms, and ranges (%.1f -- %.1f)"
    % (
        timing_result.mean() * 1000,
        timing_result.min() * 1000,
        timing_result.max() * 1000,
    )
)
print("*** exact result in s -> %f" % timing_result.mean())

# Now check if result has changed
if recompute_baseline:
    pickle.dump(turbine_powers, open("result_case_2c.p", "wb"))
else:
    saved_result = pickle.load(open("result_case_2c.p", "rb"))
    new_total = turbine_powers.sum()
    saved_total = saved_result.sum()
    power_difference = turbine_powers - saved_result

    if np.max(np.abs(power_difference)) == 0.0:
        print("*** Turbine powers unchanged from baseline")

    else:
        print("xxx Power result has changed")
        percent_change = 100 * (new_total - saved_total) / saved_total
        print(
            "xxx Total power changed by %+.1f%% (%.1f -> %.1f)"
            % (percent_change, saved_total, new_total)
        )
        for t in range(num_turbine):
            t_saved = saved_result[t]
            t_new = turbine_powers[t]
            percent_change = 100.0 * (t_new - t_saved) / t_saved
            print(
                " xxx T%d: %+.1f%%: (%.1f -> %.1f)"
                % (t, percent_change, t_saved, t_new)
            )


if show_layout:
    # Plot and show
    # Get horizontal plane at default height (hub-height)
    hor_plane = fi.get_hor_plane()
    fig, ax = plt.subplots()
    wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
    plt.show()
