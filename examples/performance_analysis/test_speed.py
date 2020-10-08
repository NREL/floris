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


# This example is meant to time the two core functions of floris:
# calculate_wake and reinitialize_flow_field
# as the size of the farm is increased

import timeit


results = {}

## Parameters
d_space = 7
d = 126
dist = d * d_space

n_row_to_test = [5, 6, 7, 8, 9, 10, 11]

for n_row in n_row_to_test:
    print(
        "****** Timing %d turbine wind farm which is %d x %d m in size"
        % (n_row * n_row, n_row * dist, n_row * dist)
    )

    setup_code = (
        """
import floris.tools as wfct
import numpy as np

# Initialize the FLORIS interface fi
fi = wfct.floris_interface.FlorisInterface("../example_input.json")

## Parameters
n_row = %d
d_space = 7
d = 126
dist = d * d_space

x_array = []
y_array = []

for x in np.arange(0,dist * n_row,dist):
    for y in np.arange(0,dist * n_row,dist):
        x_array.append(x)
        y_array.append(y)

fi.reinitialize_flow_field(layout_array=(x_array,y_array))
fi.calculate_wake()
"""
        % n_row
    )

    test_code_calc = """
fi.calculate_wake()
"""
    num_iter = 10
    t1 = timeit.repeat(stmt=test_code_calc, setup=setup_code, number=num_iter, repeat=2)
    run_time = (min(t1) / num_iter) * 1000.0
    results[(n_row, "calc")] = run_time

    test_code_re = """
fi.reinitialize_flow_field()
"""
    t1 = timeit.repeat(stmt=test_code_re, setup=setup_code, number=num_iter, repeat=2)
    run_time = (min(t1) / num_iter) * 1000.0
    results[(n_row, "re")] = run_time

for n_row in n_row_to_test:
    print("-------------------------------------------------------------------")
    print(
        "Timing results %d turbine wind farm which is %d x %d m in size"
        % (n_row * n_row, n_row * dist, n_row * dist)
    )
    print("---- Calc wake runs in %.1f ms " % results[(n_row, "calc")])
    print("---- Re - flow runs in %.1f ms " % results[(n_row, "re")])
