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


# Parameters
n_row = 5
n_iter = 1
n_ws = 100
dist = 126 * 7


print(
    "****** Timing %d turbine wind farm which is %d x %d m in size over %d wind speeds / direction"
    % (n_row * n_row, n_row * dist, n_row * dist, n_ws)
)

setup_code = """
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
ws_array = np.random.uniform(5,12,%d)
wd_array = np.random.uniform(0,359,%d)
""" % (
    n_row,
    n_ws,
    n_ws,
)

test_code_calc = """
wfct.batch_process.batch_simulate(fi, ws_array, wd_array)
"""
num_iter = n_iter
t1 = timeit.repeat(stmt=test_code_calc, setup=setup_code, number=num_iter, repeat=1)
print(t1)
