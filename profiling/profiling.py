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

# import re
# import sys
# import time
# import cProfile
# from copy import deepcopy

import copy
from floris.simulation import Floris
from conftest import SampleInputs

def run_floris():
    floris = Floris.from_file("examples/example_input.yaml")
    return floris

if __name__=="__main__":
    # if len(sys.argv) > 1:
    #     floris = Floris(sys.argv[1])
    # else:
    #     floris = Floris("example_input.json")
    # floris.farm.flow_field.calculate_wake()

    # start = time.time()
    # cProfile.run('re.compile("floris.steady_state_atmospheric_condition()")')
    # end = time.time()
    # print(start, end, end - start)

    sample_inputs = SampleInputs()

    sample_inputs.floris["wake"]["model_strings"]["velocity_model"] = "gauss"
    sample_inputs.floris["wake"]["model_strings"]["deflection_model"] = "gauss"
    sample_inputs.floris["wake"]["enable_secondary_steering"] = True
    sample_inputs.floris["wake"]["enable_yaw_added_recovery"] = True
    sample_inputs.floris["wake"]["enable_transverse_velocities"] = True

    N_TURBINES = 100
    N_WIND_DIRECTIONS = 72
    N_WIND_SPEEDS = 25

    TURBINE_DIAMETER = sample_inputs.floris["farm"]["turbine_type"][0]["rotor_diameter"]
    sample_inputs.floris["farm"]["layout_x"] = [5 * TURBINE_DIAMETER * i for i in range(N_TURBINES)]
    sample_inputs.floris["farm"]["layout_y"] = [0.0 for i in range(N_TURBINES)]

    sample_inputs.floris["flow_field"]["wind_directions"] = N_WIND_DIRECTIONS * [270.0]
    sample_inputs.floris["flow_field"]["wind_speeds"] = N_WIND_SPEEDS * [8.0]

    N = 1
    for i in range(N):
        floris = Floris.from_dict(copy.deepcopy(sample_inputs.floris))
        floris.initialize_domain()
        floris.steady_state_atmospheric_condition()
