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

from floris.simulation import Floris
from conftest import SampleInputs

def run_floris():
    floris = Floris.from_json("examples/example_input.json")
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
    floris = Floris(input_dict=sample_inputs.floris)

    factor = 100
    TURBINE_DIAMETER = sample_inputs.floris["turbine"]["rotor_diameter"]
    sample_inputs.floris["farm"]["layout_x"] = [5 * TURBINE_DIAMETER * i for i in range(factor)]
    sample_inputs.floris["farm"]["layout_y"] = [0.0 for i in range(factor)]


    factor = 10
    sample_inputs.floris["farm"]["wind_directions"] = factor * [270.0]
    sample_inputs.floris["farm"]["wind_speeds"] = factor * [8.0]
    floris = Floris(input_dict=sample_inputs.floris)
    floris.steady_state_atmospheric_condition()