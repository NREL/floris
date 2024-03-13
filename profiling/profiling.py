
# import re
# import sys
# import time
# import cProfile
# from copy import deepcopy

import copy

from conftest import SampleInputs

from floris.core import Core


def run_floris():
    core = Core.from_file("examples/example_input.yaml")
    return core

if __name__=="__main__":
    # if len(sys.argv) > 1:
    #     floris = Floris(sys.argv[1])
    # else:
    #     floris = Floris("example_input.yaml")
    # floris.farm.flow_field.calculate_wake()

    # start = time.time()
    # cProfile.run('re.compile("floris.steady_state_atmospheric_condition()")')
    # end = time.time()
    # print(start, end, end - start)

    sample_inputs = SampleInputs()

    sample_inputs.core["wake"]["model_strings"]["velocity_model"] = "gauss"
    sample_inputs.core["wake"]["model_strings"]["deflection_model"] = "gauss"
    sample_inputs.core["wake"]["enable_secondary_steering"] = True
    sample_inputs.core["wake"]["enable_yaw_added_recovery"] = True
    sample_inputs.core["wake"]["enable_transverse_velocities"] = True

    N_TURBINES = 100
    N_FINDEX = 72 * 25  # Size of a characteristic wind rose

    TURBINE_DIAMETER = sample_inputs.core["farm"]["turbine_type"][0]["rotor_diameter"]
    sample_inputs.core["farm"]["layout_x"] = [5 * TURBINE_DIAMETER * i for i in range(N_TURBINES)]
    sample_inputs.core["farm"]["layout_y"] = [0.0 for i in range(N_TURBINES)]

    sample_inputs.core["flow_field"]["wind_directions"] = N_FINDEX * [270.0]
    sample_inputs.core["flow_field"]["wind_speeds"] = N_FINDEX * [8.0]
    sample_inputs.core["flow_field"]["turbulence_intensities"] = N_FINDEX * [0.06]

    N = 1
    for i in range(N):
        core = Core.from_dict(copy.deepcopy(sample_inputs.core))
        core.initialize_domain()
        core.steady_state_atmospheric_condition()
