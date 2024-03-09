
import copy
import time
import warnings

import numpy as np
from linux_perf import perf

from floris.core import Core


wd_grid, ws_grid = np.meshgrid(
    np.arange(0, 360.0, 5),     # wind directions
    np.arange(8.0, 12.0, 0.2),  # wind speeds
    indexing="ij"
)
WIND_DIRECTIONS = wd_grid.flatten()
WIND_SPEEDS = ws_grid.flatten()
TURBULENCE_INTENSITIES = np.ones_like(WIND_DIRECTIONS) * 0.1
N_FINDEX = len(WIND_DIRECTIONS)

N_TURBINES = 3
X_COORDS, Y_COORDS = np.meshgrid(
    5.0 * 126.0 * np.arange(0, N_TURBINES, 1),
    5.0 * 126.0 * np.arange(0, N_TURBINES, 1),
)
X_COORDS = X_COORDS.flatten()
Y_COORDS = Y_COORDS.flatten()

N_ITERATIONS = 20


def run_floris(input_dict):
    try:
        start = time.perf_counter()
        core = Core.from_dict(copy.deepcopy(input_dict.core))
        core.initialize_domain()
        core.steady_state_atmospheric_condition()
        end = time.perf_counter()
        return end - start
    except KeyError:
        # Catch the errors when an invalid wake model was given because the model
        # was not yet implemented
        return -1.0


def time_profile(input_dict):

    # Run once to initialize Python and memory
    run_floris(input_dict)

    times = np.zeros(N_ITERATIONS)
    for i in range(N_ITERATIONS):
        times[i] = run_floris(input_dict)

    return np.sum(times) / N_ITERATIONS


def test_time_jensen_jimenez(sample_inputs_fixture):
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = "jensen"
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = "jimenez"
    return time_profile(sample_inputs_fixture)


def test_time_gauss(sample_inputs_fixture):
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = "gauss"
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = "gauss"
    return time_profile(sample_inputs_fixture)


def test_time_gch(sample_inputs_fixture):
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = "gauss"
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = "gauss"
    sample_inputs_fixture.core["wake"]["enable_transverse_velocities"] = True
    sample_inputs_fixture.core["wake"]["enable_secondary_steering"] = True
    sample_inputs_fixture.core["wake"]["enable_yaw_added_recovery"] = True
    return time_profile(sample_inputs_fixture)


def test_time_cumulative(sample_inputs_fixture):
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = "cc"
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = "gauss"
    return time_profile(sample_inputs_fixture)


def memory_profile(input_dict):
    # Run once to initialize Python and memory
    core = Core.from_dict(copy.deepcopy(input_dict.core))
    core.initialize_domain()
    core.steady_state_atmospheric_condition()

    with perf():
        for i in range(N_ITERATIONS):
            core = Core.from_dict(copy.deepcopy(input_dict.core))
            core.initialize_domain()
            core.steady_state_atmospheric_condition()

    print(
        "Size of one data array: "
        f"{64 * N_FINDEX * N_TURBINES * 25 / (1000 * 1000)} MB"
    )


def test_mem_jensen_jimenez(sample_inputs_fixture):
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = "jensen"
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = "jimenez"
    memory_profile(sample_inputs_fixture)


if __name__=="__main__":
    warnings.filterwarnings('ignore')

    from conftest import SampleInputs
    sample_inputs = SampleInputs()

    sample_inputs.core["farm"]["layout_x"] = X_COORDS
    sample_inputs.core["farm"]["layout_y"] = Y_COORDS
    sample_inputs.core["flow_field"]["wind_directions"] = WIND_DIRECTIONS
    sample_inputs.core["flow_field"]["wind_speeds"] = WIND_SPEEDS
    sample_inputs.core["flow_field"]["turbulence_intensities"] = TURBULENCE_INTENSITIES

    print()
    print("### Memory profiling")
    test_mem_jensen_jimenez(sample_inputs)

    print()
    print("### Performance profiling")
    time_jensen = test_time_jensen_jimenez(sample_inputs)
    time_gauss = test_time_gauss(sample_inputs)
    time_gch = test_time_gch(sample_inputs)
    # TODO: reenable this after the cc model is fixed with multiturbine
    # time_cc = test_time_cumulative(sample_inputs)

    # print("{:.4f} {:.4f} {:.4f} {:.4f}".format(time_jensen, time_gauss, time_gch, time_cc))
    print("{:.4f} {:.4f} {:.4f}".format(time_jensen, time_gauss, time_gch))
