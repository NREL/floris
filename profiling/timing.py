
import copy
import time

import matplotlib.pyplot as plt
import memory_profiler
import numpy as np
from conftest import SampleInputs

from floris.simulation import Floris


def time_profile(input_dict):
    floris = Floris.from_dict(input_dict.floris)
    start = time.perf_counter()
    floris.steady_state_atmospheric_condition()
    end = time.perf_counter()
    return end - start

def internal_probe(input_dict):
    floris = Floris(input_dict=input_dict.floris)
    internal_quantity = floris.steady_state_atmospheric_condition()
    return internal_quantity

def memory_profile(input_dict):
    floris = Floris(input_dict=input_dict.floris)
    mem_usage = memory_profiler.memory_usage(
        (floris.steady_state_atmospheric_condition, (), {}),
        max_usage=True
    )
    return mem_usage

if __name__=="__main__":
    sample_inputs = SampleInputs()
    TURBINE_DIAMETER = sample_inputs.floris["turbine"]["rotor_diameter"]

    # Use Gauss models
    sample_inputs.floris["wake"]["model_strings"] = {
        "velocity_model": "gauss",
        "deflection_model": "gauss",
        "combination_model": None,
        "turbulence_model": None,
    }

    ### Time scaling

    # N = 30

    # wd_calc_time = np.zeros(N)
    # wd_size = np.zeros(N)
    # wind_direction_scaling_inputs = copy.deepcopy(sample_inputs)
    # for i in range(N):
    #     factor = (i+1) * 50
    #     wind_direction_scaling_inputs.floris["flow_field"]["wind_directions"] = factor * [270.0]
    #     wind_direction_scaling_inputs.floris["flow_field"]["wind_speeds"] = [8.0]

    #     wd_calc_time[i] = time_profile(copy.deepcopy(wind_direction_scaling_inputs))
    #     wd_size[i] = factor
    #     print("wind direction", i, wd_calc_time[i])


    # ws_calc_time = np.zeros(N)
    # ws_size = np.zeros(N)
    # wind_speed_scaling_inputs = copy.deepcopy(sample_inputs)
    # for i in range(N):
    #     factor = (i+1) * 50
    #     wind_speed_scaling_inputs.floris["flow_field"]["wind_directions"] = [270.0]
    #     wind_speed_scaling_inputs.floris["flow_field"]["wind_speeds"] = factor * [8.0]

    #     ws_calc_time[i] = time_profile(copy.deepcopy(wind_speed_scaling_inputs))
    #     ws_size[i] = factor
    #     print("wind speed", i, ws_calc_time[i])


    # turb_calc_time = np.zeros(N)
    # turb_size = np.zeros(N)
    # turbine_scaling_inputs = copy.deepcopy(sample_inputs)
    # for i in range(N):
    #     factor = (i+1) * 3
    #     turbine_scaling_inputs.floris["farm"]["layout_x"] = [
    #         5 * TURBINE_DIAMETER * j
    #         for j in range(factor)
    #     ]
    #     turbine_scaling_inputs.floris["farm"]["layout_y"] = factor * [0.0]

    #     turb_calc_time[i] = time_profile(copy.deepcopy(turbine_scaling_inputs))
    #     turb_size[i] = factor
    #     print("n turbine", i, turb_calc_time[i])


    # internal_quantity = np.zeros(N)
    # scaling_inputs = copy.deepcopy(sample_inputs)
    # for i in range(5):
    #     factor = (i+1) * 2
    #     scaling_inputs.floris["farm"]["layout_x"] = [
    #         5 * TURBINE_DIAMETER * j
    #         for j in range(factor)
    #     ]
    #     scaling_inputs.floris["farm"]["layout_y"] = factor * [0.0]
    #     factor = (i+1) * 20
    #     scaling_inputs.floris["flow_field"]["wind_directions"] = factor * [270.0]
    #     scaling_inputs.floris["flow_field"]["wind_speeds"] = factor * [8.0]

    #     internal_quantity[i] = time_profile(scaling_inputs)
    #     print("n turbine", i, internal_quantity[i])

    # plt.figure()
    # plt.plot(wd_size, wd_calc_time, 'b+-', label='wind direction')
    # plt.plot(ws_size, ws_calc_time, 'g+-', label='wind speed')
    # plt.plot(turb_size, turb_calc_time, 'r+-', label='n turbine')
    # # plt.plot(simulation_size, internal_quantity, 'b+-', label='internal quantity')
    # plt.legend(loc="upper left")
    # plt.grid(True)


    ### Timing larger sizes in each dimension

    n_wind_directions = 1
    n_wind_speeds = 1
    n_turbines = 3
    sample_inputs.floris["wake"]["model_strings"] = {
        # "velocity_model": "jensen",
        # "deflection_model": "jimenez",
        "velocity_model": "cc",
        "deflection_model": "gauss",
        "combination_model": None,
        "turbulence_model": None,
    }
    sample_inputs.floris["solver"] = {
        "type": "turbine_grid",
        "turbine_grid_points": 5
    }

    # sample_inputs.floris["wake"]["enable_transverse_velocities"] = False
    # sample_inputs.floris["wake"]["enable_secondary_steering"] = False
    # sample_inputs.floris["wake"]["enable_yaw_added_recovery"] = False
    sample_inputs.floris["flow_field"]["wind_directions"] = n_wind_directions * [270.0]
    sample_inputs.floris["flow_field"]["wind_speeds"] = n_wind_speeds * [8.0]
    sample_inputs.floris["farm"]["layout_x"] = [5 * TURBINE_DIAMETER * j for j in range(n_turbines)]
    sample_inputs.floris["farm"]["layout_y"] = n_turbines * [0.0]

    N = 1
    times = np.zeros(N)
    for i in range(N):
        print(f"Iteration {i}")
        times[i] = time_profile(copy.deepcopy(sample_inputs))
        print(f"    {times[i]}")

    print(f"Total time: {np.sum(times)}")
    print(f"Average per iteration: { np.sum(times) / N }")

    ### Memory scaling

    # N = 6
    # simulation_size = np.arange(N)

    # wd_space = np.zeros(N)
    # wind_direction_scaling_inputs = copy.deepcopy(sample_inputs)
    # for i in range(N):
    #     factor = (i+1) * 50
    #     wind_direction_scaling_inputs.floris["farm"]["wind_directions"] = factor * [270.0]
    #     wind_direction_scaling_inputs.floris["farm"]["wind_speeds"] = [8.0]

    #     wd_space[i] = memory_profile(wind_direction_scaling_inputs)
    #     print("wind direction", i, wd_space[i])


    # ws_space = np.zeros(N)
    # wind_speed_scaling_inputs = copy.deepcopy(sample_inputs)
    # for i in range(N):
    #     factor = (i+1) * 50
    #     wind_speed_scaling_inputs.floris["farm"]["wind_directions"] = [270.0]
    #     wind_speed_scaling_inputs.floris["farm"]["wind_speeds"] = factor * [8.0]

    #     ws_space[i] = memory_profile(wind_speed_scaling_inputs)
    #     print("wind speed", i, ws_space[i])


    # turb_space = np.zeros(N)
    # turbine_scaling_inputs = copy.deepcopy(sample_inputs)
    # for i in range(N):
    #     factor = (i+1) * 50
    #     turbine_scaling_inputs.floris["farm"]["layout_x"] = [
    #         5 * TURBINE_DIAMETER * j
    #         for j in range(factor)
    #     ]
    #     turbine_scaling_inputs.floris["farm"]["layout_y"] = factor * [0.0]

    #     turb_space[i] = memory_profile(turbine_scaling_inputs)
    #     print("n turbine", turb_space[i])


    # # Remove the min from each test so that each starts at 0
    # wd_space = wd_space - min(wd_space)
    # ws_space = ws_space - min(ws_space)
    # turb_space = turb_space - min(turb_space)


    # plt.figure()
    # plt.plot(simulation_size, wd_space, 'b+-', label='wind direction')
    # plt.plot(simulation_size, ws_space, 'g+-', label='wind speed')
    # plt.plot(simulation_size, turb_space, 'r+-', label='n turbine')
    # plt.legend(loc="upper left")
    # plt.grid(True)


    ### Show plots
    # plt.show()
