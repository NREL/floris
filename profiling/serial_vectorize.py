
import copy
import numpy as np
import time
import matplotlib.pyplot as plt

from floris.simulation import Floris
from conftest import SampleInputs


def time_vec(input_dict):
    start = time.time()
    floris = Floris(input_dict=input_dict.floris)
    end = time.time()
    init_time = end - start

    start = time.time()
    floris.steady_state_atmospheric_condition()
    end = time.time()
    calc_time = end - start

    return init_time, calc_time


def time_serial(input_dict, wd, ws):
    init_times = np.zeros(len(wd))
    calc_times = np.zeros(len(wd))

    for i, (d, s) in enumerate(zip(wd, ws)):

        input_dict.floris["flow_field"]["wind_directions"] = [d]
        input_dict.floris["flow_field"]["wind_speeds"] = [s]

        start = time.time()
        floris = Floris(input_dict=input_dict.floris)
        end = time.time()
        init_times[i] = end - start

        start = time.time()
        floris.steady_state_atmospheric_condition()
        end = time.time()
        calc_times[i] = end - start

    return np.sum(init_times), np.sum(calc_times)

if __name__=="__main__":
    plt.figure()

    sample_inputs = SampleInputs()
    sample_inputs.floris["flow_field"]["wind_directions"] = [270.0]
    sample_inputs.floris["flow_field"]["wind_speeds"] = [8.0]
    TURBINE_DIAMETER = sample_inputs.floris["turbine"]["rotor_diameter"]

    N = 5
    simulation_size = np.arange(N)

    # 1 turbine
    vectorize_init, vectorize_calc = np.zeros(N), np.zeros(N)
    for i in range(N):
        vectorize_scaling_inputs = copy.deepcopy(sample_inputs)

        factor = (i+1) * 50
        vectorize_scaling_inputs.floris["flow_field"]["wind_directions"] = [270.0]
        vectorize_scaling_inputs.floris["flow_field"]["wind_speeds"] = factor * [8.0]

        vectorize_init[i], vectorize_calc[i] = time_vec(copy.deepcopy(vectorize_scaling_inputs))
        print("vectorize", i, vectorize_calc[i])

    serial_init, serial_calc = np.zeros(N), np.zeros(N)
    for i in range(N):
        serial_scaling_inputs = copy.deepcopy(sample_inputs)

        factor = (i+1) * 50
        wind_directions = [270.0]
        wind_speeds = factor * [8.0]

        serial_init[i], serial_calc[i] = time_serial(copy.deepcopy(serial_scaling_inputs), wind_directions, wind_speeds)
        print("serial", i, serial_calc[i])

    plt.plot(simulation_size, vectorize_init, 'b--', label='vectorize init - 1 turbine')
    plt.plot(simulation_size, vectorize_calc, 'bo-', label='vectorize calc - 1 turbine')
    plt.plot(simulation_size, serial_init, 'g--', label='serial init - 1 turbine')
    plt.plot(simulation_size, serial_calc, 'go-', label='serial calc - 1 turbine')


    # More than 1 turbine
    n_turbines = 10
    sample_inputs.floris["farm"]["layout_x"] = [5 * TURBINE_DIAMETER * j for j in range(n_turbines)]
    sample_inputs.floris["farm"]["layout_y"] = n_turbines * [0.0]

    vectorize_init, vectorize_calc = np.zeros(N), np.zeros(N)
    for i in range(N):
        vectorize_scaling_inputs = copy.deepcopy(sample_inputs)

        factor = (i+1) * 50
        vectorize_scaling_inputs.floris["flow_field"]["wind_speeds"] = factor * [8.0]
        vectorize_scaling_inputs.floris["flow_field"]["wind_directions"] = [270.0]

        vectorize_init[i], vectorize_calc[i] = time_vec(copy.deepcopy(vectorize_scaling_inputs))
        print("vectorize", i, vectorize_calc[i])

    serial_init, serial_calc = np.zeros(N), np.zeros(N)
    for i in range(N):
        serial_scaling_inputs = copy.deepcopy(sample_inputs)

        factor = (i+1) * 50
        speeds = factor * [8.0]
        wind_directions = [270.0]

        serial_init[i], serial_calc[i] = time_serial(copy.deepcopy(serial_scaling_inputs), wind_directions, speeds)
        print("serial", i, serial_calc[i])

    plt.plot(simulation_size, vectorize_init, 'c--', label='vectorize init - 10 turbine')
    plt.plot(simulation_size, vectorize_calc, 'co-', label='vectorize calc - 10 turbine')
    plt.plot(simulation_size, serial_init, 'y--', label='serial init - 10 turbine')
    plt.plot(simulation_size, serial_calc, 'yo-', label='serial calc - 10 turbine')

    ## Show plots
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()
