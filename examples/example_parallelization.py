import mpi4py.futures as mp
import numpy as np
import pandas as pd
from time import perf_counter as timerpc

from floris.tools import FlorisInterface
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR


def _optimize_serial(fi, wind_directions):
    # Reinitialize to wind directions
    fi.reinitialize(wind_directions=wind_directions)
    yaw_opt = YawOptimizationSR(fi, exploit_layout_symmetry=False)
    return yaw_opt.optimize()


if __name__ == "__main__":
    # User settings
    n_splits = 30
    n_cores = None
    
    # Define FLORIS object
    n_turbs = 10
    fi = FlorisInterface("gch.yaml")
    fi.reinitialize(
        layout_x=np.arange(n_turbs) * 5.0 * 126.0,
        layout_y=np.zeros(n_turbs),
        wind_speeds=np.arange(6.0, 13.0, 1.0),
        wind_directions=np.arange(0.0, 360.0, 2.0),
    )

    # Prepare the input arguments parallel execution
    wind_directions = fi.floris.flow_field.wind_directions
    wind_direction_splits = np.array_split(wind_directions, n_splits)
    multiargs = []
    for wd_split in wind_direction_splits:
        multiargs.append((fi.copy(), wd_split))

    # Execute in parallel
    if n_cores is None:
        n_cores = mp.cpu_count()
        print("Automatically detecting number of cores on your machine...")

    # Optimize yaw angles using multiprocessing Pool
    start_time = timerpc()
    with mp.MPIPoolExecutor(n_cores) as p:
        df_opt_splits = p.starmap(_optimize_serial, multiargs)
    print("Optimization finished in {:.2f} seconds.".format(timerpc() - start_time))

    # Merge solutions and print output
    df_opt = pd.concat(df_opt_splits, axis=0)
    print(df_opt)