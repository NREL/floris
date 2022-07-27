import time
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from floris.tools import FlorisInterface


# Initialize the FLORIS interface fi
file_dir = Path(__file__).resolve().parent
fi = FlorisInterface(file_dir / "inputs" / "gch.yaml")

D = 126.0  # rotor diameter for the NREL 5MW


def change_parameter_space(n_dir, n_speed, grid_dim):
    # Create the parameter space
    wind_directions = np.linspace(0, 360, n_dir)
    wind_speeds = np.linspace(0, 40, n_speed)
    layout_x = np.array([list(range(grid_dim)) for _ in range(grid_dim)]).flatten() * 6 * D
    layout_y = np.array([list(range(grid_dim)) for _ in range(grid_dim)]).T.flatten() * 4 * D

    # Reinitialize and calculate wake
    start = time.perf_counter()
    fi.reinitialize(
        # layout_x=layout_x,
        # layout_y=layout_y,
        layout=(layout_x, layout_y),
        wind_directions=wind_directions,
        wind_speeds=wind_speeds,
    )
    end = time.perf_counter()
    fi.calculate_wake()
    end_2 = time.perf_counter()

    # return reinitialize timing, wake timing, and total timing
    return end - start, end_2 - end, end_2 - start


n_dir = range(10, 361, 30)
n_speed = range(5, 41, 10)
grid_dim = range(1, 20, 3)

combos = itertools.product(n_dir, n_speed, grid_dim)

speed = []
for i, j, k in reversed(list(combos)):
    init, wake, total = change_parameter_space(i, j, k)
    speed.append([i, j, k * k, init, wake, total])
    print(i, j, k)
    print(init)
    print(wake)
    print(total)
    # time.sleep(3)
    break

df = pd.DataFrame(
    speed,
    columns=[
        "n_wind_directions",
        "n_wind_speeds",
        "n_turbines",
        "reinitialize_time",
        "calculate_wake_time",
        "total_run_time",
    ],
)
df.to_csv("python_timing.csv")
