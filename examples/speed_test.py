import time
from pathlib import Path

import numpy as np

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
        layout_x=layout_x,
        layout_y=layout_y,
        wind_directions=wind_directions,
        wind_speeds=wind_speeds,
    )
    end = time.perf_counter()
    fi.calculate_wake()
    end_2 = time.perf_counter()

    # return reinitialize timing, wake timing, and total timing
    return end - start, end_2 - end, end_2 - start


# n_dir = range(10, 361, 30)
# n_speed = range(5, 41, 10)
# grid_dim = range(1, 17, 3)

# combos = itertools.product(n_dir, n_speed, grid_dim)

# speed = []
# for i, j, k in combos:
#     init, wake, total = change_parameter_space(i, j, k)
#     speed.append([i, j, k * k, init, wake, total])
#     print(i, j, k)
#     print(init)
#     print(wake)
#     print(total)
# time.sleep(3)

# df = pd.DataFrame(
#     speed,
#     columns=[
#         "n_wind_directions",
#         "n_wind_speeds",
#         "n_turbines",
#         "reinitialize_time",
#         "calculate_wake_time",
#         "total_run_time",
#     ],
# )
# df.to_csv("python_timing_.csv")

# Use a case that takes just long enough to identify meaningful speedups
speed = []
i, j, k = 40, 5, 13
for _ in range(10):
    init, wake, total = change_parameter_space(i, j, k)
    speed.append([init, wake, total])

speed = np.array(speed)
res = speed.mean(axis=0)
print("Init Time | Calc Time | Total Time")
print(f"{res[0]:9.6f} | {res[1]:8.6f} | {res[2]:10.6f}")

# Init Time | Calc Time | Total Time
#  1.403899 | 28.699313 |  30.103212  <- Base Case with v3.1 as of 9am, 7/28/22
#  1.477987 | 29.551348 |  31.029335  <- Base Case with v3.1 as of 3pm, 8/15/22
# Wake Deflection Modifications
#  1.457128 | 28.012114 |  29.469241  <- w/ _initial_wake_expansion as of 3:30pm 8/15/22
#  1.420906 | 27.515838 |  28.936744  <- dropping reassignment from np.array(boolean operation) as of 3:35 8/15/22
#  1.536173 | 28.122597 |  29.658770  <- common variable for reused calculations and more booleans like the above as of 3:55 8/15/22
#  1.477387 | 27.615849 |  29.093236  <- gamma paramaters to a single function and undo slow change from above as of 4:25 8/15/22
#  1.450766 | 27.384075 |  28.834841  <- vortex calculation addition
#  1.366653 | 23.887163 |  25.253816  <- update vortex calculation and apply it to tranverse velocity as of 11:28 8/16/22
# Wake Velocity Modifications
#  1.423692 | 25.375699 |  26.799391  <- remove np.array(boolean) in favor of (boolean)
#  1.377274 | 23.958512 |  25.335786  <- NumExpr applied to the gaussian function
#  1.370956 | 24.076823 |  25.447779  <- Crespo numexpr addition 2:20 8/16/22
# Computer restart
#  1.351354 | 22.771568 |  24.122922  <- no changes
#  1.320245 | 22.193191 |  23.513436  <- split up far wake function and fully NumExpr rC() 2:30 8/16/22
#  1.313820 | 20.925804 |  22.239624  <- NumExpr to the vortex calculations
#  1.344035 | 20.176910 |  21.520945  <- More NumExpr 3:30 8/16/22
#  1.350094 | 19.661327 |  21.011421  <- remove np.array(boolean) in favor of (boolean) 4:00 8/16/22
