"""Example: Layout optimization using WindRoseWRG

Perform a layout optimization using a random search algorithm and the WindRoseByTurbine. Note that
within the optimization, different layouts will be tested and that when the layout in the
FlorisModel is updated, the wind roses for each turbine are also updated.

The optimization problem is a triangle boundary area.  Since in the WRG file, east wind speed
increases with increasing x, while north wind speeds increase with increasing y, the optimization
should place turbines in the east vertex of the triangle and north vertex of the triangle to
maximize the wind speed.
"""

import matplotlib.pyplot as plt
import numpy as np

from floris import (
    FlorisModel,
    WindRoseWRG,
)
from floris.optimization.layout_optimization.layout_optimization_random_search import (
    LayoutOptimizationRandomSearch,
)


if __name__ == "__main__":
    # Initialize the WindRoseWRG object with wind speeds every 2 m/s and fixed ti of 6%.  Specify
    # a wd_step of 5 degrees, which implies upsampling from wrg's 30 degree sectors to 12
    # degree sectors
    wind_rose_wrg = WindRoseWRG("wrg_example.wrg",
                                wd_step=12.0,
                                wind_speeds=np.arange(0, 21, 3.0), # Use a sparser range of speeds
                                ti_table=0.06)

    # Define an optimization boundary within the grid that is a parallelogram
    width = 200.0  # You can adjust this value as needed
    boundaries = [(0.0, 0.0), (width, 0.0), (1000.0, 2000.0), (1000.0 - width, 2000.0), (0.0, 0.0)]

    # Select and initial layout in the corners of the boundary
    layout_x = np.array([0, 1000 - width])
    layout_y = np.array([0, 2000])

    # Set up the FlorisModel
    fmodel = FlorisModel("../inputs/gch.yaml")
    fmodel.set(layout_x=layout_x, layout_y=layout_y, wind_data=wind_rose_wrg)

    # Set the layout optimization to run for 60 seconds, with 15 second iterations
    layout_opt = LayoutOptimizationRandomSearch(
        fmodel,
        boundaries,
        min_dist_D=5.0,
        seconds_per_iteration=25,
        total_optimization_seconds=100.0,
    )

    layout_opt.optimize()
    x_initial, y_initial, x_opt_wrg, y_opt_wrg = layout_opt._get_initial_and_final_locs()

    # Repeat the optimization using only a single wind rose
    wind_rose = wind_rose_wrg.get_wind_rose_at_point(0, 0)
    fmodel.set(layout_x=layout_x, layout_y=layout_y, wind_data=wind_rose)

    layout_opt = LayoutOptimizationRandomSearch(
        fmodel,
        boundaries,
        min_dist_D=5.0,
        seconds_per_iteration=25,
        total_optimization_seconds=100.0,
    )
    layout_opt.optimize()
    _, _, x_opt_wr, y_opt_wr = layout_opt._get_initial_and_final_locs()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    layout_opt.plot_layout_opt_boundary(ax=ax)
    ax.scatter(x_initial, y_initial, label="Initial Layout", s=60, color="k", marker="s")
    ax.scatter(
        x_opt_wr, y_opt_wr, label="Optimized Layout (Single Wind Rose)", s=40, color="b", marker="h"
    )
    ax.scatter(x_opt_wrg, y_opt_wrg, label="Optimized Layout (WRG)", s=20, color="r", marker="o")
    ax.legend()

    plt.show()
