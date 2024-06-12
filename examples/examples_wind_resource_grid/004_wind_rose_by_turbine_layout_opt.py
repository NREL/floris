"""Example: Wind Rose By Turbine With Layout Optimization

To Be Written

"""

import matplotlib.pyplot as plt
import numpy as np

from floris import (
    FlorisModel,
    WindResourceGrid,
    WindRoseByTurbine,
)
from floris.optimization.layout_optimization.layout_optimization_random_search import (
    LayoutOptimizationRandomSearch,
)


if __name__ == '__main__':
    # Read the WRG file
    wrg = WindResourceGrid("wrg_example.wrg")

    # Select and initial layout
    layout_x = np.array([0,0])
    layout_y = np.array([100,200])

    # Define an optimization boundary within the grid
    boundaries = [(0.0, 0.0), (1000.0, 1000.0), (0.0, 2000.0), (0.0, 0.0)]

    # Get the WindRoseByTurbine object
    wind_rose_by_turbine = wrg.get_wind_rose_by_turbine(layout_x, layout_y)

    # Set up the FlorisModel
    fmodel = FlorisModel("../inputs/gch.yaml")
    fmodel.set(layout_x=layout_x, layout_y=layout_y, wind_data=wind_rose_by_turbine)

    # Perform the optimization
    distance_pmf = None

    layout_opt = LayoutOptimizationRandomSearch(
        fmodel,
        boundaries,
        min_dist_D=5.,
        seconds_per_iteration=10,
        total_optimization_seconds=60.,
        distance_pmf=distance_pmf,
        wind_resource_grid=wrg
    )
    layout_opt.describe()
    layout_opt.plot_distance_pmf()

    layout_opt.optimize()

    layout_opt.plot_layout_opt_results()

    layout_opt.plot_progress()

    plt.show()
