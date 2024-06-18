"""Example: Layout optimization by generating a HeterogeneousMap

This example repeats the optimization in the previous example, but uses a HeterogeneousMap

"""

import matplotlib.pyplot as plt
import numpy as np

from floris import (
    FlorisModel,
    WindResourceGrid,
    WindRose,
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
    layout_y = np.array([1500,1800])

    # Define an optimization boundary within the grid
    boundaries = [(0.0, 0.0), (1000.0, 1000.0), (0.0, 2000.0), (0.0, 0.0)]

    # Get the WindRoseByTurbine object
    wind_rose_by_turbine = wrg.get_wind_rose_by_turbine(layout_x, layout_y)

    # Set up the FlorisModel
    fmodel = FlorisModel("../inputs/gch.yaml")

    # Get the wind rose from the 0th turbine and use this for the farm wind rose
    wind_rose = wind_rose_by_turbine.wind_roses[0]

    # Set up the FLORIS model to use the above layout and wind rose
    fmodel.set(layout_x=layout_x, layout_y=layout_y, wind_data=wind_rose)

    # Now generate a heterogeneous map from the WindResourceGrid
    het_map = wrg.get_heterogeneous_map(fmodel)

    # Add the heterogeneous map back to the wind rose
    wind_rose = WindRose(
        wind_directions=wind_rose.wind_directions,
        wind_speeds=wind_rose.wind_speeds,
        ti_table=wind_rose.ti_table,
        freq_table=wind_rose.freq_table,
        heterogeneous_map=het_map
    )

    # Set up the FLORIS model to use the update wind rose
    fmodel.set(wind_data=wind_rose)

    # Visualize the het_map
    fig, axarr = plt.subplots(2,6, figsize=(16, 12))
    axarr = axarr.flatten()

    for i, ax in enumerate(axarr):
        wd = het_map.wind_directions[i]
        het_map.plot_single_speed_multiplier(wind_direction=wd, wind_speed=8.0, ax=ax)
        ax.set_title(f"Wind Direction: {wd}")


    # Redo the optimization from the previous example using the het_map

    # Perform the optimization
    distance_pmf = None

    layout_opt = LayoutOptimizationRandomSearch(
        fmodel,
        boundaries,
        min_dist_D=5.,
        seconds_per_iteration=10,
        total_optimization_seconds=60.,
        distance_pmf=distance_pmf,
    )
    layout_opt.describe()
    layout_opt.plot_distance_pmf()

    layout_opt.optimize()

    layout_opt.plot_layout_opt_results()

    layout_opt.plot_progress()

    plt.show()
