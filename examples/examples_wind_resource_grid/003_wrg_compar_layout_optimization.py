"""Example: Layout optimization with WindRoseWRG comparison

This example compares a layout optimization using a WindRoseWRG

TODO: More explanation and clean this up

"""

import matplotlib.pyplot as plt
import numpy as np

from floris import (
    FlorisModel,
    WindRose,
    WindRoseWRG,
)
from floris.optimization.layout_optimization.layout_optimization_random_search import (
    LayoutOptimizationRandomSearch,
)


if __name__ == "__main__":
    # Parameters
    seconds_per_iteration = 30.0
    total_optimization_seconds = 120.0
    min_dist_D = 0.1
    use_dist_based_init = False

    # Initialize the WindRoseWRG object with wind speeds every 2 m/s and fixed ti of 6%.  Specify
    # a wd_step of 4 degrees, which implies upsampling from wrg's 90 degree sectors to 12
    # degree sectors
    wind_rose_wrg = WindRoseWRG(
        "wrg_example.wrg",
        wd_step=2.0,
        wind_speeds=np.arange(0, 21, 1.0),  # Use a sparser range of speeds
        ti_table=0.06,
    )

    # Define an optimization boundary within the grid that is a parallelogram
    buffer = 100.0
    width = 200.0  # You can adjust this value as needed

    boundaries = [
        (buffer, buffer),
        (1000 - buffer, buffer),
        (1000 - buffer, 2000 - buffer),
        (buffer, 2000 - buffer),
        (buffer, buffer),
    ]

    # Select and initial layout in the corners of the boundary
    # layout_x = np.array([buffer, 1000 - width - buffer])
    # layout_y = np.array([buffer, 2000 - buffer])
    layout_x = np.array([500, 1000 - buffer])
    layout_y = np.array([900, 2000 - buffer])

    ##########################
    # Set up the FlorisModel
    fmodel = FlorisModel("../inputs/gch.yaml")

    ##########################
    # Use the get_heterogeneous_map method to generate a WindRose that represents
    # the information in the WindRoseWRG, rather than a set of WindRose objects
    # but as a  single WindRose object (for one location) and a HeterogeneousMap
    # the describes the speed up information per direction across the domain
    # This will allow running the optimization for a single wind speed while still
    # accounting for the difference in wind speeds in location by direction
    wind_rose_het = wind_rose_wrg.get_heterogeneous_wind_rose(
        fmodel=fmodel,
        x_loc=0.0,
        y_loc=0.0,
        representative_wind_speed=9.0,
    )

    # Pull out the hetergenous plot to show the underlying speedups
    het_map = wind_rose_het.heterogeneous_map
    wind_direction_to_plot = [0.0, 10.0, 45.0, 75.0, 90.0, 180.0]

    # Show the het_map for a few wind directions
    fig, axarr = plt.subplots(1, len(wind_direction_to_plot), figsize=(16, 5))
    axarr = axarr.flatten()
    for i, wd in enumerate(wind_direction_to_plot):
        het_map.plot_single_speed_multiplier(
            wind_direction=wd,
            wind_speed=8.0,
            ax=axarr[i],
            show_colorbar=True,
        )

        axarr[i].set_title(f"Wind Direction: {wd}")

    # ##########################
    # Run the optimization as before with the WindRoseWRG first
    fmodel.set(layout_x=layout_x, layout_y=layout_y, wind_data=wind_rose_wrg)

    # Set the layout optimization
    layout_opt = LayoutOptimizationRandomSearch(
        fmodel,
        boundaries,
        min_dist_D=min_dist_D,
        seconds_per_iteration=seconds_per_iteration,
        total_optimization_seconds=total_optimization_seconds,
        use_dist_based_init=use_dist_based_init,
    )

    layout_opt.optimize()
    x_initial, y_initial, x_opt_wrg, y_opt_wrg = layout_opt._get_initial_and_final_locs()

    # Grab the log array
    objective_log_array_wrg = np.array(layout_opt.objective_candidate_log)

    # Normalize
    objective_log_array_wrg = objective_log_array_wrg / np.max(objective_log_array_wrg)

    print("=====================================")
    print("Objective log array (WRG):")
    print(objective_log_array_wrg.shape)
    print(objective_log_array_wrg)

    # ##########################
    # Repeat using wind_rose_het
    fmodel.set(layout_x=layout_x, layout_y=layout_y, wind_data=wind_rose_het)

    # Set the layout optimization
    layout_opt = LayoutOptimizationRandomSearch(
        fmodel,
        boundaries,
        min_dist_D=min_dist_D,
        seconds_per_iteration=seconds_per_iteration,
        total_optimization_seconds=total_optimization_seconds,
        use_dist_based_init=use_dist_based_init,
    )

    layout_opt.optimize()
    _, _, x_opt_het, y_opt_het = layout_opt._get_initial_and_final_locs()

    # Grab the log array
    objective_log_array_het = np.array(layout_opt.objective_candidate_log)

    # Normalize
    objective_log_array_het = objective_log_array_het / np.max(objective_log_array_het)

    # ##########################
    # Repeat using single wind rose (with het)
    wind_rose = wind_rose_wrg.get_wind_rose_at_point(0, 0)
    fmodel = FlorisModel("../inputs/gch.yaml")
    fmodel.set(layout_x=layout_x, layout_y=layout_y, wind_data=wind_rose)

    # Set the layout optimization
    layout_opt = LayoutOptimizationRandomSearch(
        fmodel,
        boundaries,
        min_dist_D=min_dist_D,
        seconds_per_iteration=seconds_per_iteration,
        total_optimization_seconds=total_optimization_seconds,
        use_dist_based_init=use_dist_based_init,
    )

    layout_opt.optimize()
    _, _, x_opt_wr, y_opt_wr = layout_opt._get_initial_and_final_locs()

    # Grab the log array
    objective_log_array_wr = np.array(layout_opt.objective_candidate_log)

    # Normalize
    objective_log_array_wr = objective_log_array_wr / np.max(objective_log_array_wr)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    layout_opt.plot_layout_opt_boundary(ax=ax)
    ax.scatter(x_initial, y_initial, label="Initial Layout", s=80, color="k", marker="s")
    ax.scatter(
        x_opt_wr, y_opt_wr, label="Optimized Layout (Single Wind Rose)", s=60, color="b", marker="^"
    )
    ax.scatter(x_opt_wrg, y_opt_wrg, label="Optimized Layout (WRG)", s=40, color="r", marker="o")
    ax.scatter(
        x_opt_het,
        y_opt_het,
        label="Optimized Layout (Single Wind Rose + Het)",
        s=20,
        color="g",
        marker="h",
    )
    ax.set_axis("equal")
    ax.legend()

    print("=====================================")
    print("Objective log array (HET):")
    print(objective_log_array_het.shape)
    print(objective_log_array_het)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for objective_log_array, label, color in zip(
        [objective_log_array_wr, objective_log_array_wrg, objective_log_array_het],
        ["WR", "WRG", "Het"],
        ["b", "r", "g"],
    ):
        ax.plot(
            np.arange(len(objective_log_array)),
            np.log10(objective_log_array * 100.0),
            label=label,
            color=color,
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective")
        ax.legend()

    plt.show()
