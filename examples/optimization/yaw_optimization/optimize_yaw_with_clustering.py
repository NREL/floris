import os
from time import perf_counter as timerpc

import numpy as np
import matplotlib.pyplot as plt

import floris.tools as wfct
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import (
    YawOptimizationSR,
)


def load_floris():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    fi = wfct.floris_interface.FlorisInterface(
        os.path.join(file_dir, "../../example_input.json")
    )
    layout_x = [
        1512.0,
        1890.0,
        2646.0,
        0.0,
        378.0,
        3402.0,
        378.0,
        882.0,
        1134.0,
        2394.0,
        2646.0,
        2268.0,
        504.0,
        2898.0,
        756.0,
    ]
    layout_y = [
        3024.0,
        3024.0,
        1512.0,
        3276.0,
        126.0,
        756.0,
        882.0,
        2898.0,
        1764.0,
        3024.0,
        2142.0,
        630.0,
        3150.0,
        1638.0,
        1008.0,
    ]
    fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))
    return fi


def plot_hor_slice(fi):
    hor_plane = fi.get_hor_plane()
    fig, ax = plt.subplots()
    wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
    return fig, ax


if __name__ == "__main__":
    # Load FLORIS
    fi = load_floris()
    num_turbs = len(fi.layout_x)

    print("Running FLORIS with no yaw misalignment...")
    fi.calculate_wake()
    power_initial = fi.get_farm_power()

    # =============================================================================
    # print("Plotting the FLORIS flowfield...")
    # # =============================================================================
    # fig, ax = plot_hor_slice(fi)
    # ax.set_title("Baseline Case for U = 8 m/s, Wind Direction = 270$^\\circ$")

    # =============================================================================
    print("Finding optimal yaw angles in FLORIS without clustering...")
    # =============================================================================
    # Instantiate the Serial Optimization (SR) Optimization object. This optimizer
    # uses the Serial Refinement approach from Fleming et al. to quickly converge
    # close to the optimal solution in a minimum number of function evaluations.
    # Then, it will refine the optimal solution using the SciPy minimize() function.
    # First, find optimal conditions without clustering.
    start_time_full = timerpc()
    yaw_opt = YawOptimizationSR(
        fi=fi,
        yaw_angles_baseline=np.zeros(num_turbs),  # Yaw angles for baseline case
        minimum_yaw_angle=0.0,  # Lower bound for the yaw angle for all turbines
        maximum_yaw_angle=20.0,  # Upper bound for the yaw angle for all turbines
        include_unc=False,  # No wind direction variability in floris simulations
        exclude_downstream_turbines=True,  # Exclude downstream turbines automatically
        cluster_turbines=False,  # Without clustering
    )
    yaw_angles_full = yaw_opt.optimize()  # Perform optimization
    end_time_full = timerpc()

    # =============================================================================
    print("Finding optimal yaw angles in FLORIS with clustering...")
    # =============================================================================
    # Now find opimal conditions with clustering.
    start_time_clust = timerpc()
    yaw_opt = YawOptimizationSR(
        fi=fi,
        yaw_angles_baseline=np.zeros(num_turbs),  # Yaw angles for baseline case
        minimum_yaw_angle=0.0,  # Lower bound for the yaw angle for all turbines
        maximum_yaw_angle=20.0,  # Upper bound for the yaw angle for all turbines
        include_unc=False,  # No wind direction variability in floris simulations
        exclude_downstream_turbines=True,  # Exclude downstream turbines automatically
        cluster_turbines=True,  # With clustering
        cluster_wake_slope=0.30,
    )
    yaw_angles_clust = yaw_opt.optimize()  # Perform optimization
    end_time_clust = timerpc()

    # Plot clusters in the farm
    yaw_opt.plot_clusters()

    # Assign yaw angles and calculate wake using full farm solution
    fi.calculate_wake(yaw_angles=yaw_angles_full)
    power_opt_full = fi.get_farm_power()

    # Assign yaw angles and calculate wake using clustered solution
    fi.calculate_wake(yaw_angles=yaw_angles_clust)
    power_opt_clust = fi.get_farm_power()

    print("==========================================")
    print(
        "Total Power Gain (full farm)= %.3f%%"
        % (100.0 * (power_opt_full - power_initial) / power_initial)
    )
    print(
        "Total Power Gain (clustered)= %.3f%%"
        % (100.0 * (power_opt_clust - power_initial) / power_initial)
    )
    print("==========================================")

    print("==========================================")
    print("Computation time (full farm)= %.2f s" % (end_time_full - start_time_full))
    print("Computation time (clustered)= %.2f s" % (end_time_clust - start_time_clust))
    print("==========================================")

    # =============================================================================
    print("Plotting the FLORIS flowfield with yaw angles from cluster optimization...")
    # =============================================================================
    fig, ax = plot_hor_slice(fi)
    ax.set_title(
        "Optimal Wake Steering (clustered) for U = 8 m/s, Wind Direction = 270$^\\circ$"
    )
    plt.show()
