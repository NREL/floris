import matplotlib.pyplot as plt

from floris.tools import FlorisInterface
from floris.tools.visualization import visualize_cut_plane


if __name__ == "__main__":
    # Data
    fi = FlorisInterface("inputs/gch.yaml")
    wind_directions = [225.0, 270.0, 315.0]
    layout = [[500.0, 1000.0, 1500.0], [0.0, 0.0, 0.0]]

    # For each wind direction
    for wd in wind_directions:

        # Compute wakes
        fi.reinitialize(layout=layout, wind_directions=[wd], wind_speeds=[7.0])
        fi.calculate_wake(yaw_angles=None)

        # Get horizontal plane
        hor_plane = fi.calculate_horizontal_plane(
            height=90.0,
            # x_bounds=[250.0, 1750.0],
            # y_bounds=[-250.0, 250.0],
            yaw_angles=None,
            north_up=True,
        )

        # Plot and save figure
        figure = plt.figure()
        render_ax = figure.add_subplot(111)
        im = visualize_cut_plane(hor_plane, ax=render_ax)
        render_ax.set_title("wind_direction={}°".format(wd))
        plt.show()
