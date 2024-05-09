from __future__ import annotations

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial._qhull
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import ConvexHull

from floris.logging_manager import LoggingManager
from floris.type_dec import NDArrayFloat


class HeterogeneousMap(LoggingManager):
    """
    Class for handling heterogeneous inflow configurations when defined by wind direction
      and wind speed.
    Args:
        x (NDArrayFloat): A 1D NumPy array (size num_points) of x-coordinates (meters).
        y (NDArrayFloat): A 1D NumPy array (size num_points) of y-coordinates (meters).
        speed_multipliers (NDArrayFloat): A 2D NumPy array (size num_wd (or num_ws) x num_points)
            of speed multipliers.  If neither wind_directions nor wind_speeds are defined, then
            this should be a single row array
        wind_directions (NDArrayFloat, optional): A 1D NumPy array (size num_wd) of wind directions
            (degrees). Optional.
        wind_speeds (NDArrayFloat, optional): A 1D NumPy array (size num_ws) of wind speeds (m/s).
            Optional.


    Notes:
        * If wind_directions and wind_speeds are both defined, then they must be the same length
            and equal the length of the 0th dimension of 'speed_multipliers'.

    """

    def __init__(
        self,
        x: NDArrayFloat,
        y: NDArrayFloat,
        speed_multipliers: NDArrayFloat,
        wind_directions: NDArrayFloat = None,
        wind_speeds: NDArrayFloat = None,
    ):
        # Check that x, y and speed_multipliers are numpy arrays
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy array")
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy array")
        if not isinstance(speed_multipliers, np.ndarray):
            raise TypeError("speed_multipliers must be a numpy array")

        # Save the values
        self.x = x
        self.y = y
        self.speed_multipliers = speed_multipliers

        # Check that the length of the 1st dimension of speed_multipliers is the
        # same as the length of both x and y
        if len(self.x) != self.speed_multipliers.shape[1]:
            raise ValueError(
                "The length of x must equal the 1th dimension of speed_multipliers"
                "Within the heterogeneous_inflow_config_by_wd dictionary"
            )
        if len(self.y) != self.speed_multipliers.shape[1]:
            raise ValueError(
                "The length of y must equal the 1th dimension of speed_multipliers"
                "Within the heterogeneous_inflow_config_by_wd dictionary"
            )

        # If wind_directions is note None, check that it is valid then save it
        if wind_directions is not None:
            if not isinstance(wind_directions, np.ndarray):
                raise TypeError("wind_directions must be a numpy array")

            # Check that length of wind_directions is the same as the length of the 0th
            # dimension of speed_multipliers
            if len(wind_directions) != self.speed_multipliers.shape[0]:
                raise ValueError(
                    "The length of wind_directions must equal "
                    "the 0th dimension of speed_multipliers"
                    "Within the heterogeneous_inflow_config_by_wd dictionary"
                )

            self.wind_directions = wind_directions
        else:
            self.wind_directions = None

        # If wind_speeds is not None, check that it is valid then save it
        if wind_speeds is not None:
            if not isinstance(wind_speeds, np.ndarray):
                raise TypeError("wind_speeds must be a numpy array")

            # Check that length of wind_speeds is the same as the length of the 0th
            # dimension of speed_multipliers
            if len(wind_speeds) != self.speed_multipliers.shape[0]:
                raise ValueError(
                    "The length of wind_speeds must equal "
                    "the 0th dimension of speed_multipliers"
                    "Within the heterogeneous_inflow_config_by_wd dictionary"
                )

            self.wind_speeds = wind_speeds
        else:
            self.wind_speeds = None

        # If both wind_directions and wind_speeds are None, then speed_multipliers should be
        # length 1 in 0th dimension
        if self.wind_speeds is None and self.wind_directions is None:
            if self.speed_multipliers.shape[0] != 1:
                raise ValueError(
                    "If both wind_speeds and wind_directions are None, then speed_multipliers"
                    "should be length 1 in 0th dimension"
                )

        # If both wind_directions and wind_speeds are not None, then make sure each row
        # of a matrix where wind directions and wind speeds are the columns is unique
        if self.wind_speeds is not None and self.wind_directions is not None:
            if len(
                np.unique(np.column_stack((self.wind_directions, self.wind_speeds)), axis=0)
            ) != len(self.wind_directions):
                raise ValueError(
                    "Each row of a matrix where wind directions and wind speeds are the columns"
                    "should be unique"
                )

    def get_heterogeneous_inflow_config(
        self,
        wind_directions: NDArrayFloat | list[float],
        wind_speeds: NDArrayFloat | list[float],
    ):
        """
        Get the heterogeneous inflow configuration for the given wind directions and wind speeds.
        Args:
            wind_directions (NDArrayFloat | list[float]): A 1D NumPy array or
                list of wind directions (degrees).
            wind_speeds (NDArrayFloat | list[float]): A 1D NumPy array or list of wind speeds (m/s).
        Returns:
            dict: A dictionary (heterogeneous_inflow_config) containing the x, y,
            and speed_multipliers for the given wind directions and wind speeds.
        """
        # Check the wind_directions and wind_speeds are either lists or numpy arrays,
        # and are the same length
        if not isinstance(wind_directions, (list, np.ndarray)):
            raise TypeError("wind_directions must be a list or numpy array")
        if not isinstance(wind_speeds, (list, np.ndarray)):
            raise TypeError("wind_speeds must be a list or numpy array")
        if len(wind_directions) != len(wind_speeds):
            raise ValueError("wind_directions and wind_speeds must be the same length")

        # Select for wind direction first
        if self.wind_directions is not None:
            angle_diffs = np.abs(wind_directions[:, None] - self.wind_directions)
            min_angle_diffs = np.minimum(angle_diffs, 360 - angle_diffs)

            # If wind_speeds is none, can return the value in each case
            if self.wind_speeds is None:
                closest_wd_indices = np.argmin(min_angle_diffs, axis=1)

                # Construct the output array using the calculated indices
                speed_multipliers_by_findex = self.speed_multipliers[closest_wd_indices]

            # Need to loop over cases and match by wind speed
            else:
                speed_diffs = np.abs(wind_speeds[:, None] - self.wind_speeds)

                # Initialize the output array
                speed_multipliers_by_findex = np.zeros((len(wind_directions), len(self.x)))

                # Loop over each wind direction
                for i in range(len(wind_directions)):
                    # Find all the indices in the ith row of min_angle_diffs
                    # that are equal to the minimum value
                    closest_wd_indices = np.where(min_angle_diffs[i] == min_angle_diffs[i].min())[0]

                    # Find the index of the minimum value in the ith row of speed_diffs
                    # conditions on that index being in closest_wd_indices
                    closest_ws_index = np.argmin(speed_diffs[i, closest_wd_indices])

                    # Construct the output array using the calculated indices
                    speed_multipliers_by_findex[i] = self.speed_multipliers[
                        closest_wd_indices[closest_ws_index]
                    ]

        # If wind speeds are defined without wind direction
        elif self.wind_speeds is not None:
            speed_diffs = np.abs(wind_speeds[:, None] - self.wind_speeds)
            closest_ws_indices = np.argmin(speed_diffs, axis=1)

            # Construct the output array using the calculated indices
            speed_multipliers_by_findex = self.speed_multipliers[closest_ws_indices]

        # Else if both are None, then speed_multipliers should be length 1 in 0th
        # dimension and so just 1 row
        # repeat this row until length of wind_directions
        else:
            speed_multipliers_by_findex = np.repeat(
                self.speed_multipliers, len(wind_directions), axis=0
            )

        # Return heterogeneous_inflow_config
        return {
            "x": self.x,
            "y": self.y,
            "speed_multipliers": speed_multipliers_by_findex,
        }

    def plot_heterogeneous_boundary(self, ax=None):
        # If not provided create the axis
        if ax is None:
            _, ax = plt.subplots()

        # Get the x and y coordinates of the het map
        points = np.array(
            list(
                zip(
                    self.x,
                    self.y,
                )
            )
        )

        # Derive and plot the convex hull surrounding the points
        hull = ConvexHull(points)
        ax.plot(
            points[np.append(hull.vertices, hull.vertices[0]), 0],
            points[np.append(hull.vertices, hull.vertices[0]), 1],
            "--",
            color="gray",
            label="Heterogeneous Boundary",
        )

    def plot_wind_direction(self, ax: plt.Axes, wind_direction: float):
        """
        Plot the wind direction as an arrow on the plot.
        Args:
            ax (matplotlib.axes.Axes): The axes on which to plot the wind direction.
            wind_direction (float): The wind direction to plot.
        """

        # Get the x and y limits of the axis
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Find a point in the top-left corner of the plot
        xm = xlim[0] + 0.2 * (xlim[1] - xlim[0])
        ym = ylim[1] - 0.2 * (ylim[1] - ylim[0])

        # Select a radius for the circle 5% the plot width
        radius = 0.075 * (xlim[1] - xlim[0])

        theta = np.linspace(0.0, 2 * np.pi, 100)
        xcirc = np.cos(theta) * radius + xm
        ycirc = np.sin(theta) * radius + ym
        ax.scatter(xm, ym, color="k", marker="o")
        ax.plot(xcirc, ycirc, color="w", linewidth=2)
        plt.arrow(
            x=xm - np.cos(-(wind_direction - 270.0) * np.pi / 180.0) * radius,
            y=ym - np.sin(-(wind_direction - 270.0) * np.pi / 180.0) * radius,
            dx=1 * np.cos(-(wind_direction - 270.0) * np.pi / 180.0) * radius,
            dy=1 * np.sin(-(wind_direction - 270.0) * np.pi / 180.0) * radius,
            width=0.125 * radius,
            head_width=0.3 * radius,
            head_length=0.3 * radius,
            length_includes_head=True,
            color="w",
        )

    def plot_single_speed_multiplier(
        self,
        wind_direction: float,
        wind_speed: float,
        ax=None,
        cmap=cm.viridis,
        show_boundary=True,
        show_wind_direction=True,
        show_colorbar=True,
    ):
        """
        Plot the speed multipliers as a heatmap.
        Args:
            wind_direction (float): The wind direction for which to plot the speed multipliers.
            wind_speed (float): The wind speed for which to plot the speed multipliers.
            ax (matplotlib.axes.Axes, optional): The axes on which to plot the speed multipliers.
                If None, a new figure and axes will be created.
            cmap (matplotlib.colors.Colormap, optional): The colormap to use for the heatmap.
                Default is matplotlib.cm.viridis.
            show_boundary (bool, optional): Whether to show the boundary of the heterogeneous
                inflow configuration. Default is True.
            show_wind_direction (bool, optional): Whether to show the wind direction as an arrow.
                Default is True.
            show_colorbar (bool, optional): Whether to show the colorbar. Default is True.

        Returns:
            matplotlib.axes.Axes: The axes on which the speed multipliers are plotted.
        """

        # Confirm wind_direction and wind_speed are floats
        if not isinstance(wind_direction, float):
            raise TypeError("wind_direction must be a float")
        if not isinstance(wind_speed, float):
            raise TypeError("wind_speed must be a float")

        # Get the speed multipliers for the given wind direction and wind speed
        speed_multiplier_row = self.get_heterogeneous_inflow_config(
            np.array([wind_direction]), np.array([wind_speed])
        )["speed_multipliers"][0]

        # If not provided create the axis
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        # Get the x and y coordinates
        x = self.x
        y = self.y

        # Get some boundary info
        min_x = np.min(x)
        max_x = np.max(x)
        min_y = np.min(y)
        max_y = np.max(y)
        delta_x = max_x - min_x
        delta_y = max_y - min_y
        plot_min_x = min_x - 0.1 * delta_x
        plot_max_x = max_x + 0.1 * delta_x
        plot_min_y = min_y - 0.1 * delta_y
        plot_max_y = max_y + 0.1 * delta_y

        # Fill in the plot area
        x_plot, y_plot = np.meshgrid(
            np.linspace(plot_min_x, plot_max_x, 100),
            np.linspace(plot_min_y, plot_max_y, 100),
            indexing="ij",
        )
        x_plot = x_plot.flatten()
        y_plot = y_plot.flatten()

        try:
            lin_interpolant = LinearNDInterpolator(
                points=np.vstack([x, y]).T,
                values=speed_multiplier_row,
                fill_value=np.nan,
            )
            lin_values = lin_interpolant(x, y)
        except scipy.spatial._qhull.QhullError:
            self.logger.warning("QhullError occurred. Falling back to nearest neighbor.")
            lin_values = np.nan * np.ones_like(x)

        nearest_interpolant = NearestNDInterpolator(
            x=np.vstack([x, y]).T,
            y=speed_multiplier_row,
        )
        nn_values = nearest_interpolant(x, y)
        ids_isnan = np.isnan(lin_values)

        het_map_mesh = np.array(lin_values, copy=True)
        het_map_mesh[ids_isnan] = nn_values[ids_isnan]

        # Produce color plot of the speed multipliers
        im = ax.tricontourf(
            x,
            y,
            het_map_mesh,
            cmap=cmap,
            vmin=speed_multiplier_row.min(),
            vmax=speed_multiplier_row.max(),
            levels=50,
            zorder=-1,
        )

        # Plot the grid coordinates as a scatter plot
        ax.scatter(x, y, color="gray", marker=".", label="Heterogeneous Points")
        ax.set_xlim

        # Show the boundary
        if show_boundary:
            self.plot_heterogeneous_boundary(ax)

        # Add a colorbar
        if show_colorbar:
            fig.colorbar(im, ax=ax)

        # Set the x and y limits
        ax.set_xlim(plot_min_x, plot_max_x)
        ax.set_ylim(plot_min_y, plot_max_y)

        # Make equal axis
        ax.set_aspect("equal")

        # Set the x and y labels
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

        # Add the wind direction arrow
        if show_wind_direction:
            self.plot_wind_direction(ax, wind_direction)

        return ax
