# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation
from __future__ import annotations

import copy
from typing import Union

import matplotlib as mpl
import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from scipy.spatial import ConvexHull

from floris.simulation import Floris
from floris.tools.cut_plane import CutPlane
from floris.tools.floris_interface import FlorisInterface
from floris.utilities import rotate_coordinates_rel_west, wind_delta


def show_plots():
    plt.show()

def plot_turbines(
    ax,
    layout_x,
    layout_y,
    yaw_angles,
    rotor_diameters,
    color: str | None = None,
    wind_direction: float = 270.0,
):
    """
    Plot wind plant layout from turbine locations.

    Args:
        ax (:py:class:`matplotlib.pyplot.axes`): Figure axes.
        layout_x (np.array): Wind turbine locations (east-west).
        layout_y (np.array): Wind turbine locations (north-south).
        yaw_angles (np.array): Yaw angles of each wind turbine.
        D (float): Wind turbine rotor diameter.
        color (str): Pyplot color option to plot the turbines.
        wind_direction (float): Wind direction (rotates farm)
    """
    if color is None:
        color = "k"

    # Rotate layout to inertial frame for plotting turbines relative to wind direction
    coordinates_array = np.array([[x, y, 0.0] for x, y in list(zip(layout_x, layout_y))])
    layout_x, layout_y, _, _, _ = rotate_coordinates_rel_west(
        np.array([wind_direction]),
        coordinates_array
    )

    for x, y, yaw, d in zip(layout_x[0,0], layout_y[0,0], yaw_angles, rotor_diameters):
        R = d / 2.0
        x_0 = x + np.sin(np.deg2rad(yaw)) * R
        x_1 = x - np.sin(np.deg2rad(yaw)) * R
        y_0 = y - np.cos(np.deg2rad(yaw)) * R
        y_1 = y + np.cos(np.deg2rad(yaw)) * R
        ax.plot([x_0, x_1], [y_0, y_1], color=color)


def plot_turbines_with_fi(
    fi: FlorisInterface,
    ax=None,
    color=None,
    wd=None,
    yaw_angles=None,
):
    """
    Wrapper function to plot turbines which extracts the data
    from a FLORIS interface object

    Args:
        fi (:py:class:`floris.tools.floris_interface.FlorisInterface`): FlorisInterface object.
        ax (:py:class:`matplotlib.pyplot.axes`): Figure axes. Defaults to None.
        color (str, optional): Color to plot turbines. Defaults to None.
        wd (list, optional): The wind direction to plot the turbines relative to. Defaults to None.
        yaw_angles (NDArray, optional): The yaw angles for the turbines. Defaults to None.
    """
    if not ax:
        fig, ax = plt.subplots()
    if yaw_angles is None:
        yaw_angles = fi.floris.farm.yaw_angles
    if wd is None:
        wd = fi.floris.flow_field.wind_directions[0]

    # Rotate yaw angles to inertial frame for plotting turbines relative to wind direction
    yaw_angles = yaw_angles - wind_delta(np.array(wd))

    plot_turbines(
        ax,
        fi.layout_x,
        fi.layout_y,
        yaw_angles.flatten(),
        fi.floris.farm.rotor_diameters.flatten(),
        color=color,
        wind_direction=fi.floris.flow_field.wind_directions[0],
    )


def add_turbine_id_labels(fi: FlorisInterface, ax: plt.Axes, **kwargs):
    """
    Adds index labels to a plot based on the given FlorisInterface.
    See the pyplot.annotate docs for more info:
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.annotate.html.
    kwargs are passed to Text
    (https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text).

    Args:
        fi (FlorisInterface): Simulation object to get the layout and index information.
        ax (plt.Axes): Axes object to add the labels.
    """

    # Rotate layout to inertial frame for plotting turbines relative to wind direction
    coordinates_array = np.array([
        [x, y, 0.0]
        for x, y in list(zip(fi.layout_x, fi.layout_y))
    ])
    wind_direction = fi.floris.flow_field.wind_directions[0]
    layout_x, layout_y, _, _, _ = rotate_coordinates_rel_west(
        np.array([wind_direction]),
        coordinates_array
    )

    for i in range(fi.floris.farm.n_turbines):
        ax.annotate(
            i,
            (layout_x[0,0,i], layout_y[0,0,i]),
            xytext=(0,10),
            textcoords="offset points",
            **kwargs
        )


def line_contour_cut_plane(cut_plane, ax=None, levels=None, colors=None, **kwargs):
    """
    Visualize a cut_plane as a line contour plot.

    Args:
        cut_plane (:py:class:`~.tools.cut_plane.CutPlane`):
            CutPlane Object.
        ax (:py:class:`matplotlib.pyplot.axes`): Figure axes. Defaults
            to None.
        levels (np.array, optional): Contour levels for plot.
            Defaults to None.
        colors (list, optional): Strings of color specification info.
            Defaults to None.
        **kwargs: Additional parameters to pass to `ax.contour`.
    """

    if not ax:
        fig, ax = plt.subplots()

    rcParams["contour.negative_linestyle"] = "solid"

    # Plot the cut-through
    contours = ax.tricontour(
        cut_plane.df.x1,
        cut_plane.df.x2,
        cut_plane.df.u,
        levels=levels,
        colors=colors,
        extend="both",
        **kwargs,
    )

    ax.clabel(contours, contours.levels, inline=True, fontsize=10, colors="black")

    # Make equal axis
    ax.set_aspect("equal")


def visualize_cut_plane(
    cut_plane,
    ax=None,
    vel_component='u',
    min_speed=None,
    max_speed=None,
    cmap="coolwarm",
    levels=None,
    clevels=None,
    color_bar=False,
    title="",
    **kwargs
):
    """
    Generate pseudocolor mesh plot of the cut_plane.

    Args:
        cut_plane (:py:class:`~.tools.cut_plane.CutPlane`): 2D
            plane through wind plant.
        ax (:py:class:`matplotlib.pyplot.axes`, optional): Figure axes. Defaults
            to None.
        vel_component (str, optional): The velocity component that the cut plane is
            perpendicular to.
        min_speed (float, optional): Minimum value of wind speed for
            contours. Defaults to None.
        max_speed (float, optional): Maximum value of wind speed for
            contours. Defaults to None.
        cmap (str, optional): Colormap specifier. Defaults to
            'coolwarm'.
        levels (np.array, optional): Contour levels for line contour plot.
            Defaults to None.
        clevels (np.array, optional): Contour levels for tricontourf plot.
            Defaults to None.
        color_bar (Boolean, optional): Flag to include a color bar on the plot.
            Defaults to False.
        title (str, optional): User-supplied title for the plot. Defaults to "".
        **kwargs: Additional parameters to pass to line contour plot.

    Returns:
        im (:py:class:`matplotlib.plt.pcolormesh`): Image handle.
    """

    if not ax:
        fig, ax = plt.subplots()

    if vel_component=='u':
        # vel_mesh = cut_plane.df.u.values.reshape(cut_plane.resolution[1], cut_plane.resolution[0])
        if min_speed is None:
            min_speed = cut_plane.df.u.min()
        if max_speed is None:
            max_speed = cut_plane.df.u.max()
    elif vel_component=='v':
        # vel_mesh = cut_plane.df.v.values.reshape(cut_plane.resolution[1], cut_plane.resolution[0])
        if min_speed is None:
            min_speed = cut_plane.df.v.min()
        if max_speed is None:
            max_speed = cut_plane.df.v.max()
    elif vel_component=='w':
        # vel_mesh = cut_plane.df.w.values.reshape(cut_plane.resolution[1], cut_plane.resolution[0])
        if min_speed is None:
            min_speed = cut_plane.df.w.min()
        if max_speed is None:
            max_speed = cut_plane.df.w.max()

    # Allow separate number of levels for tricontourf and for line_contour
    if clevels is None:
        clevels = levels

    # Plot the cut-through
    im = ax.tricontourf(
        cut_plane.df.x1,
        cut_plane.df.x2,
        cut_plane.df.u,
        vmin=min_speed,
        vmax=max_speed,
        levels=clevels,
        cmap=cmap,
        extend="both",
    )

    # Add line contour
    line_contour_cut_plane(
        cut_plane,
        ax=ax,
        levels=levels,
        colors="b",
        linewidths=0.8,
        alpha=0.3,
        **kwargs
    )

    if cut_plane.normal_vector == "x":
        ax.invert_xaxis()

    if color_bar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('m/s')

    # Set the title
    ax.set_title(title)

    # Make equal axis
    ax.set_aspect("equal")

    return im


def visualize_heterogeneous_cut_plane(
    cut_plane,
    fi,
    ax=None,
    vel_component='u',
    min_speed=None,
    max_speed=None,
    cmap="coolwarm",
    levels=None,
    clevels=None,
    color_bar=False,
    title="",
    plot_het_bounds=True,
    **kwargs
):
    """
    Generate pseudocolor mesh plot of the heterogeneous cut_plane.

    Args:
        cut_plane (:py:class:`~.tools.cut_plane.CutPlane`): 2D
            plane through wind plant.
        fi (:py:class:`~.tools.floris_interface.FlorisInterface`): FlorisInterface object.
        ax (:py:class:`matplotlib.pyplot.axes`): Figure axes. Defaults
            to None.
        vel_component (str, optional): The velocity component that the cut plane is
            perpendicular to.
        min_speed (float, optional): Minimum value of wind speed for
            contours. Defaults to None.
        max_speed (float, optional): Maximum value of wind speed for
            contours. Defaults to None.
        cmap (str, optional): Colormap specifier. Defaults to
            'coolwarm'.
        levels (np.array, optional): Contour levels for line contour plot.
            Defaults to None.
        clevels (np.array, optional): Contour levels for tricontourf plot.
            Defaults to None.
        color_bar (Boolean, optional): Flag to include a color bar on the plot.
            Defaults to False.
        title (str, optional): User-supplied title for the plot. Defaults to "".
        plot_het_bonds (boolean, optional): Flag to include the user-defined bounds of the
            heterogeneous wind speed area. Defaults to True.
        **kwargs: Additional parameters to pass to line contour plot.

    Returns:
        im (:py:class:`matplotlib.plt.pcolormesh`): Image handle.
    """

    if not ax:
        fig, ax = plt.subplots()
    if vel_component=='u':
        # vel_mesh = cut_plane.df.u.values.reshape(cut_plane.resolution[1], cut_plane.resolution[0])
        if min_speed is None:
            min_speed = cut_plane.df.u.min()
        if max_speed is None:
            max_speed = cut_plane.df.u.max()
    elif vel_component=='v':
        # vel_mesh = cut_plane.df.v.values.reshape(cut_plane.resolution[1], cut_plane.resolution[0])
        if min_speed is None:
            min_speed = cut_plane.df.v.min()
        if max_speed is None:
            max_speed = cut_plane.df.v.max()
    elif vel_component=='w':
        # vel_mesh = cut_plane.df.w.values.reshape(cut_plane.resolution[1], cut_plane.resolution[0])
        if min_speed is None:
            min_speed = cut_plane.df.w.min()
        if max_speed is None:
            max_speed = cut_plane.df.w.max()

    # Allow separate number of levels for tricontourf and for line_contour
    if clevels is None:
        clevels = levels

    # Plot the cut-through
    im = ax.tricontourf(
        cut_plane.df.x1,
        cut_plane.df.x2,
        cut_plane.df.u,
        vmin=min_speed,
        vmax=max_speed,
        levels=clevels,
        cmap=cmap,
        extend="both",
    )

    # Add line contour
    line_contour_cut_plane(
        cut_plane,
        ax=ax,
        levels=levels,
        colors="b",
        linewidths=0.8,
        alpha=0.3,
        **kwargs
    )

    # Plot the user-defined heterogeneous flow area
    if plot_het_bounds:
        points = np.array(
            list(
                zip(
                    fi.floris.flow_field.heterogenous_inflow_config['x'],
                    fi.floris.flow_field.heterogenous_inflow_config['y'],
                )
            )
        )
        hull = ConvexHull(points)
        h = ax.plot(
            points[np.append(hull.vertices, hull.vertices[0]),0],
            points[np.append(hull.vertices, hull.vertices[0]), 1],
            'k--',
            lw=2,
        )
        ax.plot(points[hull.vertices,0], points[hull.vertices,1], 'ko')
        ax.legend(h, ["defined heterogeneous bounds"], loc=1)

    if cut_plane.normal_vector == "x":
        ax.invert_xaxis()

    if color_bar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('m/s')

    # Set the title
    ax.set_title(title)

    # Make equal axis
    ax.set_aspect("equal")

    return im


def visualize_quiver(cut_plane, ax=None, min_speed=None, max_speed=None, downSamp=1, **kwargs):
    """
    Visualize the in-plane flows in a cut_plane using quiver.

    Args:
        cut_plane (:py:class:`~.tools.cut_plane.CutPlane`): 2D
            plane through wind plant.
        ax (:py:class:`matplotlib.pyplot.axes`): Figure axes. Defaults
            to None.
        min_speed (float, optional): Minimum value of wind speed for
            contours. Defaults to None.
        max_speed (float, optional): Maximum value of wind speed for
            contours. Defaults to None.
        downSamp (int, optional): Down sample the number of quiver arrows
            from underlying grid.
        **kwargs: Additional parameters to pass to `ax.streamplot`.

    Returns:
        im (:py:class:`matplotlib.plt.pcolormesh`): Image handle.
    """
    if not ax:
        fig, ax = plt.subplots()

    # Reshape UMesh internally
    x1_mesh = cut_plane.df.x1.values.reshape(cut_plane.resolution[1], cut_plane.resolution[0])
    x2_mesh = cut_plane.df.x2.values.reshape(cut_plane.resolution[1], cut_plane.resolution[0])
    v_mesh = cut_plane.df.v.values.reshape(cut_plane.resolution[1], cut_plane.resolution[0])
    w_mesh = cut_plane.df.w.values.reshape(cut_plane.resolution[1], cut_plane.resolution[0])

    # plot the stream plot
    ax.streamplot(
        (x1_mesh[::downSamp, ::downSamp]),
        (x2_mesh[::downSamp, ::downSamp]),
        v_mesh[::downSamp, ::downSamp],
        w_mesh[::downSamp, ::downSamp],
        # scale=80.0,
        # alpha=0.75,
        # **kwargs
    )

    # ax.quiverkey(QV1, -.75, -0.4, 1, '1 m/s', coordinates='data')

    # Make equal axis
    # ax.set_aspect('equal')


def reverse_cut_plane_x_axis_in_plot(ax):
    """
    Shortcut method to reverse direction of x-axis.

    Args:
        ax (:py:class:`matplotlib.pyplot.axes`): Figure axes.
    """
    ax.invert_xaxis()


def plot_rotor_values(
    values: np.ndarray,
    wd_index: int,
    ws_index: int,
    n_rows: int,
    n_cols: int,
    t_range: range | None = None,
    cmap: str = "coolwarm",
    return_fig_objects: bool = False,
    save_path: Union[str, None] = None,
    show: bool = False
) -> Union[None, tuple[plt.figure, plt.axes, plt.axis, plt.colorbar]]:
    """
    Plots the gridded turbine rotor values. This is intended to be used for
    understanding the differences between two sets of values, so each subplot can be
    used for inspection of what values are differing, and under what conditions.

    Parameters:
        values (np.ndarray): The 5-dimensional array of values to plot. Should be:
            N wind directions x N wind speeds x N turbines X N rotor points X N rotor points.
        wd_index (int): The index for the wind direction to plot.
        ws_index (int): The index of the wind speed to plot.
        n_rows (int): The number of rows to include for subplots. With ncols, this should
            generally add up to the number of turbines in the farm.
        n_cols (int): The number of columns to include for subplots. With ncols, this should
            generally add up to the number of turbines in the farm.
        t_range (range | None): Optional. The turbine count used to create the title for each
            subplot. If not provided, the size of the 2-th dimension of `values` is used.
        cmap (str): Optional. The matplotlib colormap to be used, default "coolwarm".
        return_fig_objects (bool): Optional. Flag to return the primary figure objects for
            further editing, default False.
        save_path (str | None): Optional. Where to save the figure, if a value is provided.
        show (bool): Optional. Flag to run `plt.show()` to present all the plots
            currently created with matplotlib.

    Returns:
        None | tuple[plt.figure, plt.axes, plt.axis, plt.colorbar]: If
        `return_fig_objects` is `False, then `None` is returned`, otherwise the primary
        figure objects are returned for custom editing.

    Example:
        from floris.tools.visualization import plot_rotor_values
        plot_rotor_values(floris.flow_field.u, wd_index=0, ws_index=0, n_rows=1, ncols=4)
        plot_rotor_values(floris.flow_field.v, wd_index=0, ws_index=0, n_rows=1, ncols=4)
        plot_rotor_values(floris.flow_field.w, wd_index=0, ws_index=0, n_rows=1, ncols=4, show=True)
    """

    cmap = plt.cm.get_cmap(name=cmap)

    if t_range is None:
        t_range = range(values.shape[2])

    fig = plt.figure()
    axes = fig.subplots(n_rows, n_cols)

    # For 1x1, fig.subplots returns an Axes object, but for more than 1x1 it returns a np.array.
    # In this case, convert to an array so that the rest of this function can be simplified.
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])

    titles = np.array([f"T{i}" for i in t_range])

    for ax, t, i in zip(axes.flatten(), titles, t_range):

        vmin = np.min(values[wd_index, ws_index])
        vmax = np.max(values[wd_index, ws_index])

        norm = mplcolors.Normalize(vmin, vmax)

        ax.imshow(values[wd_index, ws_index, i].T, cmap=cmap, norm=norm, origin="lower")
        ax.invert_xaxis()

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(t)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.83, 0.25, 0.03, 0.5])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if return_fig_objects:
        return fig, axes, cbar_ax, cb

    if show:
        plt.show()

def calculate_horizontal_plane_with_turbines(
    fi_in,
    x_resolution=200,
    y_resolution=200,
    x_bounds=None,
    y_bounds=None,
    wd=None,
    ws=None,
    yaw_angles=None,
) -> CutPlane:
        """
        This function creates a :py:class:`~.tools.cut_plane.CutPlane` by
        adding an additional turbine to the farm and moving it through every
        a regular grid throughout the flow field. This method allows for
        visualizing wake models that do not support the FullFlowGrid and
        its associated solver. As the new turbine is moved around the flow
        field, the velocities at its rotor are stored in local variables,
        and the flow field is reset to its initial state for every new
        location. Then, the local velocities are put into a DataFrame and
        then into a CutPlane. This method is much slower than
        `FlorisInterface.calculate_horizontal_plane`, but it is helpful
        for models where the visualization capability is not yet available.

        Args:
            fi_in (:py:class:`floris.tools.floris_interface.FlorisInterface`):
                Preinitialized FlorisInterface object.
            x_resolution (float, optional): Output array resolution. Defaults to 200 points.
            y_resolution (float, optional): Output array resolution. Defaults to 200 points.
            x_bounds (tuple, optional): Limits of output array (in m). Defaults to None.
            y_bounds (tuple, optional): Limits of output array (in m). Defaults to None.
            wd (float, optional): Wind direction setting. Defaults to None.
            ws (float, optional): Wind speed setting. Defaults to None.
            yaw_angles (np.ndarray, optional): Yaw angles settings. Defaults to None.

        Returns:
            :py:class:`~.tools.cut_plane.CutPlane`: containing values of x, y, u, v, w
        """

        # Make a local copy of fi to avoid editing passed in fi
        fi = copy.deepcopy(fi_in)

        # If wd/ws not provided, use what is set in fi
        if wd is None:
            wd = fi.floris.flow_field.wind_directions
        if ws is None:
            ws = fi.floris.flow_field.wind_speeds
        fi.check_wind_condition_for_viz(wd=wd, ws=ws)

        # Set the ws and wd
        fi.reinitialize(wind_directions=wd, wind_speeds=ws)

        # Re-set yaw angles
        if yaw_angles is not None:
            fi.floris.farm.yaw_angles = yaw_angles

        # Now place the yaw_angles back into yaw_angles
        # to be sure not None
        yaw_angles = fi.floris.farm.yaw_angles

        # Grab the turbine layout
        layout_x = copy.deepcopy(fi.layout_x)
        layout_y = copy.deepcopy(fi.layout_y)
        D = fi.floris.farm.rotor_diameters_sorted[0, 0, 0]

        # Declare a new layout array with an extra turbine
        layout_x_test = np.append(layout_x,[0])
        layout_y_test = np.append(layout_y,[0])
        yaw_angles = np.append(yaw_angles, np.zeros([len(wd), len(ws), 1]), axis=2)

        # Get a grid of points test test
        if x_bounds is None:
            x_bounds = (np.min(layout_x) - 2 * D, np.max(layout_x) + 10 * D)

        if y_bounds is None:
            y_bounds = (np.min(layout_y) - 2 * D, np.max(layout_y) + 2 * D)

        # Now generate a list of points
        x_points = np.linspace(x_bounds[0], x_bounds[1], x_resolution)
        y_points = np.linspace(y_bounds[0], y_bounds[1], y_resolution)
        num_points = len(x_points) * len(y_points)

        # Now loop over the points
        x_results = np.zeros(num_points)
        y_results = np.zeros(num_points)
        z_results = np.zeros(num_points)
        u_results = np.zeros(num_points)
        v_results = np.zeros(num_points)
        w_results = np.zeros(num_points)
        idx = 0
        for y in y_points:
            for x in x_points:

                # Save the x and y results
                x_results[idx] = x
                y_results[idx] = y

                # Place the test turbine at this location and calculate wake
                layout_x_test[-1] = x
                layout_y_test[-1] = y
                fi.reinitialize(layout_x = layout_x_test, layout_y = layout_y_test)
                fi.calculate_wake(yaw_angles=yaw_angles)

                # Get the velocity of that test turbines central point
                center_point = int(np.floor(fi.floris.flow_field.u[0,0,-1].shape[0] / 2.0))
                u_results[idx] = fi.floris.flow_field.u[0,0,-1,center_point,center_point]

                # Increment index
                idx = idx + 1

        # Make a dataframe
        df = pd.DataFrame({
            'x1':x_results,
            'x2':y_results,
            'x3':z_results,
            'u':u_results,
            'v':v_results,
            'w':w_results,
        })

        # Convert to a cut_plane
        horizontal_plane = CutPlane(df, x_resolution, y_resolution, "z")

        return horizontal_plane
