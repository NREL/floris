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


# Defines a bunch of tools for plotting and manipulating
# layouts for quick visualizations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from floris.utilities import rotate_coordinates_rel_west, wind_delta


def plot_turbine_points(fi, ax=None, turbine_indices=None, plotting_dict={}):
    """
    Plot the farm layout.

    Args:
        fi (:py:class:`floris.tools.floris_interface.FlorisInterface`): FlorisInterface object.
        ax: axes to plot on (if None, creates figure and axes)
        turbine_indices (list[int]): turbines to plot,
            default to all turbines

        plotting_dict: dictionary of plotting parameters, with the
            following (optional) fields and their (default) values:
            "color" : ("black")
            "marker" : (".")
            "markersize" : (10)
            "label" : (None) (for legend, if desired)



    Returns:
        ax: the current axes for the layout plot

    """

    # Generate axis, if needed
    if ax is None:
        _, ax = plt.subplots()

    # If turbine_indices is not none, make sure all elements correspond to real indices
    if turbine_indices is not None:
        try:
            fi.layout_x[turbine_indices]
        except IndexError:
            raise IndexError("turbine_indices does not correspond to turbine indices in fi")
    else:
        turbine_indices = list(range(len(fi.layout_x)))

    # Generate plotting dictionary
    default_plotting_dict = {
        "color": "black",
        "marker": ".",
        "markersize": 10,
        "label": None,
    }
    plotting_dict = {**default_plotting_dict, **plotting_dict}

    # Plot
    ax.plot(
        fi.layout_x[turbine_indices],
        fi.layout_y[turbine_indices],
        linestyle="None",
        **plotting_dict,
    )

    # Make sure axis set to equal
    ax.axis("equal")

    return ax


def plot_turbine_labels(
    fi,
    ax=None,
    turbine_names=None,
    turbine_indices=None,
    label_offset=None,
    show_bbox=False,
    bbox_dict={},
    plotting_dict={},
):
    """
    Labels the turbines on a farm

    Args:
        fi (:py:class:`floris.tools.floris_interface.FlorisInterface`): FlorisInterface object.
        ax: axes to plot on (if None, creates figure and axes)
        turbine_names (list[str]): Provided turbine names
            Defaults to 0-index list
        turbine_indices (list[int]): turbines to plot,
            default to all turbines
        label_offset (float): Amount in m to dispace label from point.  Defaults to r/8
        show_bbox (bool): Whether to put a box around the labels. Defaults to False
        bbox_dict (dict): Dictionary defining box around labels.
        plotting_dict: dictionary of plotting parameters, with the
            following (optional) fields and their (default) values:
            "color" : ("black")



    Returns:
        ax: the current axes for the layout plot
    """

    # Generate axis, if needed
    if ax is None:
        _, ax = plt.subplots()

    # If turbine names not none, confirm has correct number of turbines
    if turbine_names is not None:
        if len(turbine_names) != len(fi.layout_x):
            raise ValueError("Length of turbine_names not equal to number turbines in fi object")
    else:
        # Assign simple default numbering
        turbine_names = [f"{i:03d}" for i in range(len(fi.layout_x))]

    # If label_offset is None, use default value of r/8
    if label_offset is None:
        rotor_diameters = fi.floris.farm.rotor_diameters.flatten()
        r = rotor_diameters[0] / 2.0
        label_offset = r / 8.0

    # If turbine_indices is not none, make sure all elements correspond to real indices
    if turbine_indices is not None:
        try:
            fi.layout_x[turbine_indices]
        except IndexError:
            raise IndexError("turbine_indices does not correspond to turbine indices in fi")
    else:
        turbine_indices = list(range(len(fi.layout_x)))

    # Generate plotting dictionary
    default_plotting_dict = {
        "color": "black",
        "label": None,
    }
    plotting_dict = {**default_plotting_dict, **plotting_dict}

    # If showing bbox is true, if bbox_dict is None, use a default
    default_bbox_dict = {"facecolor": "gray", "alpha": 0.5, "pad": 0.1, "boxstyle": "round"}
    bbox_dict = {**default_bbox_dict, **bbox_dict}

    for ti in turbine_indices:
        if not show_bbox:
            ax.text(
                fi.layout_x[ti] + label_offset,
                fi.layout_y[ti] + label_offset,
                turbine_names[ti],
                **plotting_dict,
            )
        else:
            ax.text(
                fi.layout_x[ti] + label_offset,
                fi.layout_y[ti] + label_offset,
                turbine_names[ti],
                bbox=bbox_dict,
                **plotting_dict,
            )

    # Plot labels and aesthetics
    ax.axis("equal")

    return ax


def plot_turbines_rotors(
    fi,
    ax: plt.Axes = None,
    color: str = None,
    wd: np.ndarray = None,
    yaw_angles: np.ndarray = None,
):
    """
    Plot the wind turbine rotors including yaw angle

    Args:
        fi (:py:class:`floris.tools.floris_interface.FlorisInterface`): FlorisInterface object.
        ax (:py:class:`matplotlib.pyplot.axes`): Figure axes. Defaults to None.
        color (str, optional): Color to plot turbines. Defaults to None.
        wd (list, optional): The wind direction to plot the turbines relative to. Defaults to None.
        yaw_angles (NDArray, optional): The yaw angles for the turbines. Defaults to None.
    """
    if not ax:
        _, ax = plt.subplots()
    if yaw_angles is None:
        yaw_angles = fi.floris.farm.yaw_angles
    if wd is None:
        wd = fi.floris.flow_field.wind_directions[0]

    # Rotate yaw angles to inertial frame for plotting turbines relative to wind direction
    yaw_angles = yaw_angles - wind_delta(np.array(wd))

    if color is None:
        color = "k"

    # If yaw angles is not 1D, assume we want first findex
    yaw_angles = np.array(yaw_angles)
    if yaw_angles.ndim == 2:
        yaw_angles = yaw_angles[0, :]

    rotor_diameters = fi.floris.farm.rotor_diameters.flatten()
    for x, y, yaw, d in zip(fi.layout_x, fi.layout_y, yaw_angles, rotor_diameters):
        R = d / 2.0
        x_0 = x + np.sin(np.deg2rad(yaw)) * R
        x_1 = x - np.sin(np.deg2rad(yaw)) * R
        y_0 = y - np.cos(np.deg2rad(yaw)) * R
        y_1 = y + np.cos(np.deg2rad(yaw)) * R
        ax.plot([x_0, x_1], [y_0, y_1], color=color)


def get_wake_direction(x_i, y_i, x_j, y_j):
    """
    Calculates the compass direction where turbine i wakes turbine j

    Args:
        x_i: x-coordinate of the starting point
        y_i: y-coordinate of the starting point
        x_j: x-coordinate of the ending point
        y_j: y-coordinate of the ending point

    Returns:
        wake_direction (float): Angle in degrees, when turbine i wakes turbine j
    """

    dx = x_j - x_i
    dy = y_j - y_i

    angle_rad = np.arctan2(dy, dx)
    angle_deg = 270 - np.rad2deg(angle_rad)

    # Adjust for "from" direction (add 180 degrees) and wrap within 0-360
    wind_direction = angle_deg % 360

    return wind_direction


def label_line(
    line,
    label_text,
    ax,
    near_i=None,
    near_x=None,
    near_y=None,
    rotation_offset=0.0,
    offset=(0, 0),
    size=7,
):
    """
    [summary]

    Args:
        line (matplotlib.lines.Line2D): line to label.
        label_text (str): label to add to line.
        ax (:py:class:`matplotlib.pyplot.axes` optional): figure axes.
        near_i (int, optional): Catch line near index i.
            Defaults to None.
        near_x (float, optional): Catch line near coordinate x.
            Defaults to None.
        near_y (float, optional): Catch line near coordinate y.
            Defaults to None.
        rotation_offset (float, optional): label rotation in degrees.
            Defaults to 0.
        offset (tuple, optional): label offset from turbine location.
            Defaults to (0, 0).
        size (float): font size. Defaults to 7.

    Raises:
        ValueError: ("Need one of near_i, near_x, near_y") raised if
            insufficient information is passed in.
    """

    def put_label(i):
        """
        Add a label to index.

        Args:
            i (int): index to label.
        """
        i = min(i, len(x) - 2)
        dx = sx[i + 1] - sx[i]
        dy = sy[i + 1] - sy[i]
        rotation = np.rad2deg(np.arctan2(dy, dx)) + rotation_offset
        pos = [(x[i] + x[i + 1]) / 2.0 + offset[0], (y[i] + y[i + 1]) / 2 + offset[1]]
        ax.text(
            pos[0],
            pos[1],
            label_text,
            size=size,
            rotation=rotation,
            color=line.get_color(),
            ha="center",
            va="center",
            bbox={"ec":"1", "fc":"1", "alpha":0.8},
        )

    # extract line data
    x = line.get_xdata()
    y = line.get_ydata()

    # define screen spacing
    if ax.get_xscale() == "log":
        sx = np.log10(x)
    else:
        sx = x
    if ax.get_yscale() == "log":
        sy = np.log10(y)
    else:
        sy = y

    # find index
    if near_i is not None:
        i = near_i
        if i < 0:  # sanitize negative i
            i = len(x) + i
        put_label(i)
    elif near_x is not None:
        for i in range(len(x) - 2):
            if (x[i] < near_x and x[i + 1] >= near_x) or (x[i + 1] < near_x and x[i] >= near_x):
                put_label(i)
    elif near_y is not None:
        for i in range(len(y) - 2):
            if (y[i] < near_y and y[i + 1] >= near_y) or (y[i + 1] < near_y and y[i] >= near_y):
                put_label(i)
    else:
        raise ValueError("Need one of near_i, near_x, near_y")


def plot_waking_directions(
    fi,
    ax=None,
    turbine_indices=None,
    wake_plotting_dict={},
    D=None,
    limit_dist_D=None,
    limit_dist_m=None,
    limit_num=None,
    wake_label_size=7,

):
    """
    Plot waking directions and distances between turbines.

    Args:
        fi: Instantiated FlorisInterface object
        ax: axes to plot on (if None, creates figure and axes)
        turbine_indices (list[int]): turbines to plot,
            default to all turbines
        layout_plotting_dict: dictionary of plotting parameters for
            turbine locations. Defaults to the defaults of
            plot_layout_only.
        wake_plotting_dict: dictionary of plotting parameters for the
            waking directions, with the following (optional) fields and
            their (default) values:
            "color" : ("black"),
            "linestyle" : ("solid"),
            "linewidth" : (0.5)
        D: rotor diameter. Defaults to the rotor diamter of the first
            turbine in the Floris object.
        limit_dist_D: limit on the distance between turbines to plot,
            specified in rotor diamters.
        limit_dist_m: limit on the distance between turbines to plot,
            specified in meters. If specified, overrides limit_dist_D.
        limit_num: limit on number of outgoing neighbors to include.
            If specified, only the limit_num closest turbines are
            plotted. However, directions already plotted from other
            turbines are not considered in the count.
        wake_label_size: font size for labels of direction/distance.

    Returns:
        ax: the current axes for the thrust curve plot
    """

    # If turbine_indices is not none, make sure all elements correspond to real indices
    if turbine_indices is not None:
        try:
            fi.layout_x[turbine_indices]
        except IndexError:
            raise IndexError("turbine_indices does not correspond to turbine indices in fi")
    else:
        turbine_indices = list(range(len(fi.layout_x)))

    layout_x = fi.layout_x[turbine_indices]
    layout_y = fi.layout_y[turbine_indices]
    N_turbs = len(layout_x)

    # Combine default plotting options
    def_wake_plotting_dict = {
        "color": "black",
        "linestyle": "solid",
        "linewidth": 0.5,
    }
    wake_plotting_dict = {**def_wake_plotting_dict, **wake_plotting_dict}

    # N_turbs = len(fi.floris.farm.turbine_definitions)

    if D is None:
        D = fi.floris.farm.turbine_definitions[0]["rotor_diameter"]
        # TODO: build out capability to use multiple diameters, if of interest.
        # D = np.array([turb['rotor_diameter'] for turb in
        #      fi.floris.farm.turbine_definitions])
    # else:
    # D = D*np.ones(N_turbs)

    dists_m = np.zeros((N_turbs, N_turbs))
    angles_d = np.zeros((N_turbs, N_turbs))

    for i in range(N_turbs):
        for j in range(N_turbs):
            dists_m[i, j] = np.linalg.norm([layout_x[i] - layout_x[j], layout_y[i] - layout_y[j]])
            angles_d[i, j] = get_wake_direction(layout_x[i], layout_y[i], layout_x[j], layout_y[j])

    # Mask based on the limit distance (assumed to be in measurement D)
    if limit_dist_D is not None and limit_dist_m is None:
        limit_dist_m = limit_dist_D * D
    if limit_dist_m is not None:
        mask = dists_m > limit_dist_m
        dists_m[mask] = np.nan
        angles_d[mask] = np.nan

    # Handle default limit number case
    if limit_num is None:
        limit_num = -1

    # Loop over pairs, plot
    label_exists = np.full((N_turbs, N_turbs), False)
    for i in range(N_turbs):
        for j in range(N_turbs):
            # import ipdb; ipdb.set_trace()
            if (
                ~np.isnan(dists_m[i, j])
                and dists_m[i, j] != 0.0
                and ~(dists_m[i, j] > np.sort(dists_m[i, :])[limit_num])
                # and i in layout_plotting_dict["turbine_indices"]
                # and j in layout_plotting_dict["turbine_indices"]
            ):
                (h,) = ax.plot(fi.layout_x[[i, j]], fi.layout_y[[i, j]], **wake_plotting_dict)

                # Only label in one direction
                if ~label_exists[i, j]:
                    linetext = "{0:.1f} D --- {1:.0f}/{2:.0f}".format(
                        dists_m[i, j] / D,
                        angles_d[i, j],
                        angles_d[j, i],
                    )

                    label_line(
                        h,
                        linetext,
                        ax,
                        near_i=1,
                        near_x=None,
                        near_y=None,
                        rotation_offset=0,
                        size=wake_label_size,
                    )

                    label_exists[i, j] = True
                    label_exists[j, i] = True

    return ax


def plot_farm_terrain(fi, fig, ax):
    hub_heights = fi.floris.farm.hub_heights.flatten()
    cntr = ax.tricontourf(fi.layout_x, fi.layout_y, hub_heights, levels=14, cmap="RdBu_r")

    fig.colorbar(
        cntr,
        ax=ax,
        label="Terrain-corrected hub height (m)",
        ticks=np.linspace(
            np.min(hub_heights) - 10.0,
            np.max(hub_heights) + 10.0,
            15,
        ),
    )


def shade_region(
    points, show_points=False, plotting_dict_region={}, plotting_dict_points={}, ax=None
):
    """
    Shade a region defined by a series of vertices (points).

    Args:
        points: 2D array of vertices for the shaded region, shape N x 2,
            where each row contains a coordinate (x, y)
        show_points: Boolean to dictate whether to plot the points as well
            as the shaded region
        plotting_dict_region: dictionary of plotting parameters for the shaded
            region, with the following (optional) fields and their (default)
            values:
            "color" : ("black")
            "edgecolor": (None)
            "alpha" : (0.3)
            "label" : (None) (for legend, if desired)
        plotting_dict_region: dictionary of plotting parameters for the
            vertices (points), with the following (optional) fields and their
            (default) values:
            "color" : "black",
            "marker" : (None)
            "s" : (10)
            "label" : (None) (for legend, if desired)
        ax: axes to plot on (if None, creates figure and axes)

    Returns:
        ax: the current axes for the layout plot
    """

    # Generate axis, if needed
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

    # Generate plotting dictionary
    default_plotting_dict_region = {
        "color": "black",
        "edgecolor": None,
        "alpha": 0.3,
        "label": None,
    }
    plotting_dict_region = {**default_plotting_dict_region, **plotting_dict_region}

    ax.fill(points[:, 0], points[:, 1], **plotting_dict_region)

    if show_points:
        default_plotting_dict_points = {"color": "black", "marker": ".", "s": 10, "label": None}
        plotting_dict_points = {**default_plotting_dict_points, **plotting_dict_points}

        ax.scatter(points[:, 0], points[:, 1], **plotting_dict_points)

    # Plot labels and aesthetics
    ax.axis("equal")
    ax.grid(True)
    ax.set_xlabel("x coordinate (m)")
    ax.set_ylabel("y coordinate (m)")
    # if plotting_dict_region["label"] is not None or plotting_dict_points["label"] is not None:
    #     ax.legend()

    return ax
