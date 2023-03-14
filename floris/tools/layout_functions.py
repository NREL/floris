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


def visualize_layout(
    fi,
    ax=None,
    show_wake_lines=False,
    limit_dist_m=None,
    lim_lines_per_turbine=None,
    turbine_face_north=False,
    one_index_turbine=False,
    black_and_white=False,
    plot_rotor=False,
    turbine_names=None
):
    """
    Make a plot which shows the turbine locations, and important wakes.

    Args:
        fi object
        ax (:py:class:`matplotlib.pyplot.axes` optional):
            figure axes. Defaults to None.
        show_wake_lines (bool, optional): flag to control plotting of
            wake boundaries. Defaults to False.
        limit_dist_m (float, optional): Only plot distances less than this ammount (m)
            Defaults to None.
        lim_lines_per_turbine (int, optional): Limit number of lines eminating from a turbine
        turbine_face_north (bool, optional): Force orientation of wind
            turbines. Defaults to False.
        one_index_turbine (bool, optional): if true, 1st turbine is
            turbine 1 (ignored if turbine names provided)
        black_and_white (bool, optional): if true print in black and white
        plot_rotor (bool, optional): if true plot the turbine rotors and offset the labels
        turbines_names (list, optional): optional list of turbine names

    """

    # Build a dataframe of locations and names
    df_turbine = pd.DataFrame({
        'x':fi.layout_x,
        'y':fi.layout_y
    })

    # Get some info
    D = fi.floris.farm.rotor_diameters[0]
    N_turbine = df_turbine.shape[0]
    turbines = df_turbine.index

    # Set some color information
    if black_and_white:
        ec_color = 'k'
    else:
        ec_color = 'r'

    # If we're plotting the rotor, offset the label
    if plot_rotor:
        label_offset = D/2
    else:
        label_offset = 0.

    # If turbine names passed in apply them
    if turbine_names is not None:

        if len(turbine_names) != N_turbine:
            raise ValueError(
                "Length of turbine names array must equal number of turbines within fi"
            )

        df_turbine['turbine_names'] = turbine_names

    elif one_index_turbine:
        df_turbine['turbine_names'] = list(range(1,N_turbine+1)) # 1-indexed list
        df_turbine['turbine_names'] = df_turbine['turbine_names'].astype(int)

    else:

        df_turbine['turbine_names'] = list(range(N_turbine)) # 0-indexed list
        df_turbine['turbine_names'] = df_turbine['turbine_names'].astype(int)


    # if no axes provided, make one
    if not ax:
        fig, ax = plt.subplots(figsize=(7, 7))


    # Make ordered list of pairs sorted by distance if the distance
    # and angle matrices are provided
    if show_wake_lines:

        # Make a dataframe of distances
        dist = pd.DataFrame(
            squareform(pdist(df_turbine[['x','y']])),
            index=df_turbine.index,
            columns=df_turbine.index,
        )

        # Make a DF of turbine angles
        angle = pd.DataFrame()

        for t1 in turbines:
            for t2 in turbines:
                angle.loc[t1, t2] = wakeAngle(df_turbine, [t1, t2])
        angle.index.name = "Turbine"

        # Now limit the matrix to only show waking from (row) to (column)
        for t1 in turbines:
            for t2 in turbines:
                if dist.loc[t1, t2] == 0.0:
                    dist.loc[t1, t2] = np.nan
                    angle.loc[t1, t2] = np.nan

        ordList = pd.DataFrame()
        for t1 in turbines:
            for t2 in turbines:
                temp = pd.DataFrame(
                    {
                        "T1": [t1],
                        "T2": [t2],
                        "Dist": [dist.loc[t1, t2]],
                        "angle": angle.loc[t1, t2],
                    }
                )
                ordList = pd.concat([ordList, temp])

        ordList.dropna(how="any", inplace=True)
        ordList.sort_values("Dist", inplace=True, ascending=False)

        # If selected to limit the number of lines per turbine
        if lim_lines_per_turbine is not None:
            # Limit list to smallest lim_lines_per_turbine
            ordList = ordList.groupby(['T1'])
            ordList = ordList.apply(lambda x: x.nsmallest(n=lim_lines_per_turbine, columns='Dist'))
            ordList = ordList.reset_index(drop=True)

        # Add in the reflected version of each case (only postive directions will be
        # plotted to help test show face up)
        df_reflect = ordList.copy()
        df_reflect.columns = ['T2','T1','Dist','angle'] # Reflect T2 and T1
        ordList = pd.concat([ordList,df_reflect]).drop_duplicates().reset_index(drop=True)

        # If limiting to less than a certain distance
        if limit_dist_m is not None:
            ordList = ordList[ordList.Dist < limit_dist_m]

        # Plot wake lines and details
        for t1, t2 in zip(ordList.T1, ordList.T2):
            x = [df_turbine.loc[t1, "x"], df_turbine.loc[t2, "x"]]
            y = [df_turbine.loc[t1, "y"], df_turbine.loc[t2, "y"]]


            # Only plot positive x way
            if x[1] >= x[0]:
                continue

            if black_and_white:
                (line,) = ax.plot(x, y, color="k")
            else:
                (line,) = ax.plot(x, y)

            linetext = "%.2f D --- %.1f/%.1f" % (
                dist.loc[t1, t2] / D,
                np.min([angle.loc[t2, t1], angle.loc[t1, t2]]),
                np.max([angle.loc[t2, t1], angle.loc[t1, t2]]),
            )

            label_line(
                line, linetext, ax, near_i=1, near_x=None, near_y=None, rotation_offset=180
            )


    # If plotting rotors, mark the location of the nacelle
    if plot_rotor:
        ax.plot(df_turbine.x, df_turbine.y,'o',ls='None', color='k')

    # Also mark the place of each label to make sure figure is correct scale
    ax.plot(
        df_turbine.x + label_offset,
        df_turbine.y + label_offset,
        '.',
        ls='None',
        color='w',
        alpha=0

    )

    # Plot turbines
    for t1 in turbines:

        if plot_rotor:  # If plotting the rotors, draw these fist

            if not turbine_face_north: # Plot turbines facing west
                ax.plot(
                    [df_turbine.loc[t1].x, df_turbine.loc[t1].x],
                    [
                        df_turbine.loc[t1].y - 0.5 * D / 2.0,
                        df_turbine.loc[t1].y + 0.5 * D / 2.0,
                    ],
                    color="k",
                )
            else: # Plot facing north
                ax.plot(
                    [
                        df_turbine.loc[t1].x - 0.5 * D / 2.0,
                        df_turbine.loc[t1].x + 0.5 * D / 2.0,
                    ],
                    [df_turbine.loc[t1].y, df_turbine.loc[t1].y],
                    color="k",
                )

            # Draw a line from label to rotor
            ax.plot(
                    [
                        df_turbine.loc[t1].x,
                        df_turbine.loc[t1].x + D/2,
                    ],
                    [df_turbine.loc[t1].y, df_turbine.loc[t1].y + D/2],
                    color="k",
                    ls='--'
            )


        # Now add the label
        ax.text(

            df_turbine.loc[t1].x + label_offset,
            df_turbine.loc[t1].y + label_offset,
            df_turbine.turbine_names.values[t1],
            ha="center",
            bbox={"boxstyle": "round", "ec": ec_color, "fc": "white"}
        )

    ax.set_aspect("equal")


# Set wind direction
def set_direction(df_turbine, rotation_angle):
    """
    Rotate wind farm CCW by the given angle provided in degrees

    #TODO add center of rotation? Default = center of farm?

    Args:
        df_turbine (pd.DataFrame): turbine location data
        rotation_angle (float): rotation angle in degrees

    Returns:
        df_return (pd.DataFrame): rotated farm layout.
    """
    theta = np.deg2rad(rotation_angle)
    R = np.matrix([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    xy = np.array([df_turbine.x, df_turbine.y])

    xy_rot = R * xy

    df_return = df_turbine.copy(deep=True)
    df_return["x"] = np.squeeze(np.asarray(xy_rot[0, :]))
    df_return["y"] = np.squeeze(np.asarray(xy_rot[1, :]))
    return df_return


def turbineDist(df, turbList):
    """
    Derive distance between any two turbines.

    Args:
        df (pd.DataFrame): DataFrame with layout data.
        turbList (list): list of 2 turbines for which spacing distance
            is of interest.

    Returns:
        float: distance between turbines.
    """
    x1 = df.loc[turbList[0], "x"]
    x2 = df.loc[turbList[1], "x"]
    y1 = df.loc[turbList[0], "y"]
    y2 = df.loc[turbList[1], "y"]

    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return dist


def wakeAngle(df, turbList):
    """
    Get angles between turbines in wake direction

    Args:
        df (pd.DataFrame): DataFrame with layout data.
        turbList (list): list of 2 turbines for which spacing distance
            is of interest.

    Returns:
        wakeAngle (float): angle between turbines relative to compass
    """
    x1 = df.loc[turbList[0], "x"]
    x2 = df.loc[turbList[1], "x"]
    y1 = df.loc[turbList[0], "y"]
    y2 = df.loc[turbList[1], "y"]
    wakeAngle = (
        np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
    )  # Angle in normal cartesian coordinates

    # Convert angle to compass angle
    wakeAngle = 270.0 - wakeAngle
    if wakeAngle < 0:
        wakeAngle = wakeAngle + 360.0
    if wakeAngle > 360:
        wakeAngle = wakeAngle - 360.0

    return wakeAngle


def label_line(
    line,
    label_text,
    ax,
    near_i=None,
    near_x=None,
    near_y=None,
    rotation_offset=0.0,
    offset=(0, 0),
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

    Raises:
        ValueError: ("Need one of near_i, near_x, near_y") raised if
            insufficient information is passed in.
    """

    def put_label(i, ax):
        """
        Add a label to index.

        Args:
            i (int): index to label.
        """
        i = min(i, len(x) - 2)
        dx = sx[i + 1] - sx[i]
        dy = sy[i + 1] - sy[i]
        rotation = np.rad2deg(math.atan2(dy, dx)) + rotation_offset
        pos = [(x[i] + x[i + 1]) / 2.0 + offset[0], (y[i] + y[i + 1]) / 2 + offset[1]]
        ax.text(
            pos[0],
            pos[1],
            label_text,
            size=9,
            rotation=rotation,
            color=line.get_color(),
            ha="center",
            va="center",
            bbox={"ec": "1", "fc": "1", "alpha": 0.8},
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
        put_label(i, ax)
    elif near_x is not None:
        for i in range(len(x) - 2):
            if (x[i] < near_x and x[i + 1] >= near_x) or (
                x[i + 1] < near_x and x[i] >= near_x
            ):
                put_label(i, ax)
    elif near_y is not None:
        for i in range(len(y) - 2):
            if (y[i] < near_y and y[i + 1] >= near_y) or (
                y[i + 1] < near_y and y[i] >= near_y
            ):
                put_label(i, ax)
    else:
        raise ValueError("Need one of near_i, near_x, near_y")
