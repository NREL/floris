# Copyright 2020 NREL
 
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

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from floris.utilities import wrap_360
from ..utilities import setup_logger

# All functions assume a dataframe with index turbine, and columns x and y


def build_turbine_loc(turbine_x, turbine_y):
    """
    Make a DataFrame containing plant layout info
    (wind turbine locations).

    Args:
        turbine_x (np.array): wind turbine locations (east-west).
        turbine_y (np.array): wind turbine locations (north-south).

    Returns:
        turbineLoc (pd.DataFrame): turbine location data
    """
    turbineLoc = pd.DataFrame({'x': turbine_x, 'y': turbine_y})
    return turbineLoc


def visualize_layout(turbineLoc,
                     D,
                     ax=None,
                     show_wake_lines=False,
                     limit_dist=None,
                     turbine_face_north=False,
                     one_index_turbine=False):
    """
    Make a plot which shows the turbine locations, and important wakes.

    Args:
        turbineLoc (pd.DataFrame): turbine location data
        D (float): wind turbine rotor diameter.
        ax (:py:class:`matplotlib.pyplot.axes` optional):
            figure axes. Defaults to None.
        show_wake_lines (bool, optional): flag to control plotting of
            wake boundaries. Defaults to False.
        limit_dist (float, optional): Downstream limit to plot wakes. D
            Defaults to None.
        turbine_face_north (bool, optional): Force orientation of wind
            turbines. Defaults to False.
        one_index_turbine (bool, optional): if true, 1st turbine is turbine 1
    """

    turbines = turbineLoc.index.values

    # if no axes provided, make one
    if not ax:
        fig, ax = plt.subplots(figsize=(7, 7))

    # Plot turbine points
    # ax.scatter(turbineLoc.x,turbineLoc.y,alpha=0)

    # Make ordered list of pairs sorted by distance if the distance 
    # and angle matrices are provided
    if show_wake_lines:

        # Make a dataframe of distances
        dist = pd.DataFrame(
            squareform(pdist(turbineLoc)),
            index=turbineLoc.index,
            columns=turbineLoc.index
        )

        # Make a DF of turbine angles
        angle = pd.DataFrame()
        turbines = turbineLoc.index
        for t1 in turbines:
            for t2 in turbines:
                #d[t1,t2] = wakeAngle(turbineLoc,[t1,t2])
                angle.loc[t1, t2] = wakeAngle(turbineLoc, [t1, t2])
        angle.index.name = 'Turbine'

        # Now limit the matrix to only show waking from (row) to (column)
        for t1 in turbines:
            for t2 in turbines:
                if ((dist.loc[t1, t2] == 0.0)):
                    dist.loc[t1, t2] = np.nan
                    angle.loc[t1, t2] = np.nan

        ordList = pd.DataFrame()
        for t1 in turbines:
            for t2 in turbines:
                temp = pd.DataFrame({
                    'T1': [t1],
                    'T2': [t2],
                    'Dist': [dist.loc[t1, t2]],
                    'angle': angle.loc[t1, t2]
                })
                ordList = pd.concat([ordList, temp])

        ordList.dropna(how='any', inplace=True)
        ordList.sort_values('Dist', inplace=True, ascending=False)

        # Plot wake lines and details
        for t1, t2 in zip(ordList.T1, ordList.T2):
            x = [turbineLoc.loc[t1, 'x'], turbineLoc.loc[t2, 'x']]
            y = [turbineLoc.loc[t1, 'y'], turbineLoc.loc[t2, 'y']]

            if limit_dist:
                if dist.loc[t1, t2] > limit_dist:
                    continue

            # Only plot positive x way
            if x[1] > x[0]:
                continue

            l, = ax.plot(x, y)
            # linetext = '%.2f m --- %.2f D --- %.2f Deg --- %.2f Deg' % (
            #     dist.loc[t1,t2],
            #     dist.loc[t1,t2]/D,
            #     angle.loc[t1,t2],
            #     angle.loc[t2,t1]
            # )
            # linetext = '%.2f D --- %.2f Deg' % (dist.loc[t1, t2] / D,
            #                                     angle.loc[t2, t1])
            linetext = '%.2f D --- %.1f/%.1f' % (
                dist.loc[t1, t2] / D,
                np.min([angle.loc[t2, t1], angle.loc[t1, t2]]),
                np.max([angle.loc[t2, t1], angle.loc[t1, t2]])
            )
            # wrap_360(angle.loc[t2, t1]-180.))
            label_line(
                l,
                linetext,
                ax,
                near_i=1,
                near_x=None,
                near_y=None,
                rotation_offset=180
            )

    # Plot turbines
    for t1 in turbines:
        # print(t1)
        # ax.annotate(
        #     t1,
        #     (turbineLoc03.loc[t1].x,turbineLoc03.loc[t1].y),
        #     xycoords='data'
        # )
        if not turbine_face_north:
            ax.plot(
                [turbineLoc.loc[t1].x, turbineLoc.loc[t1].x],
                [
                    turbineLoc.loc[t1].y - 0.5 * D / 2.,
                    turbineLoc.loc[t1].y + 0.5 * D / 2.
                ],
                color='k'
            )
        else:
            ax.plot(
                [
                    turbineLoc.loc[t1].x - 0.5 * D / 2.,
                    turbineLoc.loc[t1].x + 0.5 * D / 2.
                ],
                [turbineLoc.loc[t1].y, turbineLoc.loc[t1].y],
                color='k'
            )
        if not one_index_turbine:
            ax.text(
                turbineLoc.loc[t1].x + D / 2,
                turbineLoc.loc[t1].y,
                t1,
                bbox=dict(boxstyle="round", ec='red', fc='white')
            )
        else:
            ax.text(
                turbineLoc.loc[t1].x + D / 2,
                turbineLoc.loc[t1].y,
                t1+1,
                bbox=dict(boxstyle="round", ec='red', fc='white')
            )
    ax.set_aspect('equal')


#Set wind direction
def set_direction(turbineLoc, rotation_angle):
    """
    Rotate wind farm CCW by the given angle provided in degrees

    #TODO add center of rotation? Default = center of farm?

    Args:
        turbineLoc (pd.DataFrame): turbine location data
        rotation_angle (float): rotation angle in degrees

    Returns:
        df_return (pd.DataFrame): rotated farm layout.
    """
    theta = np.deg2rad(rotation_angle)
    R = np.matrix([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta), np.cos(theta)]])

    xy = np.array([turbineLoc.x, turbineLoc.y])

    xy_rot = R * xy
    # return xy_rot
    #print(xy_rot)
    #print(xy_rot[0][0][0])
    df_return = turbineLoc.copy(deep=True)
    df_return['x'] = np.squeeze(np.asarray(xy_rot[0, :]))
    df_return['y'] = np.squeeze(np.asarray(xy_rot[1, :]))
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
    x1 = df.loc[turbList[0], 'x']
    x2 = df.loc[turbList[1], 'x']
    y1 = df.loc[turbList[0], 'y']
    y2 = df.loc[turbList[1], 'y']

    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

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
    x1 = df.loc[turbList[0], 'x']
    x2 = df.loc[turbList[1], 'x']
    y1 = df.loc[turbList[0], 'y']
    y2 = df.loc[turbList[1], 'y']
    wakeAngle = np.arctan2(
        y2 - y1,
        x2 - x1) * 180.0 / np.pi  # Angle in normal cartesian coordinates

    # Convert angle to compass angle
    wakeAngle = 270.0 - wakeAngle
    if wakeAngle < 0:
        wakeAngle = wakeAngle + 360.0
    if wakeAngle > 360:
        wakeAngle = wakeAngle - 360.0

    return wakeAngle


def label_line(line,
               label_text,
               ax,
               near_i=None,
               near_x=None,
               near_y=None,
               rotation_offset=0.0,
               offset=(0, 0)):
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

    def put_label(i):
        """
        Add a label to index.

        Args:
            i (int): index to label.
        """
        i = min(i, len(x) - 2)
        dx = sx[i + 1] - sx[i]
        dy = sy[i + 1] - sy[i]
        rotation = np.rad2deg(math.atan2(dy, dx)) + rotation_offset
        pos = [(x[i] + x[i + 1]) / 2. + offset[0],
               (y[i] + y[i + 1]) / 2 + offset[1]]
        plt.text(pos[0],
                 pos[1],
                 label_text,
                 size=9,
                 rotation=rotation,
                 color=line.get_color(),
                 ha="center",
                 va="center",
                 bbox=dict(ec='1', fc='1', alpha=0.8))

    # extract line data
    x = line.get_xdata()
    y = line.get_ydata()

    # define screen spacing
    if ax.get_xscale() == 'log':
        sx = np.log10(x)
    else:
        sx = x
    if ax.get_yscale() == 'log':
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
            if (x[i] < near_x and x[i + 1] >= near_x) or (x[i + 1] < near_x
                                                          and x[i] >= near_x):
                put_label(i)
    elif near_y is not None:
        for i in range(len(y) - 2):
            if (y[i] < near_y and y[i + 1] >= near_y) or (y[i + 1] < near_y
                                                          and y[i] >= near_y):
                put_label(i)
    else:
        raise ValueError("Need one of near_i, near_x, near_y")


def make_turbine_array(x,
                       y,
                       filename='turbineArrayProperties',
                       turbine='NREL5MWRef'):
    """
    Function to output a turbine array file given x and y locations.

    Args:
        x (np.array): wind turbine locations (east-west).
        y (np.array): wind turbine locations (north-south).
        filename (str, optional): write-to file path for wind turbine
            array info. Defaults to 'turbineArrayProperties'.
        turbine (str, optional): name of turbine to use within file.
            Defaults to 'NREL5MWRef'.
    """

    # Open the file for writing
    with open(filename, 'w') as f:

        # Write out the headerlines
        f.write(
            '/*--------------------------------*- C++ -*----------------------------------*\\\n'
        )
        f.write(
            '| =========                 |                                                 |\n'
        )
        f.write(
            '| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n'
        )
        f.write(
            '|  \\    /   O peration     | Version:  1.6                                   |\n'
        )
        f.write(
            '|   \\  /    A nd           | Web:      http://www.OpenFOAM.org               |\n'
        )
        f.write(
            '|    \\/     M anipulation  |                                                 |\n'
        )
        f.write(
            '\\*---------------------------------------------------------------------------*/\n'
        )
        f.write('FoamFile\n')
        f.write('{\n')
        f.write('    version     2.0;\n')
        f.write('    format      ascii;\n')
        f.write('    class       dictionary;\n')
        f.write('    object      turbineProperties;\n')
        f.write('}\n')
        f.write(
            '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n'
        )
        f.write('\n')
        f.write('globalProperties\n')
        f.write('{\n')
        f.write('    outputControl       "timeStep";\n')
        f.write('    outputInterval       1;\n')
        f.write('}\n')

        for idx, (x_val, y_val) in enumerate(zip(x, y)):
            f.write('\n')
            f.write('turbine%d\n' % idx)
            f.write('{\n')
            f.write('    turbineType         "%s";\n' % turbine)
            f.write('    baseLocation        (%.1f %.1f 0.0);\n' %
                    (x_val, y_val))
            f.write('    nRadial              64;\n')
            f.write('    azimuthMaxDis        2.0;\n')
            f.write('    nAvgSector           1;\n')
            f.write('    pointDistType       "uniform";\n')
            f.write('    pointInterpType     "linear";\n')
            f.write('    bladeUpdateType     "oldPosition";\n')
            f.write('    epsilon              20.0;\n')
            f.write('    forceScalar          1.0;\n')
            f.write('    inflowVelocityScalar 0.94;\n')
            f.write('    tipRootLossCorrType "Glauert";\n')
            f.write('    rotationDir         "cw";\n')
            f.write('    Azimuth              0.0;\n')
            f.write('    RotSpeed             13.0;\n')
            f.write('    TorqueGen            20000.0;\n')
            f.write('    Pitch                0.0;\n')
            f.write('    NacYaw               270.0;\n')
            f.write('    fluidDensity         1.225;\n')
            f.write('}\n')
