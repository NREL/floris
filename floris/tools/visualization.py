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
 

from ..utilities import Vec3
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np


def plot_turbines(ax, layout_x, layout_y, yaw_angles, D, color=None, 
        wind_direction=270.):
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

    # Correct for the wind direction
    yaw_angles = np.array(yaw_angles) - wind_direction - 270

    if color == None: color = 'k'
    for x, y, yaw in zip(layout_x, layout_y, yaw_angles):
        R = D / 2.
        x_0 = x + np.sin(np.deg2rad(yaw)) * R
        x_1 = x - np.sin(np.deg2rad(yaw)) * R
        y_0 = y - np.cos(np.deg2rad(yaw)) * R
        y_1 = y + np.cos(np.deg2rad(yaw)) * R
        ax.plot([x_0, x_1], [y_0, y_1], color=color)

def plot_turbines_with_fi(ax,fi,color=None):
    """
    Wrapper function to plot turbines which extracts the data
    from a FLORIS interface object

    Args:
        ax (:py:class:`matplotlib.pyplot.axes`): figure axes. Defaults 
            to None.
        fi (:py:class:`floris.tools.flow_data.FlowData`):
                FlowData object.
        color (str, optional): Color to plot turbines
    """
    # Grab D
    for i, turbine in enumerate(fi.floris.farm.turbines):
        D = turbine.rotor_diameter
        break

    plot_turbines(ax, fi.layout_x, fi.layout_y, fi.get_yaw_angles(), D, color=color,  wind_direction=fi.floris.farm.wind_map.input_direction)


def line_contour_cut_plane(cut_plane,
                           ax=None,
                           levels=None,
                           colors=None,
                           **kwargs):
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

    # Reshape UMesh internally
    x1_mesh = cut_plane.df.x1.values.reshape(cut_plane.resolution[1],
                                             cut_plane.resolution[0])
    x2_mesh = cut_plane.df.x2.values.reshape(cut_plane.resolution[1],
                                             cut_plane.resolution[0])
    u_mesh = cut_plane.df.u.values.reshape(cut_plane.resolution[1],
                                           cut_plane.resolution[0])
    Zm = np.ma.masked_where(np.isnan(u_mesh), u_mesh)
    rcParams['contour.negative_linestyle'] = 'solid'

    # # Plot the cut-through
    ax.contour(x1_mesh, x2_mesh, Zm, levels=levels, colors=colors, **kwargs)

    # Make equal axis
    ax.set_aspect('equal')


def visualize_cut_plane(cut_plane,
                        ax=None,
                        minSpeed=None,
                        maxSpeed=None,
                        cmap='coolwarm',
                        levels=None):
    """
    Generate pseudocolor mesh plot of the cut_plane.

    Args:
        cut_plane (:py:class:`~.tools.cut_plane.CutPlane`): 2D 
            plane through wind plant.
        ax (:py:class:`matplotlib.pyplot.axes`): Figure axes. Defaults 
            to None.
        minSpeed (float, optional): Minimum value of wind speed for
            contours. Defaults to None.
        maxSpeed (float, optional): Maximum value of wind speed for
            contours. Defaults to None.
        cmap (str, optional): Colormap specifier. Defaults to 
            'coolwarm'.

    Returns:
        im (:py:class:`matplotlib.plt.pcolormesh`): Image handle.
    """

    if not ax:
        fig, ax = plt.subplots()
    if minSpeed is None:
        minSpeed = cut_plane.df.u.min()
    if maxSpeed is None:
        maxSpeed = cut_plane.df.u.max()

    # Reshape to 2d for plotting
    x1_mesh = cut_plane.df.x1.values.reshape(cut_plane.resolution[1],
                                             cut_plane.resolution[0])
    x2_mesh = cut_plane.df.x2.values.reshape(cut_plane.resolution[1],
                                             cut_plane.resolution[0])
    u_mesh = cut_plane.df.u.values.reshape(cut_plane.resolution[1],
                                           cut_plane.resolution[0])
    Zm = np.ma.masked_where(np.isnan(u_mesh), u_mesh)

    # Plot the cut-through
    im = ax.pcolormesh(x1_mesh,
                       x2_mesh,
                       Zm,
                       cmap=cmap,
                       vmin=minSpeed,
                       vmax=maxSpeed)

    # Add line contour
    line_contour_cut_plane(cut_plane,
                           ax=ax,
                           levels=levels,
                           colors='w',
                           linewidths=0.8,
                           alpha=0.3)

    # Make equal axis
    ax.set_aspect('equal')

    # Return im
    return im


def visualize_quiver(cut_plane,
                     ax=None,
                     minSpeed=None,
                     maxSpeed=None,
                     downSamp=1,
                     **kwargs):
    """
        Visualize the in-plane flows in a cut_plane using quiver.

        Args:
            cut_plane (:py:class:`~.tools.cut_plane.CutPlane`): 2D 
                plane through wind plant.
            ax (:py:class:`matplotlib.pyplot.axes`): Figure axes. Defaults 
                to None.
            minSpeed (float, optional): Minimum value of wind speed for
                contours. Defaults to None.
            maxSpeed (float, optional): Maximum value of wind speed for
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
    x1_mesh = cut_plane.df.x1.values.reshape(cut_plane.resolution[1],
                                                cut_plane.resolution[0])
    x2_mesh = cut_plane.df.x2.values.reshape(cut_plane.resolution[1],
                                                cut_plane.resolution[0])
    v_mesh = cut_plane.df.v.values.reshape(cut_plane.resolution[1],
                                            cut_plane.resolution[0])
    w_mesh = cut_plane.df.w.values.reshape(cut_plane.resolution[1],
                                            cut_plane.resolution[0])

    # plot the stream plot
    QV1 = ax.streamplot((x1_mesh[::downSamp, ::downSamp]),
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