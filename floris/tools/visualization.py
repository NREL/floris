# Copyright 2017 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from ..utilities import Vec3
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np


def plot_turbines(ax, layout_x, layout_y, yaw_angles, D):

    for x, y, yaw in zip(layout_x,layout_y,yaw_angles):
        R = D/2.
        x_0 = x + np.sin(np.deg2rad(yaw)) * R
        x_1 = x - np.sin(np.deg2rad(yaw)) * R
        y_0 = y - np.cos(np.deg2rad(yaw)) * R
        y_1 = y + np.cos(np.deg2rad(yaw)) * R
        ax.plot([x_0,x_1],[y_0,y_1],color='k')

def line_contour_cut_plane(cut_plane,ax=None,levels=None,colors=None,**kwargs):
    """ Visualize the scan as a simple contour
    
    Args:
        ax: axes for plotting, if none, create a new one  
        minSpeed, maxSpeed, values used for plotting, if not provide assume to data max min
    """
    if not ax:
        fig, ax = plt.subplots()
    
    # Reshape UMesh internally
    u_mesh = cut_plane.u_mesh.reshape(cut_plane.resolution[1], cut_plane.resolution[0])
    Zm = np.ma.masked_where(np.isnan(u_mesh), u_mesh)
    rcParams['contour.negative_linestyle'] = 'solid'
    
    # # Plot the cut-through
    ax.contour(cut_plane.x1_lin,cut_plane.x2_lin, Zm,levels=levels,colors=colors,**kwargs)

    # Make equal axis
    ax.set_aspect('equal')


def visualize_cut_plane(cut_plane, ax=None, minSpeed=None, maxSpeed=None, cmap='coolwarm'):
    """ Visualize the scan
    
    Args:
        ax: axes for plotting, if none, create a new one  
        minSpeed, maxSpeed, values used for plotting, if not provide assume to data max min
    """
    if not ax:
        fig, ax = plt.subplots()
    if minSpeed is None:
        minSpeed = cut_plane.u_mesh.min()
    if maxSpeed is None:
        maxSpeed = cut_plane.u_mesh.max()

    # Reshape UMesh internally
    u_mesh = cut_plane.u_mesh.reshape(cut_plane.resolution[1], cut_plane.resolution[0])
    Zm = np.ma.masked_where(np.isnan(u_mesh), u_mesh)

    # Plot the cut-through
    im = ax.pcolormesh(cut_plane.x1_lin, cut_plane.x2_lin, Zm, cmap=cmap, vmin=minSpeed, vmax=maxSpeed)

    # Add line contour
    line_contour_cut_plane(cut_plane,ax=ax,levels=None,colors='w',linewidths=0.8,alpha=0.3)

    # Make equal axis
    ax.set_aspect('equal')

    # Return im
    return im

def reverse_cut_plane_x_axis_in_plot(ax):
    ax.invert_xaxis()