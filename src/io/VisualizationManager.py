"""
Copyright 2017 NREL

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from models.Coordinate import Coordinate
import matplotlib.pyplot as plt
import numpy as np


class VisualizationManager():
    """
    The VisualizationManager handles all of the lower level visualization instantiation 
    and data management. Currently, it produces 2D matplotlib plots for a given plane 
    of data.

    IT IS IMPORTANT to note that this class should be treated as a singleton. That is, 
    only one instance of this class should exist.
    """
    
    def __init__(self):
        self.figure_count = 0

    def _set_texts(self, plot_title, horizontal_axis_title, vertical_axis_title):
        fontsize = 15
        plt.title(plot_title, fontsize=fontsize)
        plt.xlabel(horizontal_axis_title, fontsize=fontsize)
        plt.ylabel(vertical_axis_title, fontsize=fontsize)

    def _set_colorbar(self):
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=15)

    def _set_axis(self):
        plt.axis('equal')
        plt.tick_params(which='both', labelsize=15)

    def _new_figure(self):
        plt.figure(self.figure_count)
        self.figure_count += 1

    def _new_filled_contour(self, xmesh, ymesh, data):
        self._new_figure()
        # vmin = np.amin(data)
        # vmax = np.amax(data)
        plt.contourf(xmesh, ymesh, data, 50,
                            cmap='coolwarm')#, vmin=vmin, vmax=vmax)

    def plot_constant_z(self, xmesh, ymesh, data):
        self._new_filled_contour(xmesh, ymesh, data)        

        # configure the plot
        self._set_texts("Constant Height", "x (m)", "y (m)")
        self._set_colorbar()
        self._set_axis()

        plt.grid(True)

    def add_turbine_marker(self, turbine, coords):
        
        a = Coordinate(coords.x, coords.y - turbine.rotorRadius)
        b = Coordinate(coords.x, coords.y + turbine.rotorRadius)

        a.rotate(turbine.yawAngle, (coords.x, coords.y))
        b.rotate(turbine.yawAngle, (coords.x, coords.y))

        plt.plot([a.xprime, b.xprime], [a.yprime, b.yprime],  'k', linewidth=1)

        # x = [-50, 1000]
        # y = [coords.y - radius, coords.y - radius]
        # plt.plot(x, y,  'b', linewidth=1)

        # x = [-50, 1000]
        # y = [coords.y + radius, coords.y + radius]
        # plt.plot(x, y,  'b', linewidth=1)

    def show(self):
        plt.show()
