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

    def add_turbine_marker(self, radius, coords):
        x = [coords.x, coords.x]
        y = [coords.y - radius, coords.y + radius]
        plt.plot(x, y,  'k', linewidth=1)

        # x = [-50, 1000]
        # y = [coords.y - radius, coords.y - radius]
        # plt.plot(x, y,  'b', linewidth=1)

        # x = [-50, 1000]
        # y = [coords.y + radius, coords.y + radius]
        # plt.plot(x, y,  'b', linewidth=1)

    def show_plot(self):
        plt.show()
