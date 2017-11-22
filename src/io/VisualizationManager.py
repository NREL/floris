import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.Turbine import Turbine
from models.Wake import Wake
from models.Farm import Farm
import matplotlib.pyplot as plt
import numpy as np


class VisualizationManager():

    def __init__(self):
        self.is_great = True

    def _plot_plane(self):
        plot = plt.figure()

    # def plot_constant_x(self, flowfield):


    # def plot_constant_y(self, flowfield):


    def plot_constant_z(self, gridX, gridY, data, turbine):
        plt.figure(figsize=(30, 10))
        
        # plot the flow field data
        plt.contourf(gridX, gridY, data[:, :, 24], 50, cmap='coolwarm', vmin=3.0, vmax=np.amax(data[:, :, 24]) )

        # plot the turbine marker
        plt.plot([0, 0], [-turbine.rotorDiameter / 2., turbine.rotorDiameter / 2.],  'k', linewidth=3)

        # plot configuration
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=15)
        plt.axis('equal')
        plt.xlabel('x(m)', fontsize=15)
        plt.ylabel('y(m)', fontsize=15)
        plt.tick_params(which='both', labelsize=15)
        plt.title('Horizontal', fontsize=15)

        # display the plot
        plt.show()
