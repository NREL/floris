#
# Copyright 2019 NREL
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
#

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


class _CutPlane():

    def __init__(self,
                 flow_field,
                 x1='x',
                 x2='y',
                 x3_value=None
                 ):

        # Assign the axis names
        self.x1_name = x1
        self.x2_name = x2
        #TODO: if it will be assumed that x3 is one of x, y, or z that is x1 and x2,
        #       then we should verify that x1 and x2 are one of x, y, or z
        self.x3_name = [x3 for x3 in ['x', 'y', 'z'] if x3 not in [x1, x2]][0]

        # Get x1, x2 and x3 arrays
        x1_array = getattr(flow_field, self.x1_name)
        x2_array = getattr(flow_field, self.x2_name)
        x3_array = getattr(flow_field, self.x3_name)

        search_values = np.array(sorted(np.unique(x3_array)))
        nearest_idx = (np.abs(search_values-x3_value)).argmin()
        nearest_value = search_values[nearest_idx]
        print('Nearest value in %s to %.2f is %.2f' %
              (self.x3_name, x3_value, nearest_value))

        # Select down the data
        x3_select_mask = x3_array == nearest_value

        # Store the un-interpolated input arrays at this slice 
        self.x1_in = x1_array[x3_select_mask]
        self.x2_in = x2_array[x3_select_mask]
        self.u_in = flow_field.u[x3_select_mask]
        self.v_in = flow_field.v[x3_select_mask]
        self.w_in = flow_field.w[x3_select_mask]

        # Initially, x1_lin, x2_lin are unique values of input
        self.x1_lin = np.unique(self.x1_in)
        self.x2_lin = np.unique(self.x2_in)

        # Save the resolution as the number of unique points in x1 and x2
        self.resolution = (len(np.unique(self.x1_lin)), len(np.unique(self.x2_lin)))

        # Make initial meshing
        self._remesh()


    def _remesh(self):

        # Mesh and interpolate u, v and w
        self.x1_mesh, self.x2_mesh = np.meshgrid(self.x1_lin, self.x2_lin)
        self.u_mesh = griddata(np.column_stack([self.x1_in, self.x2_in]), self.u_in, (
            self.x1_mesh.flatten(), self.x2_mesh.flatten()), method='cubic')
        self.v_mesh = griddata(np.column_stack([self.x1_in, self.x2_in]), self.v_in, (
            self.x1_mesh.flatten(), self.x2_mesh.flatten()), method='cubic')
        self.w_mesh = griddata(np.column_stack([self.x1_in, self.x2_in]), self.w_in, (
            self.x1_mesh.flatten(), self.x2_mesh.flatten()), method='cubic')

        # Save flat vectors
        self.x1_flat = self.x1_mesh.flatten()
        self.x2_flat = self.x2_mesh.flatten()

        # Save u-cubed
        self.u_cubed = self.u_mesh ** 3





# Define horizontal subclass
class HorPlane(_CutPlane):

    def __init__(self, flow_field, z_value):

        # Set up call super
        super().__init__(flow_field, x1='x', x2='y', x3_value=z_value)

# Define cross plane subclass
class CrossPlane(_CutPlane):

    def __init__(self, flow_field, x_value):

        # Set up call super
        super().__init__(flow_field, x1='y', x2='z', x3_value=x_value)






## Modification functions
def set_origin(cut_plane, center_x1=0.0, center_x2=0.0):
        # Store the un-interpolated input arrays at this slice 
        cut_plane.x1_in = cut_plane.x1_in - center_x1
        cut_plane.x2_in = cut_plane.x2_in - center_x2
        cut_plane.x1_lin = cut_plane.x1_lin - center_x1
        cut_plane.x2_lin = cut_plane.x2_lin - center_x2

        # Remesh
        cut_plane._remesh()

        return cut_plane


def change_resolution(cut_plane, resolution=(100,100)):
        
        # Grid the data
        cut_plane.x1_lin = np.linspace(
                min(cut_plane.x1_in), max(cut_plane.x1_in), resolution[0])
        cut_plane.x2_lin = np.linspace(
                min(cut_plane.x2_in), max(cut_plane.x2_in), resolution[1])

        # Save the new resolution
        cut_plane.resolution = resolution

        # Redo the mesh
        cut_plane._remesh()

        # Return the cutplane
        return cut_plane

def interpolate_onto_array(cut_plane, x1_array, x2_array):

        # Grid the data given array
        cut_plane.x1_lin = x1_array
        cut_plane.x2_lin = x2_array

        # Save the new resolution
        cut_plane.resolution = (len(np.unique(cut_plane.x1_lin)), len(np.unique(cut_plane.x2_lin)))

        # Redo the mesh
        cut_plane._remesh()

        # Return the cutplane
        return cut_plane

def rescale_axis(cut_plane,x1_factor=1.0,x2_factor=1.0):
        # Store the un-interpolated input arrays at this slice 
        cut_plane.x1_in = cut_plane.x1_in / x1_factor
        cut_plane.x2_in = cut_plane.x2_in / x2_factor
        cut_plane.x1_lin = cut_plane.x1_lin / x1_factor
        cut_plane.x2_lin = cut_plane.x2_lin / x2_factor

        # Remesh
        cut_plane._remesh()

        return cut_plane

def calculate_wind_speed(cross_plane, x1_loc, x2_loc, R):

    # Make a distance column
    distance = np.sqrt((cross_plane.x1_flat - x1_loc)**2 +
                        (cross_plane.x2_flat - x2_loc)**2)

    # Return the mean wind speed
    return np.cbrt(np.mean(cross_plane.u_cubed[distance < R]))

def calculate_power(cross_plane, x1_loc, x2_loc, R, ws_array, cp_array,air_density=1.225):

    # Compute the ws
    ws = calculate_wind_speed(cross_plane, x1_loc, x2_loc, R)

    # Compute the cp
    cp_value = np.interp(ws,ws_array,cp_array)

    #Return the power
    return 0.5 * air_density * (np.pi * R**2) * cp_value * ws**3

    # def get_profile(self, R, x2_loc, resolution=100, x1_locs=None):
    #     if x1_locs is None:
    #         x1_locs = np.linspace(
    #             min(self.x1_flat), max(self.x1_flat), resolution)
    #     v_array = np.array([self.calculate_wind_speed(
    #         x1_loc, x2_loc, R) for x1_loc in x1_locs])
    #     return x1_locs, v_array)

    # def get_power_profile(self, ws_array, cp_array, rotor_radius, air_density=1.225, resolution=100, x1_locs=None):

    #     # Get the wind speed profile
    #     x1_locs, v_array = self.get_profile(resolution=resolution, x1_locs=x1_locs)

    #     # Get Cp
    #     cp_array = np.interp(v_array,ws_array,cp_array)

    #     # Return power array
    #     return x1_locs, 0.5 * air_density * (np.pi * rotor_radius**2) * cp_array * v_array**3
