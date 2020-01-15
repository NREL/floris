# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pandas as pd

def nudge_outward(x):
    """
    Avoid numerical issue in grid data by sligly expanding input x,y

    Args:
        x (np.arraym float): Vector to be slightly expanded
    """
    nudge_val = 0.001
    min_x = np.min(x)
    max_x = np.max(x)
    x = np.where(x==min_x, min_x-nudge_val, x) 
    x = np.where(x==max_x, max_x+nudge_val, x) 
    return x

def get_plane_from_flow_data(flow_data,
                            normal_vector='z',
                            x3_value=100):
    """
    Get a plane of data, in form of dataframe, from a flow_data object
    This is used to get planes from SOWFA results and FLORIS sims with fixed grids, ie curl

    Args:
        flow_data (np.array): 3D vector field of velocity data
        normal_vector (string, optional): vector normal to plane
            Defaults to z.
        x3_value (float, optional): value of normal vector to slice through
            Defaults to 100.

    Returns:
        dataframe of x1,x2,u,v,w values
    """
    order = "f"
    if normal_vector == 'z':
        x1_array = flow_data.x.flatten(order=order)
        x2_array = flow_data.y.flatten(order=order)
        x3_array = flow_data.z.flatten(order=order)

    if normal_vector == 'x':
        x3_array = flow_data.x.flatten(order=order)
        x1_array = flow_data.y.flatten(order=order)
        x2_array = flow_data.z.flatten(order=order)

    if normal_vector == 'y':
        x3_array = flow_data.y.flatten(order=order)
        x1_array = flow_data.x.flatten(order=order)
        x2_array = flow_data.z.flatten(order=order)

    u = flow_data.u.flatten(order=order)
    v = flow_data.v.flatten(order=order)
    w = flow_data.w.flatten(order=order)

    search_values = np.array(sorted(np.unique(x3_array)))
    nearest_idx = (np.abs(search_values - x3_value)).argmin()
    nearest_value = search_values[nearest_idx]
    print('Nearest value to %.2f is %.2f' %
        (x3_value, nearest_value))

    # Select down the data
    x3_select_mask = x3_array == nearest_value

    # Store the un-interpolated input arrays at this slice
    x1 = x1_array[x3_select_mask]
    x2 = x2_array[x3_select_mask]
    x3 = np.ones_like(x1) * x3_value
    u = u[x3_select_mask]
    v = v[x3_select_mask]
    w = w[x3_select_mask]

    df = pd.DataFrame({'x1':x1,
                'x2':x2,
                'x3':x3,
                'u':u,
                'v':v,
                'w':w
                })
    return df


class CutPlane():
    def __init__(self, df):
        """
        Initialize CutPlane object. Used to extract a 2D plane from a
        3D vectoral velocity field

        Args:
            df (pd.DataFrame): Pandas DataFrame of data with columns x1,x2, u,v,w
        """
        self.df = df

        # Save the resolution as the number of unique points in x1 and x2
        self.resolution = (len(self.df.x1.unique()),
                           len(self.df.x2.unique()))



# Modification functions
def set_origin(cut_plane, center_x1=0.0, center_x2=0.0):
    """
    Establish the origin of a CutPlane object.

    Args:
        cut_plane (:py:class:`floris.tools.cut_plane._CutPlane`):
            plane of data.
        center_x1 (float, optional): x1-coordinate of orign.
                Defaults to 0.0.
        center_x2 (float, optional): x2-coordinate of orign.
                Defaults to 0.0.

    Returns:
        cut_plane (:py:class:`floris.tools.cut_plane._CutPlane`):
                updated plane of data.
    """
    # Store the un-interpolated input arrays at this slice
    cut_plane.df.x1 = cut_plane.df.x1 - center_x1
    cut_plane.df.x2 = cut_plane.df.x2 - center_x2

    return cut_plane


def change_resolution(cut_plane, resolution=(100, 100)):
    """
    Modify default resolution of a CutPlane object.

    Args:
        cut_plane (:py:class:`floris.tools.cut_plane._CutPlane`):
            plane of data.
        resolution (tuple, optional): Desired resolution in x1 and x2.
            Defaults to (100, 100).

    Returns:
        cut_plane (:py:class:`floris.tools.cut_plane._CutPlane`):
                updated plane of data.
    """

    # Linearize the data
    x1_lin = np.linspace(min(cut_plane.df.x1), max(cut_plane.df.x1),
                                   resolution[0])
    x2_lin = np.linspace(min(cut_plane.df.x2), max(cut_plane.df.x2),
                                   resolution[1])
    # x3 = np.ones_like(x1) * cut_plane.df.x3[0]

    # Mesh the data
    x1_mesh, x2_mesh = np.meshgrid(x1_lin, x2_lin)
    x3_mesh = np.ones_like(x1_mesh) * cut_plane.df.x3[0]

    # Interpolate u,v,w
    u_mesh = griddata(
            np.column_stack([nudge_outward(cut_plane.df.x1), nudge_outward(cut_plane.df.x2)]),
            cut_plane.df.u.values, (x1_mesh.flatten(), x2_mesh.flatten()),
            method='cubic')
    v_mesh = griddata(
            np.column_stack([nudge_outward(cut_plane.df.x1), nudge_outward(cut_plane.df.x2)]),
            cut_plane.df.v.values, (x1_mesh.flatten(), x2_mesh.flatten()),
            method='cubic')

    w_mesh = griddata(
            np.column_stack([nudge_outward(cut_plane.df.x1), nudge_outward(cut_plane.df.x2)]),
            cut_plane.df.w.values, (x1_mesh.flatten(), x2_mesh.flatten()),
            method='cubic')


    # Assign back to df
    cut_plane.df =   pd.DataFrame({'x1':x1_mesh.flatten(),
                'x2':x2_mesh.flatten(),
                'x3':x3_mesh.flatten(),
                'u':u_mesh.flatten(),
                'v':v_mesh.flatten(),
                'w':w_mesh.flatten()
                })

    # Save the resolution
    cut_plane.resolution = resolution

    # Return the cutplane
    return cut_plane


def interpolate_onto_array(cut_plane, x1_array, x2_array):
    """
    Interpolate a CutPlane object onto specified coordinate arrays.

    Args:
        cut_plane (:py:class:`floris.tools.cut_plane._CutPlane`):
            plane of data.
        x1_array (np.array): specified x1-coordinate.
        x2_array (np.array): specified x2-coordinate.

    Returns:
        cut_plane (:py:class:`floris.tools.cut_plane._CutPlane`):
                updated plane of data.
    """

    # Linearize the data
    x1_lin = x1_array
    x2_lin = x2_array

    # Save the new resolution
    cut_plane.resolution = (len(np.unique(x1_lin)),
                            len(np.unique(x2_lin)))

    # Mesh the data
    x1_mesh, x2_mesh = np.meshgrid(x1_lin, x2_lin)
    x3_mesh = np.ones_like(x1_mesh) * cut_plane.df.x3[0]

    # Interpolate u,v,w
    u_mesh = griddata(
            np.column_stack([nudge_outward(cut_plane.df.x1), nudge_outward(cut_plane.df.x2)]),
            cut_plane.df.u.values, (x1_mesh.flatten(), x2_mesh.flatten()),
            method='cubic')
    v_mesh = griddata(
            np.column_stack([nudge_outward(cut_plane.df.x1), nudge_outward(cut_plane.df.x2)]),
            cut_plane.df.v.values, (x1_mesh.flatten(), x2_mesh.flatten()),
            method='cubic')

    w_mesh = griddata(
            np.column_stack([nudge_outward(cut_plane.df.x1), nudge_outward(cut_plane.df.x2)]),
            cut_plane.df.w.values, (x1_mesh.flatten(), x2_mesh.flatten()),
            method='cubic')


    # Assign back to df
    cut_plane.df =   pd.DataFrame({'x1':x1_mesh.flatten(),
                'x2':x2_mesh.flatten(),
                'x3':x3_mesh.flatten(),
                'u':u_mesh.flatten(),
                'v':v_mesh.flatten(),
                'w':w_mesh.flatten()
                })

    # Return the cutplane
    return cut_plane


def rescale_axis(cut_plane, x1_factor=1.0, x2_factor=1.0):
    """
    Stretch or compress CutPlane coordinates.

    Args:
        cut_plane (:py:class:`floris.tools.cut_plane._CutPlane`):
            plane of data.
        x1_factor (float): scaling factor for x1-coordinate.
        x2_factor (float): scaling factor for x2-coordinate.

    Returns:
        cut_plane (:py:class:`floris.tools.cut_plane._CutPlane`):
                updated plane of data.
    """
    # Store the un-interpolated input arrays at this slice
    cut_plane.df.x1 = cut_plane.df.x1 / x1_factor
    cut_plane.df.x2 = cut_plane.df.x2 / x2_factor


    return cut_plane


# def calculate_wind_speed(cross_plane, x1_loc, x2_loc, R):
#     """
#     Calculate effective wind speed within specified range of a point.

#     Args:
#         cross_plane (:py:class:`floris.tools.cut_plane.CrossPlane`):
#             plane of data.
#         x1_loc (float): x1-coordinate of point of interst.
#         x2_loc (float): x2-coordinate of point of interst.
#         R (float): radius from point of interst to consider

#     Returns:
#         (float): effective wind speed
#     """
#     # Make a distance column
#     distance = np.sqrt((cross_plane.x1_flat - x1_loc)**2 +
#                        (cross_plane.x2_flat - x2_loc)**2)

#     # Return the mean wind speed
#     return np.cbrt(np.mean(cross_plane.u_cubed[distance < R]))

# def wind_speed_profile(cross_plane,
#                         R, 
#                         x2_loc, 
#                         resolution=100, 
#                         x1_locs=None):

#     if x1_locs is None:
#         x1_locs = np.linspace(
#             min(cross_plane.x1_flat), max(cross_plane.x1_flat), resolution)
#     v_array = np.array([calculate_wind_speed(cross_plane,x1_loc, x2_loc, R) for x1_loc in x1_locs])
#     return x1_locs, v_array

# def calculate_power(cross_plane,
#                     x1_loc,
#                     x2_loc,
#                     R,
#                     ws_array,
#                     cp_array,
#                     air_density=1.225):
#     """
#     Calculate maximum power available in a given cross plane.

#     Args:
#         cross_plane (:py:class:`floris.tools.cut_plane.CrossPlane`):
#             plane of data.
#         x1_loc (float): x1-coordinate of point of interst.
#         x2_loc (float): x2-coordinate of point of interst.
#         R (float): Radius of wind turbine rotor.
#         ws_array (np.array): reference wind speed for cp curve.
#         cp_array (np.array): cp curve at reference wind speeds.
#         air_density (float, optional): air density. Defaults to 1.225.

#     Returns:
#         float: Power!
#     """
#     # Compute the ws
#     ws = calculate_wind_speed(cross_plane, x1_loc, x2_loc, R)

#     # Compute the cp
#     cp_value = np.interp(ws, ws_array, cp_array)

#     #Return the power
#     return 0.5 * air_density * (np.pi * R**2) * cp_value * ws**3


#     # def get_power_profile(self, ws_array, cp_array, rotor_radius, air_density=1.225, resolution=100, x1_locs=None):

#     #     # Get the wind speed profile
#     #     x1_locs, v_array = self.get_profile(resolution=resolution, x1_locs=x1_locs)

#     #     # Get Cp
#     #     cp_array = np.interp(v_array,ws_array,cp_array)

#     #     # Return power array
#     #     return x1_locs, 0.5 * air_density * (np.pi * rotor_radius**2) * cp_array * v_array**3





# # Define horizontal subclass
# class HorPlane(_CutPlane):
#     """
#     Subclass of _CutPlane. Shortcut to extracting a horizontal plane.
#     """

#     def __init__(self, df):
#         """
#         Initialize horizontal CutPlane

#         Args:
#             flow_data (np.array): 3D vector field of velocity data
#             z_value (float): vertical position through which to slice
#         """
#         # Set up call super
#         super().__init__(df)


# # Define cross plane subclass
# class CrossPlane(_CutPlane):
#     """
#     Subclass of _CutPlane. Shortcut to extracting a cross-stream plane.
#     """

#     def __init__(self, df):
#         """
#         Initialize cross-stream CutPlane

#         Args:
#             flow_data (np.array): 3D vector field of velocity data
#             x_value (float): streamwise position through which to slice
#         """
#         # Set up call super
#         super().__init__(df)


# # Define cross plane subclass
# class VertPlane(_CutPlane):
#     """
#     Subclass of _CutPlane. Shortcut to extracting a streamwise-vertical plane.
#     """

#     def __init__(self, df):
#         """
#         Initialize streamwise-vertical  CutPlane

#         Args:
#             flow_data (np.array): 3D vector field of velocity data
#             y_value (float): spanwise position through which to slice
#         """
#         # Set up call super
#         super().__init__(df)
