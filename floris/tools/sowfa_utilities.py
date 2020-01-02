# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np
from .flow_data import FlowData
from ..utilities import Vec3
import pandas as pd
import os
import re
from .cut_plane import CutPlane, get_plane_from_flow_data


class SowfaInterface():
    """
    Object to facilitate interaction with flow data output by SOWFA.

    Returns:
        :py:class:`floris.tools.sowfa_utilities.SowfaInterface`: object
    """

    def __init__(self,
                 case_folder,
                 flow_data_sub_path='array_mean/array.mean0D_UAvg.vtk',
                 setup_sub_path='setUp',
                 turbine_array_sub_path='constant/turbineArrayProperties',
                 turbine_sub_path='constant/turbineProperties',
                 controlDict_sub_path='system/controlDict',
                 turbine_output_sub_path='turbineOutput/20000',
                 assumed_settling_time=None):
        """
        SowfaInterface object init method.

        Args:
            case_folder (str): path to folder containing SOWFA data
            flow_data_sub_path (str, optional): path to mean data.
                Defaults to 'array_mean/array.mean0D_UAvg.vtk'.
            setup_sub_path (str, optional): path to setup info.
                Defaults to 'setUp'.
            turbine_array_sub_path (str, optional): path to wind plant
                info. Defaults to 'constant/turbineArrayProperties'.
            turbine_sub_path (str, optional): path to wind turbine
                info. Defaults to 'constant/turbineProperties'.
            controlDict_sub_path (str, optional): path to turbine
                controls info. Defaults to 'system/controlDict'.
            turbine_output_sub_path (str, optional): path to turbine
                operational data. Defaults to 'turbineOutput/20000'.
            assumed_settling_time (float, optional): Time to account
                for startup transients in simulation. Defaults to None.
        """
        print(case_folder)

        # Save the case_folder and sub_paths
        self.case_folder = case_folder
        self.setup_sub_path = setup_sub_path
        self.turbine_array_sub_path = turbine_array_sub_path
        self.turbine_sub_path = turbine_sub_path
        self.controlDict_sub_path = controlDict_sub_path
        self.turbine_output_sub_path = turbine_output_sub_path

        # Read in the input files

        # Get control settings from sc input file
        #TODO Assuming not dynamic and only one setting applied for each turbine
        #TODO If not using the super controller sowfa variant, need alternative

        # Get the turbine name and locations
        turbine_array_dict = read_foam_file(
            os.path.join(self.case_folder, self.turbine_array_sub_path))
        self.turbine_name = turbine_array_dict['turbineType'].replace(
            '"', '')  # TODO Assuming only one type
        self.layout_x, self.layout_y = get_turbine_locations(
            os.path.join(self.case_folder, self.turbine_array_sub_path))

        # Save the number of turbines
        self.num_turbines = len(self.layout_x)

        # if SC input exists, use it for yaw and pitch as it will over-ride
        # if it does not exist, assume the values in turbineArray Properties
        if os.path.exists(os.path.join(self.case_folder, 'SC_INPUT.txt')):
            df_SC = read_sc_input(self.case_folder)
            self.yaw_angles = df_SC.yaw.values
            self.pitch_angles = df_SC.pitch.values
        else:
            print(
                'No SC_INPUT.txt, getting pitch and yaw from turbine array props'
            )
            self.yaw_angles = get_turbine_yaw_angles(
                os.path.join(self.case_folder, self.turbine_array_sub_path))
            self.pitch_angles = get_turbine_pitch_angles(
                os.path.join(self.case_folder, self.turbine_array_sub_path))
            print(self.yaw_angles)
            print(self.pitch_angles)

        # Get the turbine rotor diameter and hub height
        turbine_dict = read_foam_file(
            os.path.join(self.case_folder, self.turbine_sub_path,
                         self.turbine_name))
        self.D = 2 * turbine_dict['TipRad']

        # Use the setup file and control file to determine the precursor wind speed
        # And the time flow averaging begins (settling time)
        setup_dict = read_foam_file(
            os.path.join(self.case_folder, self.setup_sub_path))
        controlDict_dict = read_foam_file(
            os.path.join(self.case_folder, self.controlDict_sub_path))
        start_run_time = controlDict_dict['startTime']
        averaging_start_time = setup_dict['meanStartTime']
        if assumed_settling_time is not None:
            print('Using assumed settling time of %.1f s' %
                  assumed_settling_time)
            self.settling_time = assumed_settling_time
        else:
            self.settling_time = averaging_start_time - start_run_time
        self.precursor_wind_speed = setup_dict['U0Mag']

        # Get the wind direction
        self.precursor_wind_dir = setup_dict['dir']

        # Get the surface roughness
        self.z0 = setup_dict['z0']

        # Read the outputs
        self.turbine_output = read_sowfa_df(
            os.path.join(self.case_folder, self.turbine_output_sub_path))

        # Remove the settling time
        self.turbine_output = self.turbine_output[
            self.turbine_output.time > self.settling_time]

        # Get the sim_time
        self.sim_time_length = self.turbine_output.time.max()

        # Read the flow data
        try:
            self.flow_data = self.read_flow_frame_SOWFA(
                os.path.join(case_folder, flow_data_sub_path))

            # Re-set turbine positions to flow_field origin
            self.layout_x = self.layout_x - self.flow_data.origin.x1
            self.layout_y = self.layout_y - self.flow_data.origin.x2

        except FileNotFoundError:
            print('No flow field found, setting NULL, origin at 0')
            self.flow_data = None  #TODO might need a null flow-field

    def __str__(self):

        print('---------------------')
        print('Case: %s' % self.case_folder)
        print('==Turbine Info==')
        print('Turbine: %s' % self.turbine_name)
        print('Diameter: %dm' % self.D)
        print('Num Turbines = %d' % self.num_turbines)
        print('==Control Settings==')
        print('Yaw Angles, ', self.yaw_angles)
        print('Pitch Angles, ', self.pitch_angles)
        print('==Inflow Info==')
        print('U0Mag: %.2fm/s' % self.precursor_wind_speed)
        print('dir: %.1f' % self.precursor_wind_dir)
        print('==Timing Info==')
        print('Settling time: %.1fs' % self.settling_time)
        print('Simulation time: %.1fs' % self.sim_time_length)
        print('---------------------')
        return ' '

    def get_hor_plane(self, height,
                x_resolution=200, 
                y_resolution=200, 
                x_bounds=None,
                y_bounds=None):
        """
        Get a horizontal cut through plane at a specific height

        Args:
            height (float): height of cut plane, defaults to hub-height
                Defaults to Hub-height.
            x1_resolution (float, optional): output array resolution.
                Defaults to 200.
            x2_resolution (float, optional): output array resolution.
                Defaults to 200.
            x1_bounds (tuple, optional): limits of output array.
                Defaults to None.
            x2_bounds (tuple, optional): limits of output array.
                Defaults to None.

        Returns:
            horplane
        """

        # Get points from flow data
        df =  get_plane_from_flow_data(self.flow_data,normal_vector='z', x3_value=height)


        # Compute and return the cutplane
        return CutPlane(df)


    def get_cross_plane(self, x_loc,
                x_resolution=200, 
                y_resolution=200, 
                x_bounds=None,
                y_bounds=None):
        """
        Get a horizontal cut through plane at a specific height

        Args:
            height (float): height of cut plane, defaults to hub-height
                Defaults to Hub-height.
            x1_resolution (float, optional): output array resolution.
                Defaults to 200.
            x2_resolution (float, optional): output array resolution.
                Defaults to 200.
            x1_bounds (tuple, optional): limits of output array.
                Defaults to None.
            x2_bounds (tuple, optional): limits of output array.
                Defaults to None.

        Returns:
            horplane
        """

        # Get the points of data in a dataframe
        df =  get_plane_from_flow_data(self.flow_data,normal_vector='x', x3_value=x_loc)

        # Compute and return the cutplane
        return CutPlane(df)

    def get_y_plane(self, y_loc,
            x_resolution=200, 
            y_resolution=200, 
            x_bounds=None,
            y_bounds=None):
        """
        Get a horizontal cut through plane at a specific height

        Args:
            height (float): height of cut plane, defaults to hub-height
                Defaults to Hub-height.
            x1_resolution (float, optional): output array resolution.
                Defaults to 200.
            x2_resolution (float, optional): output array resolution.
                Defaults to 200.
            x1_bounds (tuple, optional): limits of output array.
                Defaults to None.
            x2_bounds (tuple, optional): limits of output array.
                Defaults to None.

        Returns:
            horplane
        """

        # Get the points of data in a dataframe
        df =  get_plane_from_flow_data(self.flow_data,normal_vector='y', x3_value=y_loc)

        # Compute and return the cutplane
        return CutPlane(df)

    def get_average_powers(self):
        """
        Return the average power from the simulation per turbine

        Args:


        Returns:
            pow_list (numpy array): an array of powers per turbine
        """


        pow_list = list()
        for t in range(self.num_turbines):
            df_sub = self.turbine_output[self.turbine_output.turbine==t]
            pow_list.append(df_sub.powerGenerator.mean())
        return np.array(pow_list)

    def get_average_thrust(self):
        """
        Return the average thrust from the simulation per turbine

        Args:


        Returns:
            pow_list (numpy array): an array of thrust per turbine
        """


        thrust_list = list()
        for t in range(self.num_turbines):
            df_sub = self.turbine_output[self.turbine_output.turbine==t]
            thrust_list.append(df_sub.thrust.mean())
        return np.array(thrust_list)

    def read_flow_frame_SOWFA(self, filename):
        """
        Read flow array output from SOWFA

        Args:
            filename (str): name of file containing flow data.

        Returns:
            FlowData (pd.DataFrame): a pandas table with the columns,
                of all relavent flow info (e.g. x, y, z, u, v, w).
        """

        # Read the dimension info from the file
        with open(filename, 'r') as f:
            for _ in range(10):
                read_data = f.readline()
                if 'SPACING' in read_data:
                    splitstring = read_data.rstrip().split(' ')
                    spacing = Vec3(float(splitstring[1]),
                                   float(splitstring[2]),
                                   float(splitstring[3]))
                if 'DIMENSIONS' in read_data:
                    splitstring = read_data.rstrip().split(' ')
                    dimensions = Vec3(int(splitstring[1]), int(splitstring[2]),
                                      int(splitstring[3]))
                if 'ORIGIN' in read_data:
                    splitstring = read_data.rstrip().split(' ')
                    origin = Vec3(float(splitstring[1]), float(splitstring[2]),
                                  float(splitstring[3]))

        # Set up x, y, z as lists
        if dimensions.x1 > 1.0:
            xRange = np.arange(0, dimensions.x1 * spacing.x1, spacing.x1)
        else:
            xRange = np.array([0.0])

        if dimensions.x2 > 1.0:
            yRange = np.arange(0, dimensions.x2 * spacing.x2, spacing.x2)
        else:
            yRange = np.array([0.0])

        if dimensions.x3 > 1.0:
            zRange = np.arange(0, dimensions.x3 * spacing.x3, spacing.x3)
        else:
            zRange = np.array([0.0])

        pts = np.array([(x, y, z) for z in zRange for y in yRange
                        for x in xRange])

        df = pd.read_csv(filename,
                         skiprows=10,
                         sep='\t',
                         header=None,
                         names=['u', 'v', 'w'])
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]

        return FlowData(x, y, z, df.u.values, df.v.values, df.w.values,
                        spacing, dimensions, origin)


def read_sc_input(case_folder, wind_direction=270.):
    """
    Read the super controller (SC) input file to get the wind farm
    control settings.

    Args:
        case_folder (str): path to folder containing SC data.
        wind_direction (float, optional): Wind direction.
            Defaults to 270..

    Returns:
        df_SC (pd.DataFrame): dataframe containing SC info.
    """

    sc_file = os.path.join(case_folder, 'SC_INPUT.txt')

    df_SC = pd.read_csv(sc_file, delim_whitespace=True)

    df_SC.columns = ['time', 'turbine', 'yaw', 'pitch']

    df_SC['yaw'] = wind_direction - df_SC.yaw

    df_SC = df_SC.set_index('turbine')

    return df_SC


def read_sowfa_df(folder_name, channels=[]):
    """
    New function to use pandas to read in files using pandas

    Args:
        folder_name (str): where to find the outputs of ALL channels,
            not really used for now, but could be a list of desired
            channels to only read.
        channels (list, optional): list of specific channels to read.
            Defaults to [].
    """

    # Get the availble outputs
    outputNames = [
        f for f in os.listdir(folder_name)
        if os.path.isfile(os.path.join(folder_name, f))
    ]

    # Remove the harder input files for now (undo someday)
    hardFiles = [
        'Vtangential', 'Cl', 'Cd', 'Vradial', 'x', 'y', 'z', 'alpha',
        'axialForce'
    ]
    simpleFiles = [
        'nacYaw', 'rotSpeedFiltered', 'rotSpeed', 'thrust', 'torqueGen',
        'powerRotor', 'powerGenerator', 'torqueRotor', 'azimuth', 'pitch'
    ]

    # Limit to files
    if len(channels) == 0:
        outputNames = [o for o in outputNames if o in simpleFiles]
    else:
        outputNames = channels

    # Get the number of channels
    num_channels = len(outputNames)

    if num_channels == 0:
        raise ValueError('Is %s a data folder?' % folder_name)

    # Now loop through the files
    for c_idx, chan in enumerate(outputNames):

        filename = os.path.join(folder_name, chan)

        # Load the file
        df_inner = pd.read_csv(filename, sep=' ', header=None, skiprows=1)

        # Rename the columns
        df_inner.columns = ['turbine', 'time', 'dt', chan]

        # Drop dt
        df_inner = df_inner[['time', 'turbine',
                             chan]].set_index(['time', 'turbine'])

        # On first run declare the new frame
        if c_idx == 0:
            # Declare the main data frame to return as copy
            df = df_inner.copy(deep=True)

        # On other loops just add the new frame
        else:
            df[chan] = df_inner[chan]

    # Reset the index
    df = df.reset_index()

    # Zero the time
    df['time'] = df.time - df.time.min()

    return df


def read_foam_file(filename):
    """
    Method to read scalar and boolean/string inputs from an OpenFOAM
    input file.

    Args:
        filename (str): path to file to read.

    Returns:
        data (dict): dictionary with OpenFOAM inputs
    """

    data = {}

    with open(filename, 'r') as fid:
        raw = fid.readlines()

    count = 0
    bloc_comment_test = False
    for i, line in enumerate(raw):

        if raw[i][0:2] == '/*':
            bloc_comment_test = True

        if bloc_comment_test is False:

            # Check if the string is a comment and skip line
            if raw[i].strip()[0:2] == '//' or raw[i].strip()[0:1] == '#':
                pass

            elif len(raw[i].strip()
                     ) == 0:  # Check if the string is empty and skip line
                pass

            else:
                tmp = raw[i].strip().rstrip().split()
                try:
                    data[tmp[0].replace('"', '')] = np.float(tmp[1][:-1])
                except:
                    try:
                        data[tmp[0].replace('"', '')] = tmp[1][:-1]
                    except:
                        next

        if raw[i][0:2] == '\*':
            bloc_comment_test = False

    return data


def get_turbine_locations(turbine_array_file):
    """
    Extract wind turbine locations from SOWFA data.

    Args:
        turbine_array_file (str): path to file containing wind plant
            layout data.

    Returns:
        layout_x (np.array): wind plant layout coodinates (east-west).
        layout_y (np.array): wind plant layout coodinates (north-south).
    """

    x = list()
    y = list()

    with open(turbine_array_file, 'r') as f:
        for line in f:
            if 'baseLocation' in line:
                # Extract the coordinates
                data = re.findall(r"[-+]?\d*\.\d+|\d+", line)

                # Append the data
                x.append(float(data[0]))
                y.append(float(data[1]))

    layout_x = np.array(x)
    layout_y = np.array(y)

    return layout_x, layout_y


def get_turbine_pitch_angles(turbine_array_file):
    """
    Extract wind turbine blade pitch information from SOWFA data.

    Args:
        turbine_array_file (str): path to file containing pitch info.

    Returns:
        p (np.array): blade pitch info.
    """
    p = list()

    with open(turbine_array_file, 'r') as f:
        for line in f:
            if 'Pitch' in line:
                # Extract the coordinates
                data = re.findall(r"[-+]?\d*\.\d+|\d+", line)

                # Append the data
                p.append(float(data[0]))

    return np.array(p)


def get_turbine_yaw_angles(turbine_array_file, wind_direction=270.):
    """
    Extract wind turbine yaw angle information from SOWFA data.

    Args:
        turbine_array_file (str): path to file containing yaw info.
        wind_direction (float, optional): Wind direction.
            Defaults to 270..

    Returns:
        y (np.array): wind turbine yaw info.
    """
    y = list()

    with open(turbine_array_file, 'r') as f:
        for line in f:
            if 'NacYaw' in line:
                # Extract the coordinates
                data = re.findall(r"[-+]?\d*\.\d+|\d+", line)

                # Append the data
                y.append(wind_direction - float(data[0]))

    return np.array(y)