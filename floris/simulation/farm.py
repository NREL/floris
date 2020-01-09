# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from ..utilities import Vec3
from .wake_combination import WakeCombination
from .flow_field import FlowField
from .wind_map import WindMap
from .turbine_map import TurbineMap
import copy
import numpy as np


class Farm():
    """
    Farm is a class containing the objects that make up a FLORIS model.

    Farm is the container class of the FLORIS package. It brings 
    together all of the component objects after input (i.e., Turbine, 
    Wake, FlowField) and packages everything into the appropriate data 
    type. Farm should also be used as an entry point to probe objects 
    for generating output.

    Args:
        instance_dictionary: A dictionary as generated from the 
            input_reader; it should have the following key-value pairs:

            -   **description**: A string containing a description of 
                the wind farm.
            -   **properties**: A dictionary containing the following 
                key-value pairs:

                -   **wind_speed**: A list that contains the wind speed 
                    measurements at hub height (m/s).
                -   **wind_x**: a list that contains the x coordinates
                    of the wind speed measurements.
                -   **wind_y**: a list that contains the y coordinates
                    of the wind speed measurements.
                -   **wind_direction**: A list that contains the wind 
                    direction measurements (deg).
                -   **turbulence_intensity**: A list containing turbulence 
                    intensity measurements at hub height (%).
                -   **wind_shear**: A float that is the power law wind 
                    shear exponent.
                -   **wind_veer**: A float that is the vertical change 
                    in wind direction across the rotor.
                -   **air_density**: A float that is the air 
                    density (kg/m^3).
                -   **layout_x**: A list that contains the 
                    x coordinates of the turbines.
                -   **layout_y**: A list that contains the 
                    y coordinates of the turbines.

        turbine: The Turbine object used in Farm.
        wake: The Wake object used in Farm.

    Returns:
        Farm: An instantiated Farm object.
    """

    def __init__(self, instance_dictionary, turbine, wake):
        self.description = instance_dictionary["description"]
        properties = instance_dictionary["properties"]
        layout_x = properties["layout_x"]
        layout_y = properties["layout_y"]
        wind_x = properties["wind_x"]
        wind_y = properties["wind_y"]
        self.wake = wake

        self.wind_map = WindMap(
            wind_speed=properties["wind_speed"],
            layout_array=(layout_x, layout_y),
            wind_layout=(wind_x, wind_y),
            turbulence_intensity=properties["turbulence_intensity"],
            wind_direction=properties["wind_direction"]
        )

        self.flow_field = FlowField(
            wind_shear=properties["wind_shear"],
            wind_veer=properties["wind_veer"],
            air_density=properties["air_density"],
            turbine_map=TurbineMap(
                layout_x,
                layout_y,
                [copy.deepcopy(turbine) for ii in range(len(layout_x))]),
            wake=wake,
            wind_map=self.wind_map
        )

    def __str__(self):
        return \
            "Description: {}\n".format(self.description) + \
            "Wake Model: {}\n".format(self.flow_field.wake.velocity_model) + \
            "Deflection Model: {}\n".format(
                self.flow_field.wake.deflection_model)

    def set_wake_model(self, wake_model):
        """
        This method sets the wake model used.

        Args:
            wake_model: A string containing the wake model used to 
                calculate the wake; Valid wake model options are: 
                "curl", "gauss_curl_hybrid", "gauss", "jensen",
                and "multizone".

        Returns:
            *None* -- The wake model and flow field are updated in 
            the :py:obj:`floris.simulation.flow_field` object.

        Examples:
            To set the wake model:

            >>> floris.farm.set_wake_model('curl')
        """

        valid_wake_models = [
            'curl', 'gauss_curl_hybrid', 'gauss', 'jensen', 'multizone'
        ]
        if wake_model not in valid_wake_models:
            raise Exception(
                "Invalid wake model. Valid options include: {}.".format(", ".join(valid_wake_models))
            )

        self.flow_field.wake.velocity_model = wake_model
        if wake_model == 'jensen' or wake_model == 'multizone':
            self.flow_field.wake.deflection_model = 'jimenez'
        else:
            self.flow_field.wake.deflection_model = wake_model

        self.flow_field.reinitialize_flow_field(
            with_resolution=self.flow_field.wake.velocity_model.model_grid_resolution)

    def set_yaw_angles(self, yaw_angles):
        """
        This method sets yaw angles for all turbines and optionally 
        calculates the new wake velocities and updates them in the 
        flow field.

        Args:
            yaw_angles: A single float that sets a constant yaw angle 
                for all turbines or a list of floats that are unique 
                yaw angles for each turbine in degrees.

        Returns:
            *None* -- The turbines are updated directly and the flow 
            field is updated in the 
            :py:obj:`floris.simulation.flow_field` object.

        Examples:
            To set all the yaw angles to one value:

            >>> floris.farm.set_yaw_angles(20.0)

            To set unique yaw angles for the turbines (for example, 
            a 3 turbine array):

            >>> floris.farm.set_yaw_angles([20.0, 10.0, 0.0])
        """
        if isinstance(yaw_angles, float) or isinstance(yaw_angles, int):
            yaw_angles = [yaw_angles] * len(self.turbines)

        for yaw_angle, turbine in zip(yaw_angles, self.turbines):
            turbine.yaw_angle = yaw_angle

    # Getters & Setters

    @property
    def wind_speed(self):
        """
        This property returns the wind speed for the wind farm.

        Returns:
            list: The current wind speed at each turbine  in m/s.

        Examples:
            To get the wind speed for the wind farm:

            >>> wind_speed = floris.farm.wind_speed()
        """
        return self.wind_map.turbine_wind_speed

    @property
    def wind_direction(self):
        """
        This property returns the wind direction for the wind farm.

        Returns:
            list: The current wind direction at each turbine in 
            degrees.

        Examples:
            To get the wind direction for the wind farm:

            >>> wind_direction = floris.farm.wind_direction()
        """
        return self.wind_map.turbine_wind_direction

    @property
    def wind_shear(self):
        """
        This property returns the wind shear power law exponent for 
        the wind farm.

        Returns:
            float: The current wind shear power law exponent in the 
            wind farm.

        Examples:
            To get the wind shear for the wind farm:

            >>> wind_shear = floris.farm.wind_shear()
        """
        return self.flow_field.wind_shear

    @property
    def wind_veer(self):
        """
        This property returns the wind veer -- the vertical change in 
        wind direction across the rotor.

        Returns:
            float: The current vertical change in wind direction 
            across the rotor in degrees.

        Examples:
            To get the wind veer for the wind farm:

            >>> wind_veer = floris.farm.wind_veer()
        """
        return self.flow_field.wind_veer

    @property
    def turbulence_intensity(self):
        """
        This property returns the turbulence intensity for the 
        wind farm.

        Returns:
            list: The initial turbulence intensity at each turbine 
            expressed as a decimal fraction.

        Examples:
            To get the turbulence intensity for the wind farm:

            >>> TI = floris.farm.turbulence_intensity()
        """
        return self.wind_map.turbine_turbulence_intensity

    @property
    def air_density(self):
        """
        This property returns the air density for the wind farm.

        Returns:
            float: The current air density in kg/m^3.

        Examples:
            To get the air density for the wind farm:

            >>> air_density = floris.farm.air_density()
        """
        return self.flow_field.air_density

    @property
    def wind_map(self):
        """
        This property returns the values of the 
        :py:obj:`floris.simulation.wind_map` object associated with 
        the wind farm.

        Returns:
            WindMap: A :py:obj:`floris.simulation.wind_map` 
            object that holds atmospheric input.

        Examples:
            To get the wind map for the wind farm:

            >>> wind_map = floris.farm.wind_map()
        """

        return self._wind_map

    @wind_map.setter
    def wind_map(self, value):
        self._wind_map = value
    
    @property
    def turbine_map(self):
        """
        This property returns the turbine map of the 
        :py:obj:`floris.simulation.flow_field` object associated with 
        the wind farm.

        Returns:
            TurbineMap: A :py:obj:`floris.simulation.turbine_map` 
            object that holds turbine information for the farm.

        Examples:
            To get the turbine map for the wind farm:

            >>> turbine_map = floris.farm.turbine_map()
        """
        return self.flow_field.turbine_map

    @property
    def turbines(self):
        """
        This property returns the list of 
        :py:obj:`floris.simulation.turbine` objects contained in the 
        :py:obj:`floris.simulation.turbine_map` object.

        Returns:
            [Turbine]: A list of :py:obj:`floris.simulation.turbine` 
            objects that hold the turbine information for the wind farm.

        Examples:
            To get a list of turbine objects from the wind farm:

            >>> turbines = floris.farm.turbines()
        """
        return self.turbine_map.turbines
