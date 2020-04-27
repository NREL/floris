# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from ..utilities import Vec3
from ..utilities import setup_logger
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
    """

    def __init__(self, instance_dictionary, turbine, wake):
        """
        The initialization method unpacks some of the data from the input
        dictionary in order to create a couple of unerlying data structures:

            - :py:obj:`~.wind_map.WindMap`
            - :py:obj:`~.turbine_map.TurbineMap`

        Args:
            instance_dictionary (dict): The required keys in this dictionary
                are:

                    -   **wind_speed** (*list*): The wind speed measurements at
                        hub height (m/s).
                    -   **wind_x** (*list*): The x-coordinates of the wind
                        speed measurements.
                    -   **wind_y** (*list*): The y-coordinates of the wind
                        speed measurements.
                    -   **wind_direction** (*list*): The wind direction
                        measurements (deg).
                    -   **turbulence_intensity** (*list*): Turbulence intensity
                        measurements at hub height (%).
                    -   **wind_shear** (*float*): The power law wind shear
                        exponent.
                    -   **wind_veer** (*float*): The vertical change in wind
                        direction across the rotor.
                    -   **air_density** (*float*): The air density (kg/m^3).
                    -   **layout_x** (*list*): The x-coordinates of the
                        turbines.
                    -   **layout_y** (*list*): The y-coordinates of the
                        turbines.

            turbine (:py:obj:`~.turbine.Turbine`): The turbine models used
                throughout the farm.
            wake (:py:obj:`~.wake.Wake`): The wake model used to simulate the
                freestream flow and wakes.
        """
        self.name = instance_dictionary["name"]
        properties = instance_dictionary["properties"]
        layout_x = properties["layout_x"]
        layout_y = properties["layout_y"]
        wind_x = properties["wind_x"]
        wind_y = properties["wind_y"]

        self.wind_map = WindMap(
            wind_speed=properties["wind_speed"],
            layout_array=(layout_x, layout_y),
            wind_layout=(wind_x, wind_y),
            turbulence_intensity=properties["turbulence_intensity"],
            wind_direction=properties["wind_direction"])

        self.flow_field = FlowField(
            wind_shear=properties["wind_shear"],
            wind_veer=properties["wind_veer"],
            air_density=properties["air_density"],
            turbine_map=TurbineMap(
                layout_x, layout_y,
                [copy.deepcopy(turbine) for ii in range(len(layout_x))]),
            wake=wake,
            wind_map=self.wind_map,
            specified_wind_height=properties["specified_wind_height"])

    def __str__(self):
        return \
            "Name: {}\n".format(self.name) + \
            "Wake Model: {}\n".format(self.flow_field.wake.velocity_model) + \
            "Deflection Model: {}\n".format(
                self.flow_field.wake.deflection_model)

    def set_wake_model(self, wake_model):
        """
        Sets the velocity deficit model to use as given, and determines the
        wake deflection model based on the selected velocity deficit model.

        Args:
            wake_model (str): The desired wake model.

        Raises:
            Exception: Invalid wake model.
        """
        valid_wake_models = [
             'jensen', 'multizone', 'gauss', 'gauss_legacy',
             'blondel', 'ishihara_qian', 'curl'
        ]
        if wake_model not in valid_wake_models:
            # TODO: logging
            raise Exception(
                "Invalid wake model. Valid options include: {}.".format(
                    ", ".join(valid_wake_models)))

        self.flow_field.wake.velocity_model = wake_model
        if wake_model == 'jensen' or wake_model == 'multizone':
            self.flow_field.wake.deflection_model = 'jimenez'
        elif wake_model == 'blondel' or wake_model == 'ishihara_qian' \
            or 'gauss' in wake_model:
                self.flow_field.wake.deflection_model = 'gauss'
        else:
            self.flow_field.wake.deflection_model = wake_model

        self.flow_field.reinitialize_flow_field(
            with_resolution=self.flow_field.wake.velocity_model.
            model_grid_resolution)

    def set_yaw_angles(self, yaw_angles):
        """
        Sets the yaw angles for all turbines on the
        :py:obj:`~.turbine.Turbine` objects directly.

        Args:
            yaw_angles (float or list( float )): A single value to set
                all turbine yaw angles or a list of yaw angles corresponding
                to individual turbine yaw angles. Yaw angles are expected
                in degrees.
        """
        if isinstance(yaw_angles, float) or isinstance(yaw_angles, int):
            yaw_angles = [yaw_angles] * len(self.turbines)

        for yaw_angle, turbine in zip(yaw_angles, self.turbines):
            turbine.yaw_angle = yaw_angle

    # Getters & Setters

    @property
    def wind_speed(self):
        """
        Wind speed at each wind turbine.

        Returns:
            list(float)
        """
        return self.wind_map.turbine_wind_speed

    @property
    def wind_direction(self):
        """
        Wind direction at each wind turbine.
        # TODO: Explain the wind direction change here.
        #       - Is there a transformation on wind map?
        #       - Is this always from a particular direction?

        Returns:
            list(float)
        """
        return list(
            (np.array(self.wind_map.turbine_wind_direction) - 90) % 360)

    @property
    def wind_shear(self):
        """
        Wind shear power law exponent for the flow field.

        Returns:
            float
        """
        return self.flow_field.wind_shear

    @property
    def wind_veer(self):
        """
        Wind veer (vertical change in wind direction) for the flow field.

        Returns:
            float
        """
        return self.flow_field.wind_veer

    @property
    def turbulence_intensity(self):
        """
        Initial turbulence intensity at each turbine expressed as a
        decimal fraction.

        Returns:
            list(float)
        """
        return self.wind_map.turbine_turbulence_intensity

    @property
    def air_density(self):
        """
        Air density for the wind farm in kg/m^3.

        Returns:
            float
        """
        return self.flow_field.air_density

    @property
    def wind_map(self):
        """
        WindMap object attached to the Farm.

        Args:
            value (:py:obj:`~.wind_map.WindMap`): WindMap object to be set.

        Returns:
            :py:obj:`~.wind_map.WindMap`
        """
        # TODO: Does this need to be a virtual propert?
        return self._wind_map

    @wind_map.setter
    def wind_map(self, value):
        self._wind_map = value

    @property
    def turbine_map(self):
        """
        TurbineMap attached to the Farm's :py:obj:`~.flow_field.FlowField`
        object. This is used to reduce the depth of the object-hierachy
        required to modify the wake models from a script.

        Returns:
            :py:obj:`~.turbine_map.TurbineMap`
        """
        return self.flow_field.turbine_map

    @property
    def turbines(self):
        """
        All turbines included in the model.

        Returns:
            list(:py:obj:`~.turbine.Turbine`) 
        """
        return self.turbine_map.turbines

    @property
    def wake(self):
        """
        The Farm's Wake object. This is used to reduce the depth of the
        object-hierachy required to modify the wake models from a script.

        Returns:
            :py:obj:`~.wake.Wake`.
        """
        return self.flow_field.wake
