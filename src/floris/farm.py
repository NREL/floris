"""
Copyright 2017 NREL

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

from .types import Vec3
from .wake_combination import WakeCombination
from .flow_field import FlowField
from .turbine_map import TurbineMap
import copy
import numpy as np


class Farm():
    """
    Farm is the container class of the FLORIS package. It brings together all
    of the component objects after input (ie Turbine, Wake, FlowField) and
    packages everything into the appropriate data type. Farm should also be used
    as an entry point to probe objects for generating output.

    inputs:
        instance_dictionary: dict - the input dictionary as generated from the input_reader;
            it should have the following key-value pairs:
                {
                    "wind_speed": float,

                    "wind_direction": float,

                    "turbulence_intensity": float,

                    "wind_shear": float,

                    "wind_veer": float,

                    "air_density": float,

                    "wake_combination": str,

                    "layout_x": list,

                    "layout_y": list

                }

        turbine: Turbine - Turbine instance used in Farm

        wake: Wake - Wake instance used in Farm

    outputs:
        self: Farm - an instantiated Farm object
    """

    def __init__(self, instance_dictionary, turbine, wake):
        self.description = instance_dictionary["description"]
        properties = instance_dictionary["properties"]
        self.wind_speed = properties["wind_speed"]
        self.wind_direction = properties["wind_direction"]
        self.turbulence_intensity = properties["turbulence_intensity"]
        self.wind_shear = properties["wind_shear"]
        self.wind_veer = properties["wind_veer"]
        self.air_density = properties["air_density"]
        self.layout_x = properties["layout_x"]
        self.layout_y = properties["layout_y"]
        self.wake = wake

        turbine_dict = {}
        for c in list(zip(self.layout_x, self.layout_y)):
            turbine_dict[Vec3(c[0], c[1], turbine.hub_height)] = copy.deepcopy(turbine)
        self.turbine_map = TurbineMap(turbine_dict)

        self.set_wind_direction(self.wind_direction, calculate_wake=False)
        self.flow_field = FlowField(wind_speed=self.wind_speed,
                                    wind_direction=self.wind_direction,
                                    wind_shear=self.wind_shear,
                                    wind_veer=self.wind_veer,
                                    turbulence_intensity=self.turbulence_intensity,
                                    air_density=self.air_density,
                                    turbine_map=self.turbine_map,
                                    wake=wake)

    def set_wake_model(self, wake_model, calculate_wake=False):
        """
        Sets the flow model

        inputs:
        wake_model: string - the wake model used to calculate the wake;
            valid wake model options are:
                {
                    "curl",

                    "gauss",

                    "jensen",

                    "floris"
                }
        calculate_wake: boolean - option to calculate the wake after the wake model is set
        """

        valid_wake_models = ['curl', 'gauss', 'jensen', 'floris']
        if wake_model not in valid_wake_models:
            raise Exception("Invalid wake model. Valid options include: {}.".format(", ".join(valid_wake_models)))

        if wake_model == 'gauss':
            self.flow_field.wake.velocity_model = 'gauss'
            self.flow_field.wake.deflection_model = 'gauss_deflection'
        elif wake_model == 'curl':
            self.flow_field.wake.velocity_model = 'curl'
            self.flow_field.wake.deflection_model = 'curl'
        elif wake_model == 'jensen':
            self.flow_field.wake.velocity_model = 'jensen'
            self.flow_field.wake.deflection_model = 'jimenez'
        elif wake_model == 'floris':
            self.flow_field.wake.velocity_model = 'floris'
            self.flow_field.wake.deflection_model = 'floris'

        self.flow_field.reinitialize_flow_field()

        if calculate_wake:
            self.flow_field.calculate_wake()

    def set_description(self, value, calculate_wake=False):
        """
        Sets the farm desciption
        """
        self._set_flow_property("description",
                                value,
                                calculate_wake=calculate_wake) 

    def _set_flow_property(self, property_name, value, calculate_wake=True):
        """
        Sets a flow property, then recreates the flow field and calculates wake
        """
        self.__setattr__(property_name, value)
        self._create_flow_field()
        if calculate_wake:
            self.flow_field.calculate_wake()

    def _create_flow_field(self):
        """
        Creates the flow field with respective attributes
        """
        self.flow_field = FlowField(wind_speed=self.wind_speed,
                                    wind_direction=self.wind_direction,
                                    wind_shear=self.wind_shear,
                                    wind_veer=self.wind_veer,
                                    turbulence_intensity=self.turbulence_intensity,
                                    air_density=self.air_density,
                                    turbine_map=self.turbine_map,
                                    wake=self.wake)

    def set_wind_speed(self, value, calculate_wake=True):
        """
        Sets wind speed
        """
        self._set_flow_property(
            "wind_speed",
            value,
            calculate_wake=calculate_wake
        )

    def set_wind_direction(self, value, calculate_wake=True):
        """
        Sets wind direction (in degrees)
        """
        value = np.radians(value - 270)
        self._set_flow_property(
            "wind_direction",                    
            value,
            calculate_wake=calculate_wake
        )

    def set_wind_shear(self, value, calculate_wake=True):
        """
        Sets wind shear
        """
        self._set_flow_property(
            "wind_shear",
            value,
            calculate_wake=calculate_wake
        )

    def set_wind_veer(self, value, calculate_wake=True):
        """
        Sets wind shear
        """
        self._set_flow_property(
            "wind_veer",
            value,
            calculate_wake=calculate_wake
        )

    def set_turbulence_intensity(self, value, calculate_wake=True):
        """
        Sets turbulence intensity
        """
        self._set_flow_property(
            "turbulence_intensity",
            value,
            calculate_wake=calculate_wake
        )

    def set_air_density(self, value, calculate_wake=True):
        """
        Sets air density
        """
        self._set_flow_property(
            "air_density",
            value,
            calculate_wake=calculate_wake
        )

    def set_yaw_angles(self, yaw_angles, calculate_wake=True):
        """
        Sets yaw angles for all turbines and calculates the wake

        inputs:
            yaw_angles:
                float - constant yaw angle for all turbines in degrees
                [float] - unique yaw angle for each turbine in degrees

        outputs:
            none
        """
        if isinstance(yaw_angles, float) or isinstance(yaw_angles, int):
            yaw_angles = [yaw_angles] * len(self.turbines)

        for yaw_angle, turbine in zip(yaw_angles, self.turbines):
            turbine.yaw_angle = yaw_angle

        if calculate_wake:
            self.flow_field.calculate_wake()

    def set_turbine_locations(self, layout_x, layout_y, calculate_wake=True): 
        """
        Sets the locations for all turbines and calculates the wake

        inputs:
            layout_x:
                float - x coordinate(s) for the turbine(s)
            
            layout_y:
                float - y coordinate(s) for the turbine(s)

        outputs:
            none

        """
        # assign coordinates to turbines      
        self.layout_x = layout_x
        self.layout_y = layout_y

        turbine_dict = {}
        turbine = self.turbines[0]
        for c in list(zip(self.layout_x, self.layout_y)):
            turbine_dict[Vec3(c[0], c[1], turbine.hub_height)] = copy.deepcopy(turbine)

        self.turbine_map = TurbineMap(turbine_dict)

        #update relevant flow_field values
        self.flow_field.turbine_map = self.turbine_map
        self.flow_field.xmin, self.flow_field.xmax, self.flow_field.ymin, self.flow_field.ymax, self.flow_field.zmin, self.flow_field.zmax = self.flow_field._set_domain_bounds()
        self.flow_field.x, self.flow_field.y, self.flow_field.z = self.flow_field._discretize_freestream_domain() 

        if calculate_wake:
            self.flow_field.calculate_wake()

    @property
    def turbines(self):
        """
        Returns a list of turbine objects
        """
        return [turbine for _, turbine in self.flow_field.turbine_map.items()]
