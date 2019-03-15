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
        self.layout_x = properties["layout_x"]
        self.layout_y = properties["layout_y"]
        self.wake = wake

        turbine_dict = {}
        for c in list(zip(self.layout_x, self.layout_y)):
            turbine_dict[Vec3(c[0], c[1], turbine.hub_height)] = copy.deepcopy(turbine)

        self.flow_field = FlowField(
            wind_speed=properties["wind_speed"],
            wind_direction=properties["wind_direction"],
            wind_shear=properties["wind_shear"],
            wind_veer=properties["wind_veer"],
            turbulence_intensity=properties["turbulence_intensity"],
            air_density=properties["air_density"],
            turbine_map=TurbineMap(turbine_dict),
            wake=wake
        )

    def __str__(self):
        return \
            "Description: {}\n".format(self.description) + \
            "Wake Model: {}\n".format(self.flow_field.wake.velocity_model) + \
            "Deflection Model: {}\n".format(self.flow_field.wake.deflection_model)

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

        if wake_model == 'jensen':
            self.flow_field.wake.velocity_model = 'jensen'
            self.flow_field.wake.deflection_model = 'jimenez'
        elif wake_model == 'floris':
            self.flow_field.wake.velocity_model = 'floris'
            self.flow_field.wake.deflection_model = 'floris'
        elif wake_model == 'gauss':
            self.flow_field.wake.velocity_model = 'gauss'
            self.flow_field.wake.deflection_model = 'gauss_deflection'
        elif wake_model == 'curl':
            self.flow_field.wake.velocity_model = 'curl'
            self.flow_field.wake.deflection_model = 'curl'

        self.flow_field.reinitialize_flow_field(with_resolution=self.flow_field.wake.velocity_model.model_grid_resolution)

        if calculate_wake:
            self.flow_field.calculate_wake()

    def _update_flow_field(self, calculate_wake=True):
        """
        Sets a flow property, then recreates the flow field and calculates wake
        """
        self.flow_field.reinitialize_flow_field(
            wind_speed=self.wind_speed,
            wind_direction=self.wind_direction,
            wind_shear=self.wind_shear,
            wind_veer=self.wind_veer,
            turbulence_intensity=self.turbulence_intensity,
            air_density=self.air_density,
            wake=self.wake,
            turbine_map=self.turbine_map,
            with_resolution=self.wake.velocity_model.model_grid_resolution
        )
        if calculate_wake:
            self.flow_field.calculate_wake()

    def set_yaw_angles(self, yaw_angles, calculate_wake=False):
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

    def set_turbine_locations(self, layout_x, layout_y, calculate_wake=False):
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
        # TODO: this function requires all turbines to be the same.
        # Is it reasonable to require the user to give a TurbineMap-like object
        # where the turbines are instantiated and located externally?

        # assign coordinates to turbines      
        self.layout_x = layout_x
        self.layout_y = layout_y

        turbine_dict = {}
        turbine = self.turbines[0]
        for c in list(zip(self.layout_x, self.layout_y)):
            turbine_dict[Vec3(c[0], c[1], turbine.hub_height)] = copy.deepcopy(turbine)

        #update relevant farm and flow_field values
        self.flow_field.reinitialize_flow_field(
            turbine_map=TurbineMap(turbine_dict),
            with_resolution=self.wake.velocity_model.model_grid_resolution
        )

        if calculate_wake:
            self.flow_field.calculate_wake()

    # Getters & Setters
    @property
    def wind_speed(self):
        return self.flow_field.wind_speed

    @wind_speed.setter
    def wind_speed(self, value):
        self.flow_field.wind_speed = value

    @property
    def wind_direction(self):
        return self.flow_field.wind_direction

    @wind_direction.setter
    def wind_direction(self, value):
        self.flow_field.wind_direction = value

    @property
    def wind_shear(self):
        return self.flow_field.wind_shear

    @wind_shear.setter
    def wind_shear(self, value):
        self.flow_field.wind_shear = value

    @property
    def wind_veer(self):
        return self.flow_field.wind_veer

    @wind_veer.setter
    def wind_veer(self, value):
        self.flow_field.wind_veer = value

    @property
    def turbulence_intensity(self):
        return self.flow_field.turbulence_intensity

    @turbulence_intensity.setter
    def turbulence_intensity(self, value):
        self.flow_field.turbulence_intensity = value

    @property
    def air_density(self):
        return self.flow_field.air_density

    @air_density.setter
    def air_density(self, value):
        self.flow_field.air_density = value

    @property
    def turbine_map(self):
        return self.flow_field.turbine_map
    
    @turbine_map.setter
    def turbine_map(self, value):
        self.flow_field.turbine_map = value

    @property
    def turbines(self):
        """
        Returns a list of turbine objects gotten from the TurbineMap
        """
        return self.turbine_map.turbines
