# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import numpy as np

from .base_velocity_deficit import VelocityDeficit


class Jensen(VelocityDeficit):
    """
    The Jensen model computes the wake velocity deficit based on the classic
    Jensen/Park model :cite:`jvm-jensen1983note`.

    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: jvm-
    """

    default_parameters = {"we": 0.05}

    def __init__(self, parameter_dictionary):
        """
        Stores model parameters for use by methods.

        Args:
            parameter_dictionary (dict): Model-specific parameters.
                Default values are used when a parameter is not included
                in `parameter_dictionary`. Possible key-value pairs include:

                -   **we** (*float*): The linear wake decay constant that
                    defines the cone boundary for the wake as well as the
                    velocity deficit. D/2 +/- we*x is the cone boundary for the
                    wake.
        """
        super().__init__(parameter_dictionary)
        self.model_string = "jensen"
        model_dictionary = self._get_model_dict(__class__.default_parameters)
        self.we = float(model_dictionary["we"])

    def function(
        self,
        x_locations,
        y_locations,
        z_locations,
        turbine,
        turbine_coord,
        deflection_field,
        flow_field,
    ):
        """
        Using the Jensen wake model, this method calculates and returns
        the wake velocity deficits, caused by the specified turbine,
        relative to the freestream velocities at the grid of points
        comprising the wind farm flow field.

        Args:
            x_locations (np.array): An array of floats that contains the
                streamwise direction grid coordinates of the flow field
                domain (m).
            y_locations (np.array): An array of floats that contains the grid
                coordinates of the flow field domain in the direction normal to
                x and parallel to the ground (m).
            z_locations (np.array): An array of floats that contains the grid
                coordinates of the flow field domain in the vertical
                direction (m).
            turbine (:py:obj:`floris.simulation.turbine`): Object that
                represents the turbine creating the wake.
            turbine_coord (:py:obj:`floris.utilities.Vec3`): Object containing
                the coordinate of the turbine creating the wake (m).
            deflection_field (np.array): An array of floats that contains the
                amount of wake deflection in meters in the y direction at each
                grid point of the flow field.
            flow_field (:py:class:`floris.simulation.flow_field`): Object
                containing the flow field information for the wind farm.

        Returns:
            np.array, np.array, np.array:
                Three arrays of floats that contain the wake velocity
                deficit in m/s created by the turbine relative to the freestream
                velocities for the U, V, and W components, aligned with the
                x, y, and z directions, respectively. The three arrays contain
                the velocity deficits at each grid point in the flow field.
        """

        # define the boundary of the wake model ... y = mx + b
        m = self.we
        x = x_locations - turbine_coord.x1
        b = turbine.rotor_radius

        boundary_line = m * x + b

        y_upper = boundary_line + turbine_coord.x2 + deflection_field
        y_lower = -1 * boundary_line + turbine_coord.x2 + deflection_field

        z_upper = boundary_line + turbine.hub_height
        z_lower = -1 * boundary_line + turbine.hub_height

        # calculate the wake velocity
        c = (
            turbine.rotor_diameter
            / (2 * self.we * (x_locations - turbine_coord.x1) + turbine.rotor_diameter)
        ) ** 2

        # filter points upstream and beyond the upper and
        # lower bounds of the wake
        c[x_locations - turbine_coord.x1 < 0] = 0
        c[y_locations > y_upper] = 0
        c[y_locations < y_lower] = 0
        c[z_locations > z_upper] = 0
        c[z_locations < z_lower] = 0

        return (
            2 * turbine.aI * c * flow_field.u_initial,
            np.zeros(np.shape(flow_field.u_initial)),
            np.zeros(np.shape(flow_field.u_initial)),
        )

    @property
    def we(self):
        """
        The linear wake decay constant that defines the cone boundary for the
        wake as well as the velocity deficit. D/2 +/- we*x is the cone boundary
        for the wake.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._we

    @we.setter
    def we(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for we: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._we = value
        if value != __class__.default_parameters["we"]:
            self.logger.info(
                (
                    "Current value of we, {0}, is not equal to tuned " + "value of {1}."
                ).format(value, __class__.default_parameters["we"])
            )
