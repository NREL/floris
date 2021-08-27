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


class TurbOPark(VelocityDeficit):
    """
    An implementation of the TurbOPark model by Nicolai Nygaard
    :cite:`jvm-nygaard2020modelling`.
    Default tuning calibrations taken from same paper.

    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: jvm-
    """

    default_parameters = {"A": 0.6, "c1": 1.5, "c2": 0.8}

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
        self.model_string = "turbopark"
        model_dictionary = self._get_model_dict(__class__.default_parameters)
        self.A = float(model_dictionary["A"])
        self.c1 = float(model_dictionary["c1"])
        self.c2 = float(model_dictionary["c2"])

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
        Using the TubrOPark wake model, this method calculates and returns
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

        # Get model parameters
        A = self.A
        c1 = self.c1
        c2 = self.c2

        # Initial flowfield used to calculate velocity deficit
        U0 = flow_field.u_initial

        # Get turbulence intensity for current turbine
        I0 = turbine.current_turbulence_intensity

        # Parameters from turbine
        D = turbine.rotor_diameter
        Ct = turbine.Ct
        V_in = turbine.average_velocity

        # Computed values
        alpha = c1 * I0  # (Page 4)
        beta = c2 * I0 / np.sqrt(Ct)

        # get the x term
        x = x_locations - turbine_coord.x1

        # Solve for the wake diameter
        # (Equation 6 (in steps))
        term1 = np.sqrt((alpha + (beta * x / D)) ** 2 + 1)
        term2 = np.sqrt(1 + alpha ** 2)
        term3 = (term1 + 1) * alpha
        term4 = (term2 + 1) * (alpha + (beta * x / D))
        Dwx = D + ((A * I0 * D) / beta) * (term1 - term2 - np.log(term3 / term4))

        # Solve for the velocity deficit
        delta = (1 - (V_in / U0) * np.sqrt(1 - Ct)) * (D / Dwx) ** 2

        # Solve for velocity deficit c
        c = delta

        # Define these bounds as in jensen
        boundary_line = Dwx / 2.0
        y_upper = boundary_line + turbine_coord.x2 + deflection_field
        y_lower = -1 * boundary_line + turbine_coord.x2 + deflection_field
        z_upper = boundary_line + turbine.hub_height
        z_lower = -1 * boundary_line + turbine.hub_height

        # filter points upstream and beyond the upper and
        # lower bounds of the wake
        c[x_locations - turbine_coord.x1 < 0] = 0
        c[y_locations > y_upper] = 0
        c[y_locations < y_lower] = 0
        c[z_locations > z_upper] = 0
        c[z_locations < z_lower] = 0

        return (
            c * flow_field.u_initial,
            np.zeros(np.shape(flow_field.u_initial)),
            np.zeros(np.shape(flow_field.u_initial)),
        )

    @property
    def A(self):
        """
        Model calibration constant A used in determining the wake expansion rate.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._A

    @A.setter
    def A(self, value):
        if type(value) is not float and type(value) is not int:
            err_msg = (
                "Invalid value type given for A: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._A = value
        if value != __class__.default_parameters["A"]:
            self.logger.info(
                (
                    "Current value of A, {0}, is not equal to tuned " + "value of {1}."
                ).format(value, __class__.default_parameters["A"])
            )

    @property
    def c1(self):
        """
        Calibration constant for the wake added turbelence.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._c1

    @c1.setter
    def c1(self, value):
        if type(value) is not float and type(value) is not int:
            err_msg = (
                "Invalid value type given for c1: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._c1 = value
        if value != __class__.default_parameters["c1"]:
            self.logger.info(
                (
                    "Current value of c1, {0}, is not equal to tuned " + "value of {1}."
                ).format(value, __class__.default_parameters["c1"])
            )

    @property
    def c2(self):
        """
        Calibration constant for the wake added turbelence.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._c2

    @c2.setter
    def c2(self, value):
        if type(value) is not float and type(value) is not int:
            err_msg = (
                "Invalid value type given for c2: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._c2 = value
        if value != __class__.default_parameters["c2"]:
            self.logger.info(
                (
                    "Current value of c2, {0}, is not equal to tuned " + "value of {1}."
                ).format(value, __class__.default_parameters["c2"])
            )
