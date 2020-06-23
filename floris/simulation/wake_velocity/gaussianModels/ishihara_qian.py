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

from ....utilities import cosd, sind, tand
from .gaussian_model_base import GaussianModel
from ..base_velocity_deficit import VelocityDeficit


class IshiharaQian(GaussianModel):
    """
    Ishihara is a wake velocity subclass that contains objects related to the
    Gaussian wake model that include a near-wake correction.

    Ishihara is a subclass of
    :py:class:`floris.simulation.wake_velocity.VelocityDeficit` that is
    used to compute the wake velocity deficit based on the Gaussian
    wake model with self-similarity and a near wake correction. The Ishihara
    wake model includes a Gaussian wake velocity deficit profile in the y and z
    directions and includes the effects of ambient turbulence, added turbulence
    from upstream wakes, as well as wind shear and wind veer. For more info,
    see :cite:`iqv-qian2018new`.

    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: iqv-
    """

    default_parameters = {
        "kstar": {"const": 0.11, "Ct": 1.07, "TI": 0.2},
        "epsilon": {"const": 0.23, "Ct": -0.25, "TI": 0.17},
        "a": {"const": 0.93, "Ct": -0.75, "TI": 0.17},
        "b": {"const": 0.42, "Ct": 0.6, "TI": 0.2},
        "c": {"const": 0.15, "Ct": -0.25, "TI": -0.7},
        "calculate_VW_velocities": False,
        "use_yaw_added_recovery": False,
        "eps_gain": 0.2,
    }

    def __init__(self, parameter_dictionary):
        """
        Stores model parameters for use by methods.

        All model parameters combine a constant coefficient, the thrust
        coefficient of the turbine, and the local turbulence intensity.
        Paremeter values are calculated with
        :py:meth:`~.IshiharaQian.parameter_value_from_dict` as:

        Args:
            parameter_dictionary (dict): Model-specific parameters.
                Default values are used when a parameter is not included
                in `parameter_dictionary`. Possible key-value pairs include:

                -   **kstar** (*dict*): A dict that is a parameter used to
                    determine the linear relationship between the
                    turbulence intensity and the width of the Gaussian
                    wake shape.
                -   **epsilon** (*dict*): A dict that is a second parameter
                    used to determine the linear relationship between the
                    turbulence intensity and the width of the Gaussian
                    wake shape.
                -   **a** (*dict*): A dict that is a constant coefficient used
                    in calculation of wake-added turbulence.
                -   **b** (*dict*): A dict that is linear coefficient used in
                    calculation of wake-added turbulence.
                -   **c** (*dict*): A dict that is near-wake coefficient used
                    in calculation of wake-added turbulence.

        """
        super().__init__(parameter_dictionary)
        self.model_string = "ishihara_qian"
        model_dictionary = self._get_model_dict(__class__.default_parameters)

        # wake model parameter
        self.kstar = model_dictionary["kstar"]
        self.epsilon = model_dictionary["epsilon"]
        self.a = model_dictionary["a"]
        self.b = model_dictionary["b"]
        self.c = model_dictionary["c"]

        # GCH Parameters
        self.calculate_VW_velocities = model_dictionary["calculate_VW_velocities"]
        self.use_yaw_added_recovery = model_dictionary["use_yaw_added_recovery"]
        self.eps_gain = model_dictionary["eps_gain"]

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
        Main function for calculating the IshiharaQian Gaussian wake model.
        This method calculates and returns the wake velocity deficits caused by
        the specified turbine, relative to the freestream velocities at the
        grid of points comprising the wind farm flow field. This function is
        accessible through the :py:class:`~.Wake` class as the
        :py:meth:`~.Wake.velocity_function` method.

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
                velocities for the U, V, and W components, aligned with the x,
                y, and z directions, respectively. The three arrays contain the
                velocity deficits at each grid point in the flow field.
        """
        # added turbulence model
        TI = turbine._turbulence_intensity

        # turbine parameters
        D = turbine.rotor_diameter

        yaw = -1 * turbine.yaw_angle  # opposite sign convention in this model
        Ct = turbine.Ct
        U_local = flow_field.u_initial
        local_x = x_locations - turbine_coord.x1
        local_y = y_locations - turbine_coord.x2
        local_z = z_locations - turbine_coord.x3  # adjust for hub height

        # coordinate info
        r = np.sqrt(local_y ** 2 + (local_z) ** 2)

        kstar = self.parameter_value_from_dict(self.kstar, Ct, TI)
        epsilon = self.parameter_value_from_dict(self.epsilon, Ct, TI)
        a = self.parameter_value_from_dict(self.a, Ct, TI)
        b = self.parameter_value_from_dict(self.b, Ct, TI)
        c = self.parameter_value_from_dict(self.c, Ct, TI)

        k1 = np.cos(np.pi / 2 * (r / D - 0.5)) ** 2
        k1[r / D > 0.5] = 1.0

        k2 = np.cos(np.pi / 2 * (r / D + 0.5)) ** 2
        k2[r / D > 0.5] = 1.0

        # Representative wake width = \sigma / D
        wake_width = kstar * (local_x / D) + epsilon

        # wake velocity deficit = \Delta U (x,y,z) / U_h
        C = 1 / (a + b * (local_x / D) + c * (1 + (local_x / D)) ** (-2)) ** 2
        r_tilde = r
        n = 2
        sigma_tilde = wake_width * D
        velDef = GaussianModel.gaussian_function(U_local, C, r_tilde, n, sigma_tilde)

        # trim wakes to 1 D upstream to avoid artifacts
        yR = y_locations - turbine_coord.x2
        xR = yR * tand(yaw) + turbine_coord.x1 - D
        velDef[x_locations < xR] = 0

        return velDef, np.zeros(np.shape(velDef)), np.zeros(np.shape(velDef))

    def parameter_value_from_dict(self, pdict, Ct, TI):
        """
        Function to calculate model parameters using current conditions and
        model dictionaries.

        Args:
            pdict (dict): Wake turbulence parameter.
            Ct (float): Thrust coefficient of the current turbine.
            ti_initial (float): Turbulence intensity.

        Returns:
            float: Current value of model parameter.
        """
        return pdict["const"] * Ct ** (pdict["Ct"]) * TI ** (pdict["TI"])

    @property
    def kstar(self):
        """
        Parameter that is used to determine the linear relationship between the
        turbulence intensity and the width of the Gaussian wake shape.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._kstar

    @kstar.setter
    def kstar(self, value):
        if not (type(value) is dict and set(value) == set(["const", "Ct", "TI"])):
            err_msg = (
                "Invalid value type given for kstar: {}, expected "
                + 'dict with keys ["const", "Ct", "TI"]'
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._kstar = value
        if value != __class__.default_parameters["kstar"]:
            self.logger.info(
                (
                    "Current value of kstar, {0}, is not equal to tuned "
                    + "value of {1}."
                ).format(value, __class__.default_parameters["kstar"])
            )

    @property
    def epsilon(self):
        """
        Parameter that is used to determine the linear relationship between the
        turbulence intensity and the width of the Gaussian wake shape.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        if not (type(value) is dict and set(value) == set(["const", "Ct", "TI"])):
            err_msg = (
                "Invalid value type given for epsilon: {}, expected "
                + 'dict with keys ["const", "Ct", "TI"]'
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._epsilon = value
        if value != __class__.default_parameters["epsilon"]:
            self.logger.info(
                (
                    "Current value of epsilon, {0}, is not equal to tuned "
                    + "value of {1}."
                ).format(value, __class__.default_parameters["epsilon"])
            )

    @property
    def a(self):
        """
        Constant coefficient used in calculation of wake-added turbulence.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._a

    @a.setter
    def a(self, value):
        if not (type(value) is dict and set(value) == set(["const", "Ct", "TI"])):
            err_msg = (
                "Invalid value type given for a: {}, expected "
                + 'dict with keys ["const", "Ct", "TI"]'
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._a = value
        if value != __class__.default_parameters["a"]:
            self.logger.info(
                (
                    "Current value of a, {0}, is not equal to tuned " + "value of {1}."
                ).format(value, __class__.default_parameters["a"])
            )

    @property
    def b(self):
        """
        Linear coefficient used in calculation of wake-added turbulence.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._b

    @b.setter
    def b(self, value):
        if not (type(value) is dict and set(value) == set(["const", "Ct", "TI"])):
            err_msg = (
                "Invalid value type given for b: {}, expected "
                + 'dict with keys ["const", "Ct", "TI"]'
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._b = value
        if value != __class__.default_parameters["b"]:
            self.logger.info(
                (
                    "Current value of b, {0}, is not equal to tuned " + "value of {1}."
                ).format(value, __class__.default_parameters["b"])
            )

    @property
    def c(self):
        """
        Near-wake coefficient used in calculation of wake-added turbulence.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._c

    @c.setter
    def c(self, value):
        if not (type(value) is dict and set(value) == set(["const", "Ct", "TI"])):
            err_msg = (
                "Invalid value type given for c: {}, expected "
                + 'dict with keys ["const", "Ct", "TI"]'
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._c = value
        if value != __class__.default_parameters["c"]:
            self.logger.info(
                (
                    "Current value of c, {0}, is not equal to tuned " + "value of {1}."
                ).format(value, __class__.default_parameters["c"])
            )
