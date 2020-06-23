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


class LegacyGauss(GaussianModel):
    """
    The LegacyGauss model ports the previous Gauss model to the new FLORIS
    framework of inheritance of the GaussianModel. It is based on the gaussian
    wake models described in :cite:`glvm-bastankhah2014new`,
    :cite:`glvm-abkar2015influence`, :cite:`glvm-bastankhah2016experimental`,
    :cite:`glvm-niayifar2016analytical`, and :cite:`glvm-dilip2017wind`.

    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: glvm-
    """

    default_parameters = {
        "ka": 0.38,
        "kb": 0.004,
        "alpha": 0.58,
        "beta": 0.077,
        "calculate_VW_velocities": False,
        "use_yaw_added_recovery": False,
        "eps_gain": 0.2,
    }

    def __init__(self, parameter_dictionary):
        """
        Stores model parameters for use by methods.

        Args:
            parameter_dictionary (dict): Model-specific parameters.
                Default values are used when a parameter is not included
                in `parameter_dictionary`. Possible key-value pairs include:

                    -   **ka**: Parameter used to determine the linear
                        relationship between the turbulence intensity and the
                        width of the Gaussian wake shape.
                    -   **kb**: Parameter used to determine the linear
                        relationship between the turbulence intensity and the
                        width of the Gaussian wake shape.
                    -   **alpha**: Parameter that determines the dependence of
                        the downstream boundary between the near wake and far
                        wake region on the turbulence intensity.
                    -   **beta**: Parameter that determines the dependence of
                        the downstream boundary between the near wake and far
                        wake region on the turbine's induction factor.

        """

        super().__init__(parameter_dictionary)

        self.model_string = "gauss_legacy"
        model_dictionary = self._get_model_dict(__class__.default_parameters)

        # near wake / far wake boundary parameters
        self.alpha = model_dictionary["alpha"]
        self.beta = model_dictionary["beta"]

        # wake expansion parameters
        self.ka = model_dictionary["ka"]
        self.kb = model_dictionary["kb"]

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
        Using the Gaussian wake model, this method calculates and
        returns the wake velocity deficits, caused by the specified turbine,
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
                velocities for the U, V, and W components, aligned with the x, y,
                and z directions, respectively. The three arrays contain the
                velocity deficits at each grid point in the flow field.
        """
        # veer (degrees)
        veer = flow_field.wind_veer

        # added turbulence model
        TI = turbine.current_turbulence_intensity

        # turbine parameters
        D = turbine.rotor_diameter
        HH = turbine.hub_height
        yaw = -1 * turbine.yaw_angle  # opposite sign convention in this model
        Ct = turbine.Ct
        U_local = flow_field.u_initial

        # wake deflection
        delta = deflection_field

        xR, _ = GaussianModel.mask_upstream_wake(y_locations, turbine_coord, yaw)
        uR, u0 = GaussianModel.initial_velocity_deficits(U_local, Ct)
        sigma_y0, sigma_z0 = GaussianModel.initial_wake_expansion(
            turbine, U_local, veer, uR, u0
        )

        # quantity that determines when the far wake starts
        x0 = (
            D
            * (cosd(yaw) * (1 + np.sqrt(1 - Ct)))
            / (
                np.sqrt(2)
                * (4 * self.alpha * TI + 2 * self.beta * (1 - np.sqrt(1 - Ct)))
            )
            + turbine_coord.x1
        )

        # velocity deficit in the near wake
        sigma_y = (((x0 - xR) - (x_locations - xR)) / (x0 - xR)) * 0.501 * D * np.sqrt(
            Ct / 2.0
        ) + ((x_locations - xR) / (x0 - xR)) * sigma_y0
        sigma_z = (((x0 - xR) - (x_locations - xR)) / (x0 - xR)) * 0.501 * D * np.sqrt(
            Ct / 2.0
        ) + ((x_locations - xR) / (x0 - xR)) * sigma_z0
        sigma_y[x_locations < xR] = 0.5 * D
        sigma_z[x_locations < xR] = 0.5 * D

        a = cosd(veer) ** 2 / (2 * sigma_y ** 2) + sind(veer) ** 2 / (2 * sigma_z ** 2)
        b = -sind(2 * veer) / (4 * sigma_y ** 2) + sind(2 * veer) / (4 * sigma_z ** 2)
        c = sind(veer) ** 2 / (2 * sigma_y ** 2) + cosd(veer) ** 2 / (2 * sigma_z ** 2)
        r = (
            a * ((y_locations - turbine_coord.x2) - delta) ** 2
            - 2 * b * ((y_locations - turbine_coord.x2) - delta) * ((z_locations - HH))
            + c * ((z_locations - HH)) ** 2
        )
        C = 1 - np.sqrt(1 - (Ct * cosd(yaw) / (8.0 * sigma_y * sigma_z / D ** 2)))

        velDef = GaussianModel.gaussian_function(U_local, C, r, 1, np.sqrt(0.5))
        velDef[x_locations < xR] = 0
        velDef[x_locations > x0] = 0

        # wake expansion in the lateral (y) and the vertical (z)
        ky = self.ka * TI + self.kb  # wake expansion parameters
        kz = self.ka * TI + self.kb  # wake expansion parameters
        sigma_y = ky * (x_locations - x0) + sigma_y0
        sigma_z = kz * (x_locations - x0) + sigma_z0
        sigma_y[x_locations < x0] = sigma_y0[x_locations < x0]
        sigma_z[x_locations < x0] = sigma_z0[x_locations < x0]

        # velocity deficit outside the near wake
        a = cosd(veer) ** 2 / (2 * sigma_y ** 2) + sind(veer) ** 2 / (2 * sigma_z ** 2)
        b = -sind(2 * veer) / (4 * sigma_y ** 2) + sind(2 * veer) / (4 * sigma_z ** 2)
        c = sind(veer) ** 2 / (2 * sigma_y ** 2) + cosd(veer) ** 2 / (2 * sigma_z ** 2)
        r = (
            a * (y_locations - turbine_coord.x2 - delta) ** 2
            - 2 * b * (y_locations - turbine_coord.x2 - delta) * (z_locations - HH)
            + c * (z_locations - HH) ** 2
        )
        C = 1 - np.sqrt(1 - (Ct * cosd(yaw) / (8.0 * sigma_y * sigma_z / D ** 2)))

        # compute velocities in the far wake
        velDef1 = GaussianModel.gaussian_function(U_local, C, r, 1, np.sqrt(0.5))
        velDef1[x_locations < x0] = 0

        U = np.sqrt(velDef ** 2 + velDef1 ** 2)

        return U, np.zeros(np.shape(velDef1)), np.zeros(np.shape(velDef1))

    @property
    def ka(self):
        """
        Parameter used to determine the linear relationship between the
        turbulence intensity and the width of the Gaussian wake shape.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._ka

    @ka.setter
    def ka(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for ka: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._ka = value
        if value != __class__.default_parameters["ka"]:
            self.logger.info(
                (
                    "Current value of ka, {0}, is not equal to tuned " + "value of {1}."
                ).format(value, __class__.default_parameters["ka"])
            )

    @property
    def kb(self):
        """
        Parameter used to determine the linear relationship between the
        turbulence intensity and the width of the Gaussian wake shape.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._kb

    @kb.setter
    def kb(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for kb: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._kb = value
        if value != __class__.default_parameters["kb"]:
            self.logger.info(
                (
                    "Current value of kb, {0}, is not equal to tuned " + "value of {1}."
                ).format(value, __class__.default_parameters["kb"])
            )

    @property
    def alpha(self):
        """
        Parameter that determines the dependence of the downstream boundary
        between the near wake and far wake region on the turbulence
        intensity.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for alpha: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._alpha = value
        if value != __class__.default_parameters["alpha"]:
            self.logger.info(
                (
                    "Current value of alpha, {0}, is not equal to tuned "
                    + "value of {1}."
                ).format(value, __class__.default_parameters["alpha"])
            )

    @property
    def beta(self):
        """
        Parameter that determines the dependence of the downstream boundary
        between the near wake and far wake region on the turbine's
        induction factor.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._beta

    @beta.setter
    def beta(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for beta: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._beta = value
        if value != __class__.default_parameters["beta"]:
            self.logger.info(
                (
                    "Current value of beta, {0}, is not equal to tuned "
                    + "value of {1}."
                ).format(value, __class__.default_parameters["beta"])
            )
