# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import numpy as np

from ...utilities import cosd, sind, tand
from .base_velocity_deflection import VelocityDeflection


class Gauss(VelocityDeflection):
    """
    The Gauss deflection model is a blend of the models described in
    :cite:`gdm-bastankhah2016experimental` and :cite:`gdm-King2019Controls` for
    calculating the deflection field in turbine wakes.

    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: gdm-
    """

    default_parameters = {
        "ka": 0.38,
        "kb": 0.004,
        "alpha": 0.58,
        "beta": 0.077,
        "ad": 0.0,
        "bd": 0.0,
        "dm": 1.0,
        "use_secondary_steering": True,
        "eps_gain": 0.2,
    }

    def __init__(self, parameter_dictionary):
        """
        Stores model parameters for use by methods.

        Args:
            parameter_dictionary (dict): Model-specific parameters.
                Default values are used when a parameter is not included
                in `parameter_dictionary`. Possible key-value pairs include:

                    -   **ka** (*float*): Parameter used to determine the linear
                        relationship between the turbulence intensity and the
                        width of the Gaussian wake shape.
                    -   **kb** (*float*): Parameter used to determine the linear
                        relationship between the turbulence intensity and the
                        width of the Gaussian wake shape.
                    -   **alpha** (*float*): Parameter that determines the
                        dependence of the downstream boundary between the near
                        wake and far wake region on the turbulence intensity.
                    -   **beta** (*float*): Parameter that determines the
                        dependence of the downstream boundary between the near
                        wake and far wake region on the turbine's induction
                        factor.
                    -   **ad** (*float*): Additional tuning parameter to modify
                        the wake deflection with a lateral offset.
                        Defaults to 0.
                    -   **bd** (*float*): Additional tuning parameter to modify
                        the wake deflection with a lateral offset.
                        Defaults to 0.
                    -   **dm** (*float*): Additional tuning parameter to scale
                        the amount of wake deflection. Defaults to 1.0
                    -   **use_secondary_steering** (*bool*): Flag to use
                        secondary steering on the wake velocity using methods
                        developed in [2].
                    -   **eps_gain** (*float*): Tuning value for calculating
                        the V- and W-component velocities using methods
                        developed in [7].
                        TODO: Believe this should be removed, need to verify.
                        See property on super-class for more details.
        """
        super().__init__(parameter_dictionary)
        self.model_string = "gauss"
        model_dictionary = self._get_model_dict(__class__.default_parameters)
        self.ka = model_dictionary["ka"]
        self.kb = model_dictionary["kb"]
        self.ad = model_dictionary["ad"]
        self.bd = model_dictionary["bd"]
        self.alpha = model_dictionary["alpha"]
        self.beta = model_dictionary["beta"]
        self.dm = model_dictionary["dm"]
        self.use_secondary_steering = model_dictionary["use_secondary_steering"]
        self.eps_gain = model_dictionary["eps_gain"]

    def function(
        self, x_locations, y_locations, z_locations, turbine, coord, flow_field
    ):
        """
        Calculates the deflection field of the wake. See
        :cite:`gdm-bastankhah2016experimental` and :cite:`gdm-King2019Controls`
        for details on the methods used.

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
            coord (:py:obj:`floris.utilities.Vec3`): Object containing
                the coordinate of the turbine creating the wake (m).
            flow_field (:py:class:`floris.simulation.flow_field`): Object
                containing the flow field information for the wind farm.

        Returns:
            np.array: Deflection field for the wake.
        """
        # ==============================================================
        # free-stream velocity (m/s)
        wind_speed = flow_field.wind_map.grid_wind_speed
        # veer (degrees)
        veer = flow_field.wind_veer

        # added turbulence model
        TI = turbine.current_turbulence_intensity
        # TI = flow_field.wind_map.grid_turbulence_intensity
        # hard-coded model input data (goes in input file)
        ka = self.ka  # wake expansion parameter
        kb = self.kb  # wake expansion parameter
        alpha = self.alpha  # near wake parameter
        beta = self.beta  # near wake parameter
        ad = self.ad  # natural lateral deflection parameter
        bd = self.bd  # natural lateral deflection parameter

        # turbine parameters
        D = turbine.rotor_diameter
        yaw = -1 * self.calculate_effective_yaw_angle(
            x_locations, y_locations, z_locations, turbine, coord, flow_field
        )
        # opposite sign convention in this model
        tilt = turbine.tilt_angle
        Ct = turbine.Ct

        # U_local = flow_field.wind_map.grid_wind_speed
        # just a placeholder for now, should be initialized with the flow_field
        U_local = flow_field.u_initial

        # initial velocity deficits
        uR = (
            U_local
            * Ct
            * cosd(tilt)
            * cosd(yaw)
            / (2.0 * (1 - np.sqrt(1 - (Ct * cosd(tilt) * cosd(yaw)))))
        )
        u0 = U_local * np.sqrt(1 - Ct)

        # length of near wake
        x0 = (
            D
            * (cosd(yaw) * (1 + np.sqrt(1 - Ct * cosd(yaw))))
            / (np.sqrt(2) * (4 * alpha * TI + 2 * beta * (1 - np.sqrt(1 - Ct))))
            + coord.x1
        )

        # wake expansion parameters
        ky = ka * TI + kb
        kz = ka * TI + kb

        C0 = 1 - u0 / wind_speed
        M0 = C0 * (2 - C0)
        E0 = C0 ** 2 - 3 * np.exp(1.0 / 12.0) * C0 + 3 * np.exp(1.0 / 3.0)

        # initial Gaussian wake expansion
        sigma_z0 = D * 0.5 * np.sqrt(uR / (U_local + u0))
        sigma_y0 = sigma_z0 * cosd(yaw) * cosd(veer)

        yR = y_locations - coord.x2
        xR = yR * tand(yaw) + coord.x1

        # yaw parameters (skew angle and distance from centerline)
        theta_c0 = (
            self.dm
            * (0.3 * np.radians(yaw) / cosd(yaw))
            * (1 - np.sqrt(1 - Ct * cosd(yaw)))
        )  # skew angle in radians
        delta0 = np.tan(theta_c0) * (x0 - coord.x1)  # initial wake deflection;
        # NOTE: use np.tan here since theta_c0 is radians

        # deflection in the near wake
        delta_near_wake = ((x_locations - xR) / (x0 - xR)) * delta0 + (
            ad + bd * (x_locations - coord.x1)
        )
        delta_near_wake[x_locations < xR] = 0.0
        delta_near_wake[x_locations > x0] = 0.0

        # deflection in the far wake
        sigma_y = ky * (x_locations - x0) + sigma_y0
        sigma_z = kz * (x_locations - x0) + sigma_z0
        sigma_y[x_locations < x0] = sigma_y0[x_locations < x0]
        sigma_z[x_locations < x0] = sigma_z0[x_locations < x0]

        ln_deltaNum = (1.6 + np.sqrt(M0)) * (
            1.6 * np.sqrt(sigma_y * sigma_z / (sigma_y0 * sigma_z0)) - np.sqrt(M0)
        )
        ln_deltaDen = (1.6 - np.sqrt(M0)) * (
            1.6 * np.sqrt(sigma_y * sigma_z / (sigma_y0 * sigma_z0)) + np.sqrt(M0)
        )

        delta_far_wake = (
            delta0
            + (theta_c0 * E0 / 5.2)
            * np.sqrt(sigma_y0 * sigma_z0 / (ky * kz * M0))
            * np.log(ln_deltaNum / ln_deltaDen)
            + (ad + bd * (x_locations - coord.x1))
        )
        delta_far_wake[x_locations <= x0] = 0.0

        deflection = delta_near_wake + delta_far_wake

        return deflection

    def calculate_effective_yaw_angle(
        self, x_locations, y_locations, z_locations, turbine, coord, flow_field
    ):
        """
        This method determines the effective yaw angle to be used when
        secondary steering is enabled. For more details on how the effective
        yaw angle is calculated, see :cite:`bvd-King2019Controls`.

        Args:
            x_locations (np.array): Streamwise locations in wake.
            y_locations (np.array): Spanwise locations in wake.
            z_locations (np.array): Vertical locations in wake.
            turbine (:py:class:`floris.simulation.turbine.Turbine`):
                Turbine object.
            coord (:py:obj:`floris.simulation.turbine_map.TurbineMap.coords`):
                Spatial coordinates of wind turbine.
            flow_field (:py:class:`floris.simulation.flow_field.FlowField`):
                Flow field object.

        Raises:
            ValueError: It appears that 'use_secondary_steering' is set
                to True and 'calculate_VW_velocities' is set to False.
                This configuration is not valid. Please set
                'use_secondary_steering' to True if you wish to use
                yaw-added recovery.

        Returns:
            float: The turbine yaw angle, including any effective yaw if
            secondary steering is enabled.
        """
        if self.use_secondary_steering:
            if not flow_field.wake.velocity_model.calculate_VW_velocities:
                err_msg = (
                    "It appears that 'use_secondary_steering' is set "
                    + "to True and 'calculate_VW_velocities' is set to False. "
                    + "This configuration is not valid. Please set "
                    + "'use_secondary_steering' to True if you wish to use "
                    + "yaw-added recovery."
                )
                self.logger.error(err_msg, stack_info=True)
                raise ValueError(err_msg)
            # turbine parameters
            Ct = turbine.Ct
            D = turbine.rotor_diameter
            HH = turbine.hub_height
            aI = turbine.aI
            TSR = turbine.tsr
            V = flow_field.v
            Uinf = np.mean(flow_field.wind_map.grid_wind_speed)

            eps = self.eps_gain * D  # Use set value
            idx = np.where(
                (np.abs(x_locations - coord.x1) < D / 4)
                & (np.abs(y_locations - coord.x2) < D / 2)
            )

            yLocs = y_locations[idx] + 0.01 - coord.x2

            # location of top vortex
            zT = z_locations[idx] + 0.01 - (HH + D / 2)
            rT = yLocs ** 2 + zT ** 2

            # location of bottom vortex
            zB = z_locations[idx] + 0.01 - (HH - D / 2)
            rB = yLocs ** 2 + zB ** 2

            # wake rotation vortex
            zC = z_locations[idx] + 0.01 - (HH)
            rC = yLocs ** 2 + zC ** 2

            # find wake deflection from CRV
            min_yaw = -45.0
            max_yaw = 45.0
            test_yaw = np.linspace(min_yaw, max_yaw, 91)
            avg_V = np.mean(V[idx])

            # what yaw angle would have produced that same average spanwise velocity
            vel_top = (
                (HH + D / 2) / flow_field.specified_wind_height
            ) ** flow_field.wind_shear
            vel_bottom = (
                (HH - D / 2) / flow_field.specified_wind_height
            ) ** flow_field.wind_shear
            Gamma_top = (
                (np.pi / 8) * D * vel_top * Uinf * Ct * sind(test_yaw) * cosd(test_yaw)
            )
            Gamma_bottom = (
                -(np.pi / 8)
                * D
                * vel_bottom
                * Uinf
                * Ct
                * sind(test_yaw)
                * cosd(test_yaw)
            )
            Gamma_wake_rotation = (
                0.25 * 2 * np.pi * D * (aI - aI ** 2) * turbine.average_velocity / TSR
            )

            Veff = (
                np.divide(np.einsum("i,j", Gamma_top, zT), (2 * np.pi * rT))
                * (1 - np.exp(-rT / (eps ** 2)))
                + np.einsum("i,j", Gamma_bottom, zB)
                / (2 * np.pi * rB)
                * (1 - np.exp(-rB / (eps ** 2)))
                + (zC * Gamma_wake_rotation)
                / (2 * np.pi * rC)
                * (1 - np.exp(-rC / (eps ** 2)))
            )

            tmp = avg_V - np.mean(Veff, axis=1)

            # return indices of sorted residuals to find effective yaw angle
            order = np.argsort(np.abs(tmp))
            idx_1 = order[0]
            idx_2 = order[1]

            # check edge case, if true, assign max yaw value
            if idx_1 == 90 or idx_2 == 90:
                yaw_effective = max_yaw
            # check edge case, if true, assign min yaw value
            elif idx_1 == 0 or idx_2 == 0:
                yaw_effective = min_yaw
            # for each identified minimum residual, use adjacent points to determine
            # two equations of line and find the intersection of the two lines to
            # determine the effective yaw angle to add; the if/else structure is based
            # on which residual index is larger
            else:
                if idx_1 > idx_2:
                    idx_right = idx_1 + 1  # adjacent point
                    idx_left = idx_2 - 1  # adjacent point
                    mR = abs(tmp[idx_right]) - abs(tmp[idx_1])  # slope
                    mL = abs(tmp[idx_2]) - abs(tmp[idx_left])  # slope
                    bR = abs(tmp[idx_1]) - mR * float(idx_1)  # intercept
                    bL = abs(tmp[idx_2]) - mL * float(idx_2)  # intercept
                else:
                    idx_right = idx_2 + 1  # adjacent point
                    idx_left = idx_1 - 1  # adjacent point
                    mR = abs(tmp[idx_right]) - abs(tmp[idx_2])  # slope
                    mL = abs(tmp[idx_1]) - abs(tmp[idx_left])  # slope
                    bR = abs(tmp[idx_2]) - mR * float(idx_2)  # intercept
                    bL = abs(tmp[idx_1]) - mL * float(idx_1)  # intercept

                # find the value at the intersection of the two lines
                ival = (bR - bL) / (mL - mR)
                # convert the indice into degrees
                yaw_effective = ival - max_yaw

            return yaw_effective + turbine.yaw_angle
        else:
            return turbine.yaw_angle

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
        if type(value) is not float and type(value) is not int:
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
        if type(value) is not float and type(value) is not int:
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
        if type(value) is not float and type(value) is not int:
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
        if type(value) is not float and type(value) is not int:
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

    @property
    def ad(self):
        """
        Parameter available for additional tuning of the wake deflection with a
        lateral offset.

        ****TODO: Should this be removed? Not sure if it has been used.****

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._ad

    @ad.setter
    def ad(self, value):
        if type(value) is not float and type(value) is not int:
            err_msg = (
                "Invalid value type given for ad: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._ad = value
        if value != __class__.default_parameters["ad"]:
            self.logger.info(
                (
                    "Current value of ad, {0}, is not equal to tuned " + "value of {1}."
                ).format(value, __class__.default_parameters["ad"])
            )

    @property
    def bd(self):
        """
        Parameter available for additional tuning of the wake deflection with a
        lateral offset.

        ****TODO: Should this be removed? Not sure if it has been used.****

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._bd

    @bd.setter
    def bd(self, value):
        if type(value) is not float and type(value) is not int:
            err_msg = (
                "Invalid value type given for bd: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._bd = value
        if value != __class__.default_parameters["bd"]:
            self.logger.info(
                (
                    "Current value of bd, {0}, is not equal to tuned " + "value of {1}."
                ).format(value, __class__.default_parameters["bd"])
            )

    @property
    def use_secondary_steering(self):
        """
        Flag to use secondary steering on the wake deflection using methods
        developed in :cite:`bvd-King2019Controls`.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (bool): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._use_secondary_steering

    @use_secondary_steering.setter
    def use_secondary_steering(self, value):
        if type(value) is not bool:
            err_msg = (
                "Value of use_secondary_steering must be type "
                + "bool; {} given.".format(type(value))
            )
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._use_secondary_steering = value

    @property
    def eps_gain(self):
        """
        Tuning value for yaw added recovery on the wake velocity using methods
        developed in :cite:`bvd-King2019Controls`.

        TODO: Don't believe this needs to be defined here. Already defined in
        gaussian_model_model.py. Verify that it can be removed.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (bool): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._eps_gain

    @eps_gain.setter
    def eps_gain(self, value):
        if type(value) is not float and type(value) is not int:
            err_msg = "Value of eps_gain must be type " + "float; {} given.".format(
                type(value)
            )
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._eps_gain = value
