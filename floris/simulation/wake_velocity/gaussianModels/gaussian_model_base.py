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

from ....utilities import cosd, sind, tand
from ..base_velocity_deficit import VelocityDeficit


class GaussianModel(VelocityDeficit):
    """
    This is the super-class for all Gaussian-type wake models. It includes
    implementations of functions that subclasses should use to perform
    Gaussian-related calculations (see :cite:`gmb-King2019Controls`)

    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: gmb-
    """

    def __init__(self, parameter_dictionary):
        """
        See super-class for initialization details.

        Args:
            parameter_dictionary (dict): Model-specific parameters.
        """
        super().__init__(parameter_dictionary)

    def yaw_added_turbulence_mixing(
        self, coord, turbine, flow_field, x_locations, y_locations, z_locations
    ):
        """
        This method computes the added turbulence due to yaw misalignment.  Yaw misalignment induces additional
        mixing in the flow and causes the wakes to recover faster.
        Args:
            coord (:py:obj:`floris.simulation.turbine_map.TurbineMap.coords`):
                Spatial coordinates of wind turbine.
            turbine (:py:class:`floris.simulation.turbine.Turbine`):
                Turbine object.
            flow_field ([type]): [description]
            x_locations (np.array): Streamwise locations in wake.
            y_locations (np.array): Spanwise locations in wake.
            z_locations (np.array): Vertical locations in wake.
        Returns:
            np.array: turbulence mixing adjustment.
        """
        if self.use_yaw_added_recovery:
            # compute turbulence modification
            V, W = self.calc_VW(
                coord, turbine, flow_field, x_locations, y_locations, z_locations
            )

            # calculate fluctuations
            v_prime = flow_field.v + V
            w_prime = flow_field.w + W

            # get u_prime from current turbulence intensity
            u_prime = turbine.u_prime()

            # compute the new TKE
            idx = np.where(
                (np.abs(x_locations - coord.x1) <= turbine.rotor_diameter / 4)
                & (np.abs(y_locations - coord.x2) < turbine.rotor_diameter)
            )
            TKE = (1 / 2) * (
                u_prime ** 2 + np.mean(v_prime[idx]) ** 2 + np.mean(w_prime[idx]) ** 2
            )

            # convert TKE back to TI
            TI_total = turbine.TKE_to_TI(TKE)

            # convert to turbulence due to mixing
            TI_mixing = np.array(TI_total) - turbine.current_turbulence_intensity
        else:
            TI_mixing = 0.0

        return TI_mixing

    def calculate_VW(
        self, V, W, coord, turbine, flow_field, x_locations, y_locations, z_locations
    ):
        """
        This method calculates the V- and W-component velocities using
        methods developed in [1].
        # TODO add reference to 1
        # TODO is this function needed? It simply calls another function

        Args:
            V (np.array): V-component velocity deficits across the flow field.
            W (np.array): W-component velocity deficits across the flow field.
            coord (:py:obj:`floris.simulation.turbine_map.TurbineMap.coords`):
                Spatial coordinates of wind turbine.
            turbine (:py:class:`floris.simulation.turbine.Turbine`):
                Turbine object.
            flow_field ([type]): [description]
            x_locations (np.array): Streamwise locations in wake.
            y_locations (np.array): Spanwise locations in wake.
            z_locations (np.array): Vertical locations in wake.

        Raises:
            ValueError: It appears that 'use_yaw_added_recovery' is set
                to True and 'calculate_VW_velocities' is set to False.
                This configuration is not valid. Please set
                'calculate_VW_velocities' to True if you wish to use
                yaw-added recovery.

        Returns:
            np.array, np.array:

                - V-component velocity deficits across the flow field.
                - W-component velocity deficits across the flow field.
        """
        if self.use_yaw_added_recovery:
            if not self.calculate_VW_velocities:
                err_msg = (
                    "It appears that 'use_yaw_added_recovery' is set "
                    + "to True and 'calculate_VW_velocities' is set to False. "
                    + "This configuration is not valid. Please set "
                    + "'calculate_VW_velocities' to True if you wish to use "
                    + "yaw-added recovery."
                )
                self.logger.error(err_msg, stack_info=True)
                raise ValueError(err_msg)
        if self.calculate_VW_velocities:
            V, W = self.calc_VW(
                coord, turbine, flow_field, x_locations, y_locations, z_locations
            )
        return V, W

    def yaw_added_recovery_correction(
        self, U_local, U, W, x_locations, y_locations, turbine, turbine_coord
    ):
        """
        This method corrects the U-component velocities when yaw added recovery
        is enabled. For more details on how the velocities are changed, see [1].
        # TODO add reference to 1

        Args:
            U_local (np.array): U-component velocities across the flow field.
            U (np.array): U-component velocity deficits across the flow field.
            W (np.array): W-component velocity deficits across the flow field.
            x_locations (np.array): Streamwise locations in wake.
            y_locations (np.array): Spanwise locations in wake.
            turbine (:py:class:`floris.simulation.turbine.Turbine`):
                Turbine object.
            turbine_coord (:py:obj:`floris.simulation.turbine_map.TurbineMap.coords`):
                Spatial coordinates of wind turbine.

        Returns:
            np.array: U-component velocity deficits across the flow field.
        """
        # compute the velocity without modification
        U1 = U_local - U

        # set dimensions
        D = turbine.rotor_diameter
        xLocs = x_locations - turbine_coord.x1
        ky = self.ka * turbine.current_turbulence_intensity + self.kb
        U2 = (np.mean(W) * xLocs) / ((ky * xLocs + D / 2))
        U_total = U1 + np.nan_to_num(U2)

        # turn it back into a deficit
        U = U_local - U_total

        # zero out anything before the turbine
        U[x_locations < turbine_coord.x1] = 0

        return U

    def calc_VW(
        self, coord, turbine, flow_field, x_locations, y_locations, z_locations
    ):
        """
        This method calculates the V- and W-component velocities using
        methods developed in [1].
        # TODO add reference to 1

        Args:
            coord (:py:obj:`floris.simulation.turbine_map.TurbineMap.coords`):
                Spatial coordinates of wind turbine.
            turbine (:py:class:`floris.simulation.turbine.Turbine`):
                Turbine object.
            flow_field ([type]): [description]
            x_locations (np.array): Streamwise locations in wake.
            y_locations (np.array): Spanwise locations in wake.
            z_locations (np.array): Vertical locations in wake.

        Returns:
            np.array, np.array:

                - V-component velocity deficits across the flow field.
                - W-component velocity deficits across the flow field.
        """
        # turbine parameters
        D = turbine.rotor_diameter
        HH = turbine.hub_height
        yaw = turbine.yaw_angle
        Ct = turbine.Ct
        TSR = turbine.tsr
        aI = turbine.aI

        # flow parameters
        Uinf = np.mean(flow_field.wind_map.grid_wind_speed)

        scale = 1.0
        vel_top = (
            Uinf
            * ((HH + D / 2) / flow_field.specified_wind_height) ** flow_field.wind_shear
        ) / Uinf
        vel_bottom = (
            Uinf
            * ((HH - D / 2) / flow_field.specified_wind_height) ** flow_field.wind_shear
        ) / Uinf
        Gamma_top = (
            scale * (np.pi / 8) * D * vel_top * Uinf * Ct * sind(yaw) * cosd(yaw)
        )
        Gamma_bottom = (
            -scale * (np.pi / 8) * D * vel_bottom * Uinf * Ct * sind(yaw) * cosd(yaw)
        )
        Gamma_wake_rotation = (
            0.25 * 2 * np.pi * D * (aI - aI ** 2) * turbine.average_velocity / TSR
        )

        # compute the spanwise and vertical velocities induced by yaw
        eps = self.eps_gain * D  # Use set value

        # decay the vortices as they move downstream - using mixing length
        lmda = D / 8
        kappa = 0.41
        lm = kappa * z_locations / (1 + kappa * z_locations / lmda)
        z = np.linspace(
            z_locations.min(), z_locations.max(), flow_field.u_initial.shape[2]
        )
        dudz_initial = np.gradient(flow_field.u_initial, z, axis=2)
        nu = lm ** 2 * np.abs(dudz_initial[0, :, :])

        # top vortex
        yLocs = y_locations + 0.01 - (coord.x2)
        zT = z_locations + 0.01 - (HH + D / 2)
        rT = yLocs ** 2 + zT ** 2
        V1 = (
            (zT * Gamma_top)
            / (2 * np.pi * rT)
            * (1 - np.exp(-rT / (eps ** 2)))
            * eps ** 2
            / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)
        )

        W1 = (
            (-yLocs * Gamma_top)
            / (2 * np.pi * rT)
            * (1 - np.exp(-rT / (eps ** 2)))
            * eps ** 2
            / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)
        )

        # bottom vortex
        zB = z_locations + 0.01 - (HH - D / 2)
        rB = yLocs ** 2 + zB ** 2
        V2 = (
            (zB * Gamma_bottom)
            / (2 * np.pi * rB)
            * (1 - np.exp(-rB / (eps ** 2)))
            * eps ** 2
            / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)
        )

        W2 = (
            ((-yLocs * Gamma_bottom) / (2 * np.pi * rB))
            * (1 - np.exp(-rB / (eps ** 2)))
            * eps ** 2
            / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)
        )

        # top vortex - ground
        yLocs = y_locations + 0.01 - (coord.x2)
        zLocs = z_locations + 0.01 + (HH + D / 2)
        V3 = (
            (
                ((zLocs * -Gamma_top) / (2 * np.pi * (yLocs ** 2 + zLocs ** 2)))
                * (1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2)))
                + 0.0
            )
            * eps ** 2
            / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)
        )

        W3 = (
            ((-yLocs * -Gamma_top) / (2 * np.pi * (yLocs ** 2 + zLocs ** 2)))
            * (1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2)))
            * eps ** 2
            / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)
        )

        # bottom vortex - ground
        yLocs = y_locations + 0.01 - (coord.x2)
        zLocs = z_locations + 0.01 + (HH - D / 2)
        V4 = (
            (
                ((zLocs * -Gamma_bottom) / (2 * np.pi * (yLocs ** 2 + zLocs ** 2)))
                * (1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2)))
                + 0.0
            )
            * eps ** 2
            / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)
        )

        W4 = (
            ((-yLocs * -Gamma_bottom) / (2 * np.pi * (yLocs ** 2 + zLocs ** 2)))
            * (1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2)))
            * eps ** 2
            / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)
        )

        # wake rotation vortex
        zC = z_locations + 0.01 - (HH)
        rC = yLocs ** 2 + zC ** 2
        V5 = (
            (zC * Gamma_wake_rotation)
            / (2 * np.pi * rC)
            * (1 - np.exp(-rC / (eps ** 2)))
            * eps ** 2
            / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)
        )

        W5 = (
            (-yLocs * Gamma_wake_rotation)
            / (2 * np.pi * rC)
            * (1 - np.exp(-rC / (eps ** 2)))
            * eps ** 2
            / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)
        )

        # wake rotation vortex - ground effect
        yLocs = y_locations + 0.01 - coord.x2
        zLocs = z_locations + 0.01 + HH
        V6 = (
            (
                (
                    (zLocs * -Gamma_wake_rotation)
                    / (2 * np.pi * (yLocs ** 2 + zLocs ** 2))
                )
                * (1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2)))
                + 0.0
            )
            * eps ** 2
            / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)
        )

        W6 = (
            ((-yLocs * -Gamma_wake_rotation) / (2 * np.pi * (yLocs ** 2 + zLocs ** 2)))
            * (1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2)))
            * eps ** 2
            / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)
        )

        # total spanwise velocity
        V = V1 + V2 + V3 + V4 + V5 + V6
        W = W1 + W2 + W3 + W4 + W5 + W6

        # no spanwise and vertical velocity upstream of the turbine
        V[
            x_locations < coord.x1 - 1
        ] = 0.0  # Subtract by 1 to avoid numerical issues on rotation
        W[
            x_locations < coord.x1 - 1
        ] = 0.0  # Subtract by 1 to avoid numerical issues on rotation

        W[W < 0] = 0

        return V, W

    @property
    def calculate_VW_velocities(self):
        """
        Flag to enable the calculation of V- and W-component velocities using
        methods developed in [1].

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (bool): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._calculate_VW_velocities

    @calculate_VW_velocities.setter
    def calculate_VW_velocities(self, value):
        if type(value) is not bool:
            err_msg = (
                "Value of calculate_VW_velocities must be type "
                + "float; {} given.".format(type(value))
            )
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._calculate_VW_velocities = value

    @property
    def use_yaw_added_recovery(self):
        """
        Flag to use yaw added recovery on the wake velocity using methods
        developed in [1].

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (bool): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._use_yaw_added_recovery

    @use_yaw_added_recovery.setter
    def use_yaw_added_recovery(self, value):
        if type(value) is not bool:
            # TODO Shouldn't this be a bool?
            err_msg = (
                "Value of use_yaw_added_recovery must be type "
                + "float; {} given.".format(type(value))
            )
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._use_yaw_added_recovery = value

    @property
    def eps_gain(self):
        """
        Tuning value for calculating the V- and W- component velocities using
        methods developed in [1].

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
        if type(value) is not float:
            err_msg = "Value of eps_gain must be type " + "float; {} given.".format(
                type(value)
            )
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._eps_gain = value

    @staticmethod
    def mask_upstream_wake(y_locations, turbine_coord, yaw):
        """
        Calculates values to be used for masking the upstream wake relative to
        the current turbine.

        Args:
            y_locations (np.array): Spanwise locations in wake.
            turbine_coord (:py:obj:`floris.simulation.turbine_map.TurbineMap.coords`):
                Spatial coordinates of wind turbine.
            yaw (float): The turbine yaw angle.

        Returns:
            tuple: tuple containing:

                -   yR (np.array): Y locations to mask upstream wake.
                -   xR (np.array): X locations to mask upstream wake.
        """
        yR = y_locations - turbine_coord.x2
        xR = yR * tand(yaw) + turbine_coord.x1
        return xR, yR

    @staticmethod
    def initial_velocity_deficits(U_local, Ct):
        """
        Calculates the initial velocity deficits used in determining the wake
        expansion in a Gaussian wake velocity model.

        Args:
            U_local (np.array): U-component velocities across the flow field.
            Ct (float): The thrust coefficient of a turbine at the current
                operating conditions.

        Returns:
            tuple: tuple containing:

                -   uR (np.array): Initial velocity deficit used in calculation
                    of wake expansion.
                -   u0 (np.array): Initial velocity deficit used in calculation
                    of wake expansion.
        """
        uR = U_local * Ct / (2.0 * (1 - np.sqrt(1 - Ct)))
        u0 = U_local * np.sqrt(1 - Ct)
        return uR, u0

    @staticmethod
    def initial_wake_expansion(turbine, U_local, veer, uR, u0):
        """
        Calculates the initial wake widths associated with wake expansion.

        Args:
            turbine (:py:class:`floris.simulation.turbine.Turbine`):
                Turbine object.
            U_local (np.array): U-component velocities across the flow field.
            veer (float): The amount of veer across the rotor.
            uR (np.array): Initial velocity deficit used in calculation of wake
                expansion.
            u0 (np.array): Initial velocity deficit used in calculation of wake
                expansion.

        Returns:
            tuple: tuple containing:

                -   sigma_y0 (np.array): Initial wake width in the spanwise
                    direction.
                -   sigma_z0 (np.array): Initial wake width in the vertical
                    direction.
        """
        yaw = -1 * turbine.yaw_angle
        sigma_z0 = turbine.rotor_diameter * 0.5 * np.sqrt(uR / (U_local + u0))
        sigma_y0 = sigma_z0 * cosd(yaw) * cosd(veer)
        return sigma_y0, sigma_z0

    @staticmethod
    def gaussian_function(U, C, r, n, sigma):
        """
        A general form of the Gaussian function used in the Gaussian wake
        models.

        Args:
            U (np.array): U-component velocities across the flow field.
            C (np.array): Velocity deficit at the wake center normalized by the
                incoming wake velocity.
            r (float): Radial distance from the wake center.
            n (float): Exponent of radial distance from the wake center.
            sigma (np.array): Standard deviation of the wake.

        Returns:
            np.array: U (np.array): U-component velocity deficits across the
            flow field.
        """
        return U * C * np.exp(-1 * r ** n / (2 * sigma ** 2))
