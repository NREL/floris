# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from typing import Any, Dict

import attr
import numpy as np

from floris.simulation import BaseModel
from floris.simulation import TurbineGrid
from floris.simulation import FlowField
from floris.simulation import Farm
from floris.utilities import cosd, sind, tand, float_attrib, model_attrib, bool_attrib


@attr.s(auto_attribs=True)
class GaussVelocityDeflection(BaseModel):
    """
    The Gauss deflection model is a blend of the models described in
    :cite:`gdm-bastankhah2016experimental` and :cite:`gdm-King2019Controls` for
    calculating the deflection field in turbine wakes.

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

    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: gdm-
    """
    ad: float = float_attrib(default=0.0)
    bd: float = float_attrib(default=0.0)
    alpha: float = float_attrib(default=0.58)
    beta: float = float_attrib(default=0.077)
    ka: float = float_attrib(default=0.38)
    kb: float = float_attrib(default=0.004)
    dm: float = float_attrib(default=1.0)
    eps_gain: float = float_attrib(default=0.2)
    use_secondary_steering: bool = bool_attrib(default=True)
    model_string: str = model_attrib(default="gauss")

    def prepare_function(
        self,
        grid: TurbineGrid,
        farm: Farm,
        flow_field: FlowField
    ) -> Dict[str, Any]:

        reference_rotor_diameter = farm.reference_turbine_diameter * np.ones(
            (
                flow_field.n_wind_directions,
                flow_field.n_wind_speeds,
                grid.n_turbines,
                1,
                1
            )
        )

        kwargs = dict(
            x=grid.x,
            y=grid.y,
            z=grid.z,
            freestream_velocity=flow_field.u_initial,
            wind_veer=flow_field.wind_veer,
            reference_rotor_diameter=reference_rotor_diameter,
            yaw_angle=farm.farm_controller.yaw_angles,
        )
        return kwargs

    def function(
        self,
        i: int,
        turbulence_intensity: np.ndarray,
        Ct: np.ndarray,
        *,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        freestream_velocity: np.ndarray,
        wind_veer: float,
        reference_rotor_diameter: float,
        yaw_angle: np.ndarray,
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

        # opposite sign convention in this model
        tilt = 0.0 #turbine.tilt_angle

        yaw = yaw_angle[:, :, i:i+1, None, None]

        # Ct is given for only the current turbine, so broadcast
        # this to the grid dimesions
        Ct = Ct[:, :, :, None, None] * np.ones((1,1,1,5,5))

        # Construct arrays for the current turbine's location
        x_i = np.mean(x[:, :, i:i+1], axis=(3,4))
        x_i = x_i[:, :, :, None, None]
        # x_i = x[:, :, i:i+1, :, :]
        # y_i = y[:, :, i:i+1, :, :]
        y_i = np.mean(y[:, :, i:i+1], axis=(3,4))
        y_i = y_i[:, :, :, None, None]
        z_i = np.mean(z[:, :, i:i+1], axis=(3,4))
        z_i = z_i[:, :, :, None, None]

        # initial velocity deficits
        uR = (
            freestream_velocity
          * Ct
          * cosd(tilt)
          * cosd(yaw)
          / (2.0 * (1 - np.sqrt(1 - (Ct * cosd(tilt) * cosd(yaw)))))
        )
        u0 = freestream_velocity * np.sqrt(1 - Ct)

        # length of near wake
        x0 = (
            reference_rotor_diameter
            * (cosd(yaw) * (1 + np.sqrt(1 - Ct * cosd(yaw))))
            / (np.sqrt(2) * (4 * self.alpha * turbulence_intensity + 2 * self.beta * (1 - np.sqrt(1 - Ct))))
            + x_i
        )

        # wake expansion parameters
        ky = self.ka * turbulence_intensity + self.kb
        kz = self.ka * turbulence_intensity + self.kb

        C0 = 1 - u0 / freestream_velocity
        M0 = C0 * (2 - C0)
        E0 = C0 ** 2 - 3 * np.exp(1.0 / 12.0) * C0 + 3 * np.exp(1.0 / 3.0)

        # initial Gaussian wake expansion
        sigma_z0 = reference_rotor_diameter * 0.5 * np.sqrt(uR / (freestream_velocity + u0))
        sigma_y0 = sigma_z0 * cosd(yaw) * cosd(wind_veer)

        yR = y - y_i
        xR = x_i # yR * tand(yaw) + x_i
        # print(x_i[0,0])
        # print(xR[0,0])
        # yaw parameters (skew angle and distance from centerline)
        # skew angle in radians
        theta_c0 = self.dm * (0.3 * np.radians(yaw) / cosd(yaw)) * (1 - np.sqrt(1 - Ct * cosd(yaw)))
        delta0 = np.tan(theta_c0) * (x0 - x_i)  # initial wake deflection;
        # NOTE: use np.tan here since theta_c0 is radians

        # deflection in the near wake
        delta_near_wake = ((x - xR) / (x0 - xR)) * delta0 + (self.ad + self.bd * (x - x_i))
        delta_near_wake = delta_near_wake * np.array(x >= xR)
        delta_near_wake = delta_near_wake * np.array(x <= x0)

        # deflection in the far wake
        sigma_y = ky * (x - x0) + sigma_y0
        sigma_z = kz * (x - x0) + sigma_z0
        sigma_y = sigma_y * np.array(x >= x0) + sigma_y0 * np.array(x < x0)
        sigma_z = sigma_z * np.array(x >= x0) + sigma_z0 * np.array(x < x0)

        ln_deltaNum = (1.6 + np.sqrt(M0)) * (
            1.6 * np.sqrt(sigma_y * sigma_z / (sigma_y0 * sigma_z0)) - np.sqrt(M0)
        )
        ln_deltaDen = (1.6 - np.sqrt(M0)) * (
            1.6 * np.sqrt(sigma_y * sigma_z / (sigma_y0 * sigma_z0)) + np.sqrt(M0)
        )

        delta_far_wake = (
            delta0
          + theta_c0 * E0 / 5.2
          * np.sqrt(sigma_y0 * sigma_z0 / (ky * kz * M0))
          * np.log(ln_deltaNum / ln_deltaDen)
          + (self.ad + self.bd * (x - x_i))
        )

        delta_far_wake = delta_far_wake * np.array(x > x0)
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
            aI = turbine.axial_induction
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
                (HH + D / 2) / flow_field.reference_wind_height
            ) ** flow_field.wind_shear
            vel_bottom = (
                (HH - D / 2) / flow_field.reference_wind_height
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
                yaw_effective = -min_yaw
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

