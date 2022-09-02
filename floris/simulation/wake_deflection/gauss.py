# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from __future__ import annotations

from typing import Any

import numpy as np
import numexpr as ne
from attrs import field, define
from numpy import pi, exp, log, sqrt  # noqa: F401

from floris.utilities import cosd, sind
from floris.simulation import Grid, BaseModel, FlowField


@define
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

    ad: float = field(converter=float, default=0.0)
    bd: float = field(converter=float, default=0.0)
    alpha: float = field(converter=float, default=0.58)
    beta: float = field(converter=float, default=0.077)
    ka: float = field(converter=float, default=0.38)
    kb: float = field(converter=float, default=0.004)
    dm: float = field(converter=float, default=1.0)
    eps_gain: float = field(converter=float, default=0.2)
    use_secondary_steering: bool = field(converter=bool, default=True)

    def prepare_function(
        self,
        grid: Grid,
        flow_field: FlowField,
    ) -> dict[str, Any]:

        kwargs = dict(
            x=grid.x_sorted,
            y=grid.y_sorted,
            z=grid.z_sorted,
            freestream_velocity=flow_field.u_initial_sorted,
            wind_veer=flow_field.wind_veer,
        )
        return kwargs

    def _initial_wake_expansion(self, tilt, yaw_i, ct_i, freestream_velocity, rotor_diameter_i, wind_veer):
        # initial velocity deficits
        uR = (  # noqa: F841
            freestream_velocity
            * ct_i
            * cosd(tilt)
            * cosd(yaw_i)
            / (2.0 * (1 - np.sqrt(1 - (ct_i * cosd(tilt) * cosd(yaw_i)))))
        )
        u0 = freestream_velocity * np.sqrt(1 - ct_i)  # NOTE: used once

        # wake expansion parameters
        C0 = 1 - u0 / freestream_velocity
        M0 = C0 * (2 - C0)
        E0 = ne.evaluate("C0**2 - 3 * exp(1.0 / 12.0) * C0 + 3 * exp(1.0 / 3.0)")

        # initial Gaussian wake expansion
        sigma_z0 = ne.evaluate("rotor_diameter_i * 0.5 * sqrt(uR / (freestream_velocity + u0))")
        sigma_y0 = sigma_z0 * cosd(yaw_i) * cosd(wind_veer)

        return M0, E0, sigma_y0, sigma_z0

    def function(
        self,
        x_i: np.ndarray,
        y_i: np.ndarray,
        yaw_i: np.ndarray,
        turbulence_intensity_i: np.ndarray,
        ct_i: np.ndarray,
        rotor_diameter_i: float,
        *,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        freestream_velocity: np.ndarray,
        wind_veer: float,
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

        # Opposite sign convention in this model
        yaw_i *= -1

        # TODO: connect support for tilt
        tilt = 0.0  # turbine.tilt_angle

        # initial velocity deficits, wake expansion, and Gaussian parameters
        M0, E0, sigma_y0, sigma_z0 = self._initial_wake_expansion(
            tilt, yaw_i, ct_i, freestream_velocity, rotor_diameter_i, wind_veer
        )

        # wake expansion parameters
        ky = self.ka * turbulence_intensity_i + self.kb
        kz = self.ka * turbulence_intensity_i + self.kb

        # length of near wake
        x0 = (
            rotor_diameter_i
            * (cosd(yaw_i) * (1 + np.sqrt(1 - ct_i * cosd(yaw_i))))
            / (np.sqrt(2) * (4 * self.alpha * turbulence_intensity_i + 2 * self.beta * (1 - np.sqrt(1 - ct_i))))
            + x_i
        )

        yR = y - y_i  # noqa: F841
        xR = x_i  # yR * tand(yaw) + x_i

        # yaw parameters (skew angle and distance from centerline)
        # skew angle in radians
        theta_c0 = self.dm * (0.3 * np.radians(yaw_i) / cosd(yaw_i)) * (1 - np.sqrt(1 - ct_i * cosd(yaw_i)))
        delta0 = np.tan(theta_c0) * (x0 - x_i)  # initial wake deflection;
        # NOTE: use np.tan here since theta_c0 is radians

        # deflection in the near wake
        delta_near_wake = ((x - xR) / (x0 - xR)) * delta0 + (self.ad + self.bd * (x - x_i))
        delta_near_wake *= (x >= xR) & (x <= x0)

        # deflection in the far wake
        sigma_y = ky * (x - x0) + sigma_y0
        sigma_z = kz * (x - x0) + sigma_z0
        sigma_y = sigma_y * (x >= x0) + sigma_y0 * (x < x0)
        sigma_z = sigma_z * (x >= x0) + sigma_z0 * (x < x0)

        M0_sqrt = np.sqrt(M0)
        middle_term = np.sqrt(sigma_y * sigma_z / (sigma_y0 * sigma_z0))
        ln_deltaNum = (1.6 + M0_sqrt) * (1.6 * middle_term - M0_sqrt)  # noqa: F841
        ln_deltaDen = (1.6 - M0_sqrt) * (1.6 * middle_term + M0_sqrt)  # noqa: F841

        middle_term = ne.evaluate(
            "theta_c0 * E0 / 5.2 * sqrt(sigma_y0 * sigma_z0 / (ky * kz * M0)) * log(ln_deltaNum / ln_deltaDen)"
        )
        delta_far_wake = delta0 + middle_term + (self.ad + self.bd * (x - x_i))

        delta_far_wake = delta_far_wake * (x > x0)
        deflection = delta_near_wake + delta_far_wake

        return deflection


# GCH components


def gamma(
    D,
    velocity,
    Uinf,
    Ct,
    scale=1.0,
):
    """
    Vortex circulation strength. Units of XXX TODO

    Args:
        D (float): Rotor diameter of the current turbine
        velocity (np.array(float)): Velocities at the current turbine
        Uinf (float): Free-stream velocity
        Ct (float): Thrust coefficient at the current turbine

    Returns:
        [type]: [description]
    """
    return scale * (np.pi / 8) * D * velocity * Uinf * Ct  # * cosd(yaw)  <- the cos is included in Ct


def _calculate_gamma(HH, D, Uinf, Ct, scale, u_i, aI, TSR, yaw=1, with_scaling=False):
    # TODO: Needs an appropriate name (and with_scaling)
    scale_factor = sind(yaw) * cosd(yaw) if with_scaling else 1

    vel_top = ((HH + D / 2) / HH) ** 0.12 * np.ones((1, 1, 1, 1, 1))
    Gamma_top = scale_factor * gamma(D, vel_top, Uinf, Ct, scale)

    vel_bottom = ((HH - D / 2) / HH) ** 0.12 * np.ones((1, 1, 1, 1, 1))
    Gamma_bottom = -1 * scale_factor * gamma(D, vel_bottom, Uinf, Ct, scale)

    turbine_average_velocity = np.cbrt(np.mean(u_i**3, axis=(3, 4)))[:, :, :, None, None]
    Gamma_wake_rotation = 0.25 * 2 * np.pi * D * (aI - aI**2) * turbine_average_velocity / TSR
    return Gamma_top, Gamma_bottom, Gamma_wake_rotation


def _calculate_vortex(
    z_i, HH, D, yLocs, eps, Gamma, decay=1, which: str | None = None, with_decay: bool = False, ground: bool = False
):
    """Calcuatues the V and W parameters for the top, bottom, or wake rotation, with or without a decay, or at the ground
    boundary positions.

    Args:
        z_i (_type_): _description_
        HH (_type_): _description_
        D (_type_): _description_
        yLocs (_type_): _description_
        eps (_type_): _description_
        Gamma (_type_): _description_
        decay (int, optional): _description_. Defaults to 1.
        which (str | None, optional): _description_. Defaults to None.
        with_decay (bool, optional): _description_. Defaults to False.
        ground (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: Raised if the input to which is not one of "top", "bottom", "rotation".
        ValueError: Raised if the input to `with_decay` is not a `boolean`.
        ValueError: Raised if the input to `ground` is not a `boolean`.

    Returns:
        _type_: _description_
    """
    if which not in ("top", "bottom", "rotation"):
        raise ValueError(
            "Cannot determine which vortex to calculate, please set `which` to one of: 'top', 'bottom', or 'rotation'."
        )

    if not isinstance(with_decay, bool):
        raise ValueError("Inputs to `with_decay` must be a boolean.")

    if not isinstance(ground, bool):
        raise ValueError("Inputs to `ground` must be a boolean.")

    z_mid = HH if which == "rotation" else (HH + (D if which == "top" else -D) / 2)
    z = z_i + (z_mid if ground else -z_mid)  # noqa: F841
    r = ne.evaluate("yLocs**2 + z**2")  # noqa: F841  # TODO: This is in the paper
    # This looks like spanwise decay - it defines the vortex profile in the spanwise directions
    core_shape = ne.evaluate("1 - exp(-r / (eps**2))")  # noqa: F841

    V = ne.evaluate("(Gamma * z) / (2 * pi * r) * core_shape")
    V = V * decay if with_decay else np.mean(V, axis=(3, 4))
    W = ne.evaluate("(-1 * Gamma * yLocs) / (2 * pi * r) * core_shape * decay")

    if ground:
        return -1 * V, -1 * W
    return V, W


def wake_added_yaw(
    u_i,
    v_i,
    u_initial,
    delta_y,
    z_i,
    rotor_diameter,
    hub_height,
    ct_i,
    tip_speed_ratio,
    axial_induction_i,
    scale=1.0,
):
    """
    what yaw angle would have produced that same average spanwise velocity

    These calculations focus around the current turbine. The formulation could
    remove the dimension for n-turbines, but for consistency with other
    similar equations it is left. However, the turbine dimension should
    always have length 1.
    """

    # turbine parameters
    D = rotor_diameter  # scalar
    HH = hub_height  # scalar
    Ct = ct_i  # (wd, ws, 1, 1, 1) for the current turbine
    TSR = tip_speed_ratio  # scalar
    aI = axial_induction_i  # (wd, ws, 1, 1, 1) for the current turbine
    avg_v = np.mean(v_i, axis=(3, 4))  # (wd, ws, 1, grid, grid)

    # flow parameters
    Uinf = np.mean(u_initial, axis=(2, 3, 4))
    Uinf = Uinf[:, :, None, None, None]

    # TODO: Allow user input for eps gain
    eps_gain = 0.2
    eps = eps_gain * D  # Use set value

    Gamma_top, Gamma_bottom, Gamma_wake_rotation = _calculate_gamma(HH, D, Uinf, Ct, scale, u_i, aI, TSR)

    # compute the spanwise and vertical velocities induced by yaw

    # decay = eps ** 2 / (4 * nu * delta_x / Uinf + eps ** 2)   # This is the decay downstream
    yLocs = delta_y + BaseModel.NUM_EPS

    # top vortex
    v_top, _ = _calculate_vortex(z_i, HH, D, yLocs, eps, Gamma_top, which="top")

    # bottom vortex
    v_bottom, _ = _calculate_vortex(z_i, HH, D, yLocs, eps, Gamma_bottom, which="bottom")

    # wake rotation vortex
    v_core, _ = _calculate_vortex(z_i, HH, D, yLocs, eps, Gamma_wake_rotation, which="rotation")

    # Cap the effective yaw values between -45 and 45 degrees
    val = 2 * (avg_v - v_core) / (v_top + v_bottom)
    val = np.where(val < -1.0, -1.0, val)
    val = np.where(val > 1.0, 1.0, val)
    y = np.degrees(0.5 * np.arcsin(val))

    return y[:, :, :, None, None]


def calculate_transverse_velocity(
    u_i,
    u_initial,
    dudz_initial,
    delta_x,
    delta_y,
    z,
    rotor_diameter,
    hub_height,
    yaw,
    ct_i,
    tsr_i,
    axial_induction_i,
    scale=1.0,
):
    """
    Calculate transverse velocity components for all downstream turbines
    given the vortices at the current turbine.
    """

    # turbine parameters
    D = rotor_diameter
    HH = hub_height
    Ct = ct_i
    TSR = tsr_i
    aI = axial_induction_i

    # flow parameters
    Uinf = np.mean(u_initial, axis=(2, 3, 4))[:, :, None, None, None]

    eps_gain = 0.2
    eps = eps_gain * D  # Use set value

    # TODO: wind sheer is hard-coded here but should be connected to the input
    Gamma_top, Gamma_bottom, Gamma_wake_rotation = _calculate_gamma(
        HH, D, Uinf, Ct, scale, u_i, aI, TSR, yaw=yaw, with_scaling=True
    )

    # compute the spanwise and vertical velocities induced by yaw

    # decay the vortices as they move downstream - using mixing length
    lmda = D / 8
    kappa = 0.41
    lm = kappa * z / (1 + kappa * z / lmda)
    nu = lm**2 * np.abs(dudz_initial)  # noqa: F841

    decay = ne.evaluate("eps**2 / (4 * nu * delta_x / Uinf + eps**2)")  # This is the decay downstream
    yLocs = delta_y + BaseModel.NUM_EPS

    # top vortex
    V1, W1 = _calculate_vortex(z, HH, D, yLocs, eps, Gamma_top, decay=decay, which="top", with_decay=True)

    # bottom vortex
    V2, W2 = _calculate_vortex(z, HH, D, yLocs, eps, Gamma_bottom, decay=decay, which="bottom", with_decay=True)

    # wake rotation vortex
    V5, W5 = _calculate_vortex(
        z, HH, D, yLocs, eps, Gamma_wake_rotation, decay=decay, which="rotation", with_decay=True
    )

    # Boundary condition - ground mirror vortex

    # top vortex - ground
    V3, W3 = _calculate_vortex(z, HH, D, yLocs, eps, Gamma_top, decay=decay, which="top", with_decay=True, ground=True)

    # bottom vortex - ground
    V4, W4 = _calculate_vortex(
        z, HH, D, yLocs, eps, Gamma_bottom, decay=decay, which="bottom", with_decay=True, ground=True
    )

    # wake rotation vortex - ground effect
    V6, W6 = _calculate_vortex(
        z, HH, D, yLocs, eps, Gamma_wake_rotation, decay=decay, which="rotation", with_decay=True, ground=True
    )

    # total spanwise velocity
    V = V1 + V2 + V3 + V4 + V5 + V6
    W = W1 + W2 + W3 + W4 + W5 + W6

    # no spanwise and vertical velocity upstream of the turbine
    # V[delta_x < -1] = 0.0  # Subtract by 1 to avoid numerical issues on rotation
    # W[delta_x < -1] = 0.0  # Subtract by 1 to avoid numerical issues on rotation
    # TODO Should this be <= ? Shouldn't be adding V and W on the current turbine?
    V[delta_x < 0.0] = 0.0  # Subtract by 1 to avoid numerical issues on rotation
    W[delta_x < 0.0] = 0.0  # Subtract by 1 to avoid numerical issues on rotation

    # TODO: Why would the say W cannot be negative?
    W[W < 0] = 0

    return V, W


def yaw_added_turbulence_mixing(u_i, I_i, v_i, w_i, turb_v_i, turb_w_i):
    # Since turbulence mixing is constant for the turbine,
    # use the left two dimensions only here and expand
    # before returning. Dimensions are (wd, ws).

    I_i = I_i[:, :, 0, 0, 0]

    average_u_i = np.cbrt(np.mean(u_i**3, axis=(2, 3, 4)))

    # Convert ambient turbulence intensity to TKE (eq 24)
    k = (average_u_i * I_i) ** 2 / (2 / 3)

    u_term = np.sqrt(2 * k)
    v_term = np.mean(v_i + turb_v_i, axis=(2, 3, 4))
    w_term = np.mean(w_i + turb_w_i, axis=(2, 3, 4))

    # Compute the new TKE (eq 23)
    k_total = 0.5 * (u_term**2 + v_term**2 + w_term**2)

    # Convert TKE back to TI
    I_total = np.sqrt((2 / 3) * k_total) / average_u_i

    # Remove ambient from total TI leaving only the TI due to mixing
    I_mixing = I_total - I_i

    return I_mixing[:, :, None, None, None]


# def yaw_added_recovery_correction(
#     self, U_local, U, W, x_locations, y_locations, turbine, turbine_coord
# ):
#         """
#         This method corrects the U-component velocities when yaw added recovery
#         is enabled. For more details on how the velocities are changed, see [1].
#         # TODO add reference to 1

#         Args:
#             U_local (np.array): U-component velocities across the flow field.
#             U (np.array): U-component velocity deficits across the flow field.
#             W (np.array): W-component velocity deficits across the flow field.
#             x_locations (np.array): Streamwise locations in wake.
#             y_locations (np.array): Spanwise locations in wake.
#             turbine (:py:class:`floris.simulation.turbine.Turbine`):
#                 Turbine object.
#             turbine_coord (:py:obj:`floris.simulation.turbine_map.TurbineMap.coords`):
#                 Spatial coordinates of wind turbine.

#         Returns:
#             np.array: U-component velocity deficits across the flow field.
#         """
#         # compute the velocity without modification
#         U1 = U_local - U

#         # set dimensions
#         D = turbine.rotor_diameter
#         xLocs = x_locations - turbine_coord.x1
#         ky = self.ka * turbine.turbulence_intensity + self.kb
#         U2 = (np.mean(W) * xLocs) / ((ky * xLocs + D / 2))
#         U_total = U1 + np.nan_to_num(U2)

#         # turn it back into a deficit
#         U = U_local - U_total

#         # zero out anything before the turbine
#         U[x_locations < turbine_coord.x1] = 0

#         return U
