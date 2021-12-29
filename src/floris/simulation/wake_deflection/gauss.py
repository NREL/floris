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
from floris.simulation import Grid
from floris.simulation import Farm
from floris.simulation import FlowField
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

    model_string = "gauss"

    def prepare_function(
        self,
        grid: Grid,
        farm: Farm,
        flow_field: FlowField
    ) -> Dict[str, Any]:

        reference_rotor_diameter = farm.reference_turbine_diameter

        kwargs = dict(
            x=grid.x,
            y=grid.y,
            z=grid.z,
            freestream_velocity=flow_field.u_initial,
            wind_veer=flow_field.wind_veer,
            reference_rotor_diameter=reference_rotor_diameter,
        )
        return kwargs

    # @profile
    def function(
        self,
        x_i: np.ndarray,
        y_i: np.ndarray,
        yaw_i: np.ndarray,
        turbulence_intensity_i: np.ndarray,
        ct_i: np.ndarray,
        *,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        freestream_velocity: np.ndarray,
        wind_veer: float,
        reference_rotor_diameter: float,
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

        # initial velocity deficits
        uR = (
            freestream_velocity
          * ct_i
          * cosd(tilt)
          * cosd(yaw_i)
          / (2.0 * (1 - np.sqrt(1 - (ct_i * cosd(tilt) * cosd(yaw_i)))))
        )
        u0 = freestream_velocity * np.sqrt(1 - ct_i)

        # length of near wake
        x0 = (
            reference_rotor_diameter
            * (cosd(yaw_i) * (1 + np.sqrt(1 - ct_i * cosd(yaw_i))))
            / (np.sqrt(2) * (4 * self.alpha * turbulence_intensity_i + 2 * self.beta * (1 - np.sqrt(1 - ct_i))))
            + x_i
        )

        # wake expansion parameters
        ky = self.ka * turbulence_intensity_i + self.kb
        kz = self.ka * turbulence_intensity_i + self.kb

        C0 = 1 - u0 / freestream_velocity
        M0 = C0 * (2 - C0)
        E0 = C0 ** 2 - 3 * np.exp(1.0 / 12.0) * C0 + 3 * np.exp(1.0 / 3.0)

        # initial Gaussian wake expansion
        sigma_z0 = reference_rotor_diameter * 0.5 * np.sqrt(uR / (freestream_velocity + u0))
        sigma_y0 = sigma_z0 * cosd(yaw_i) * cosd(wind_veer)

        yR = y - y_i
        xR = x_i # yR * tand(yaw) + x_i

        # yaw parameters (skew angle and distance from centerline)
        # skew angle in radians
        theta_c0 = self.dm * (0.3 * np.radians(yaw_i) / cosd(yaw_i)) * (1 - np.sqrt(1 - ct_i * cosd(yaw_i)))
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


def gamma(
    D,
    velocity,
    Uinf,
    Ct,
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
    scale = 1.0
    # NOTE: Ct includes a cosd(yaw)
    return scale * (np.pi / 8) * D * velocity * Uinf * Ct


# def calculate_effective_yaw(
def wake_added_yaw(
    u_i,
    v_i,
    u_initial,
    delta_x,
    delta_y,
    z_i,
    rotor_diameter,
    hub_height,
    yaw_angle,
    ct_i,
    tip_speed_ratio,
    axial_induction_i
):
    """
    what yaw angle would have produced that same average spanwise velocity

    These calculations focus around the current turbine. The formulation could
    remove the dimension for n-turbines, but for consistency with other
    similar equations it is left. However, the turbine dimension should
    always have length 1.
    """

    # turbine parameters
    D = rotor_diameter              # scalar
    HH = hub_height                 # scalar
    yaw = yaw_angle                 # (wd, ws, 1, 1, 1) for the current turbine
    Ct = ct_i                       # (wd, ws, 1, 1, 1) for the current turbine
    TSR = tip_speed_ratio           # scalar
    aI = axial_induction_i          # (wd, ws, 1, 1, 1) for the current turbine
    avg_v = np.mean(v_i, axis=(3,4))  # (wd, ws, 1, grid, grid)
    avg_v = avg_v[:,:,:,None,None]

    # flow parameters
    # Uinf = np.mean(flow_field.wind_map.grid_wind_speed)
    Uinf = 9.0

    eps_gain = 0.2
    eps = eps_gain * D  # Use set value

    vel_top = np.mean(u_i[:,:,:,:,-1] / Uinf, axis=3)
    vel_top = vel_top[:,:,:,None,None]
    Gamma_top = sind(yaw) * cosd(yaw) * gamma(
        D,
        vel_top,
        Uinf,
        Ct,
    )

    vel_bottom = np.mean(u_i[:,:,:,:,0] / Uinf, axis=3)
    vel_bottom = vel_bottom[:,:,:,None,None]
    Gamma_bottom = -1 * sind(yaw) * cosd(yaw) * gamma(
        D,
        vel_bottom,
        Uinf,
        Ct,
    )

    turbine_average_velocity = np.cbrt(np.mean(u_i ** 3, axis=(3,4)))
    turbine_average_velocity = turbine_average_velocity[:,:,:,None,None]
    Gamma_wake_rotation = 0.25 * 2 * np.pi * D * (aI - aI ** 2) * turbine_average_velocity / TSR

    ### compute the spanwise and vertical velocities induced by yaw

    # decay the vortices as they move downstream - using mixing length
    # lmda = D / 8
    # kappa = 0.41
    # lm = kappa * z_i / (1 + kappa * z_i / lmda)
    # z = np.linspace(np.min(z), np.max(z), np.shape(u_initial)[2])
    # dudz_initial = np.gradient(u_initial, z, axis=4)
    # dudz_initial = np.gradient(u_initial, axis=4)
    # nu = lm ** 2 * np.abs(dudz_initial)
    # nu = lm ** 2 * np.abs(dudz_initial)

    # decay = eps ** 2 / (4 * nu * delta_x / Uinf + eps ** 2)   # This is the decay downstream
    yLocs = delta_y + 0.01

    # top vortex
    # NOTE: this is the top of the grid, not the top of the rotor
    zT = z_i - (HH + D / 2) + 0.01  # distance from the top of the grid
    rT = yLocs ** 2 + zT ** 2  # TODO: This is - in the paper
    core_shape = 1 - np.exp(-rT / (eps ** 2))  # This looks like spanwise decay - it defines the vortex profile in the spanwise directions
    v_top = (Gamma_top * zT) / (2 * np.pi * rT) * core_shape # * decay
    # w_top = (-1 * Gamma_top * yLocs) / (2 * np.pi * rT) * core_shape * decay

    # bottom vortex
    zB = z_i - (HH - D / 2) + 0.01
    rB = yLocs ** 2 + zB ** 2
    core_shape = 1 - np.exp(-rB / (eps ** 2))
    v_bottom = (Gamma_bottom * zB) / (2 * np.pi * rB) * core_shape # * decay    
    # w_bottom = (-1 * Gamma_bottom * yLocs) / (2 * np.pi * rB) * core_shape * decay

    # wake rotation vortex
    zC = z_i - HH + 0.01
    rC = yLocs ** 2 + zC ** 2
    core_shape = 1 - np.exp(-rC / (eps ** 2))
    v_core = (Gamma_wake_rotation * zC) / (2 * np.pi * rC) * core_shape # * decay
    v_core = np.mean(v_core, axis=(3,4))
    v_core = v_core[:,:,:,None,None]
    # w_core = (-1 * Gamma_wake_rotation * yLocs) / (2 * np.pi * rC) * core_shape * decay

    if (avg_v == 0.0).all():
        return np.zeros_like(avg_v)

    """
    By calculating the circulation strength and applying it to velocities here, we are imposing
    the vortices from the current turbine onto itself. However, we really just want to take the
    tranverse velocities in the flow field from the previous turbine to calculate the effective
    velocity here. I've replaced v_top and v_bottom with vel_top and vel_bottom.

    Also, v_core is the wake rotation vortex on the entire rotor area. That probably does make
    sense, but I've used the mean v_core here so that the dimensions work out. Maybe it should
    instead be the full v_core and then use the mean proir to arcsin. Is that equivalent anyway?
    """
    # y = np.degrees( 0.5 * np.arcsin( 2 * (avg_v - v_core) / (v_top + v_bottom) ) )
    y = np.degrees( 0.5 * np.arcsin( 2 * (avg_v - v_core) / (vel_top + vel_bottom) ) )
    return y


def calculate_transverse_velocity(
    u_i,
    u_initial,
    delta_x,
    delta_y,
    z,
    rotor_diameter,
    hub_height,
    yaw,
    ct_i,
    tsr_i,
    axial_induction_i
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
    # Uinf = np.mean(flow_field.wind_map.grid_wind_speed)
    Uinf = 9.0

    eps_gain = 0.2
    eps = eps_gain * D  # Use set value

    vel_top = np.mean(u_i[:,:,:,:,-1] / Uinf, axis=3)
    vel_top = vel_top[:,:,:,None,None]
    Gamma_top = sind(yaw) * cosd(yaw) * gamma(
        D,
        vel_top,
        Uinf,
        Ct,
    )

    vel_bottom = np.mean(u_i[:,:,:,:,0] / Uinf, axis=3)
    vel_bottom = vel_bottom[:,:,:,None,None]
    Gamma_bottom = -1 * sind(yaw) * cosd(yaw) * gamma(
        D,
        vel_bottom,
        Uinf,
        Ct,
    )

    turbine_average_velocity = np.cbrt(np.mean(u_i ** 3, axis=(3,4)))
    turbine_average_velocity = turbine_average_velocity[:,:,:,None,None]
    Gamma_wake_rotation = 0.25 * 2 * np.pi * D * (aI - aI ** 2) * turbine_average_velocity / TSR

    ### compute the spanwise and vertical velocities induced by yaw

    # decay the vortices as they move downstream - using mixing length
    lmda = D / 8
    kappa = 0.41
    lm = kappa * z / (1 + kappa * z / lmda)
    # z = np.linspace(np.min(z), np.max(z), np.shape(u_initial)[2])
    # dudz_initial = np.gradient(u_initial, z, axis=4)
    dudz_initial = np.gradient(u_initial, axis=4)
    # nu = lm ** 2 * np.abs(dudz_initial)
    nu = lm ** 2 * np.abs(dudz_initial)

    decay = eps ** 2 / (4 * nu * delta_x / Uinf + eps ** 2)   # This is the decay downstream
    yLocs = delta_y + 0.01

    # top vortex
    zT = z - (HH + D / 2) + 0.01
    rT = yLocs ** 2 + zT ** 2  # TODO: This is - in the paper
    core_shape = 1 - np.exp(-rT / (eps ** 2))  # This looks like spanwise decay - it defines the vortex profile in the spanwise directions
    V1 = (Gamma_top * zT) / (2 * np.pi * rT) * core_shape * decay
    W1 = (-1 * Gamma_top * yLocs) / (2 * np.pi * rT) * core_shape * decay

    # bottom vortex
    zB = z - (HH - D / 2) + 0.01
    rB = yLocs ** 2 + zB ** 2
    core_shape = 1 - np.exp(-rB / (eps ** 2))
    V2 = (Gamma_bottom * zB) / (2 * np.pi * rB) * core_shape * decay
    W2 = (-1 * Gamma_bottom * yLocs) / (2 * np.pi * rB) * core_shape * decay

    # wake rotation vortex
    zC = z - HH + 0.01
    rC = yLocs ** 2 + zC ** 2
    core_shape = 1 - np.exp(-rC / (eps ** 2))
    V5 = (Gamma_wake_rotation * zC) / (2 * np.pi * rC) * core_shape * decay
    W5 = (-1 * Gamma_wake_rotation * yLocs) / (2 * np.pi * rC) * core_shape * decay


    ### Boundary condition - ground mirror vortex
    yLocs = delta_y + 0.01

    # top vortex - ground
    zTb = z + (HH + D / 2) + 0.01
    rTb = yLocs ** 2 + zTb ** 2
    core_shape = 1 - np.exp(-rTb / (eps ** 2))  # This looks like spanwise decay - it defines the vortex profile in the spanwise directions
    V3 = (-1 * Gamma_top * zTb) / (2 * np.pi * rTb) * core_shape * decay
    W3 = (Gamma_top * -1 * yLocs) / (2 * np.pi * rTb) * core_shape * decay

    # bottom vortex - ground
    zBb = z + (HH - D / 2) + 0.01
    rBb = yLocs ** 2 + zBb ** 2
    V4 = (-1 * Gamma_bottom * zBb) / (2 * np.pi * rBb) * core_shape * decay
    W4 = (Gamma_bottom * -1 * yLocs) / (2 * np.pi * rBb) * core_shape * decay

    # wake rotation vortex - ground effect
    zCb = z + HH + 0.01
    rCb = yLocs ** 2 + zCb ** 2
    V6 = (-1 * Gamma_wake_rotation * zCb) / (2 * np.pi * rCb) * core_shape * decay
    W6 = (Gamma_wake_rotation * -1 * yLocs) / (2 * np.pi * rCb) * core_shape * decay

    # total spanwise velocity
    V = V1 + V2 + V3 + V4 + V5 + V6
    W = W1 + W2 + W3 + W4 + W5 + W6

    # no spanwise and vertical velocity upstream of the turbine
    # V[delta_x < -1] = 0.0  # Subtract by 1 to avoid numerical issues on rotation
    # W[delta_x < -1] = 0.0  # Subtract by 1 to avoid numerical issues on rotation
    V[delta_x <= 0.0] = 0.0  # Subtract by 1 to avoid numerical issues on rotation
    W[delta_x <= 0.0] = 0.0  # Subtract by 1 to avoid numerical issues on rotation

    # TODO: Why would the say W cannot be negative?
    W[W < 0] = 0

    return V, W


def yaw_added_turbulence_mixing(
    u_i,
    I,
    v,
    w,
    turb_v,
    turb_w
):
    average_u_i = np.mean(u_i, axis=(3,4))
    # average_u_i = average_u_i[:,:,:,None,None]
    I = np.mean(I, axis=(3,4))
    # Convert ambient turbulence intensity to TKE (eq 24)
    k = (average_u_i * I) ** 2 / (2 / 3)

    u_term = np.sqrt(2 * k)
    v_term = np.mean(v + turb_v, axis=(3,4))
    # v_term = v_term[:,:,:,None,None]
    w_term = np.mean(w + turb_w, axis=(3,4))
    # w_term = w_term[:,:,:,None,None]

    # Compute the new TKE (eq 23)
    k_total = 0.5 * ( u_term ** 2 + v_term ** 2 + w_term ** 2 )

    # Convert TKE back to TI
    I_total = np.sqrt( (2 / 3) * k_total ) / average_u_i

    # Remove ambient from total TI leaving only the TI due to mixing
    I_mixing = I_total - I

    return I_mixing
