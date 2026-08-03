import numexpr as ne
import numpy as np
from attrs import (
    fields,
)
from numpy import pi

from floris.core import BaseModel
from floris.utilities import cosd, sind


NUM_EPS = fields(BaseModel).NUM_EPS.default

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
    # NOTE the cos commented below is included in Ct
    return scale * (pi / 8) * D * velocity * Uinf * Ct # * cosd(yaw)


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
    wind_shear,
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
    D = rotor_diameter              # scalar
    HH = hub_height                 # scalar
    Ct = ct_i                       # (findex, 1, 1, 1) for the current turbine
    TSR = tip_speed_ratio           # scalar
    aI = axial_induction_i          # (findex, 1, 1, 1) for the current turbine
    avg_v = np.mean(v_i, axis=(2,3))  # (findex, 1, grid, grid)

    # flow parameters
    Uinf = np.mean(u_initial, axis=(1, 2, 3))
    Uinf = Uinf[:, None, None, None]

    # TODO: Allow user input for eps gain
    eps_gain = 0.2
    eps = eps_gain * D  # Use set value

    vel_top = ((HH + D / 2) / HH) ** wind_shear * np.ones((1, 1, 1, 1))
    Gamma_top = gamma(
        D,
        vel_top,
        Uinf,
        Ct,
        scale,
    )

    vel_bottom = ((HH - D / 2) / HH) ** wind_shear * np.ones((1, 1, 1, 1))
    Gamma_bottom = -1 * gamma(
        D,
        vel_bottom,
        Uinf,
        Ct,
        scale,
    )

    turbine_average_velocity = np.cbrt(np.mean(u_i ** 3, axis=(2, 3), keepdims=True))
    Gamma_wake_rotation = 0.25 * 2 * pi * D * (aI - aI ** 2) * turbine_average_velocity / TSR

    ### compute the spanwise and vertical velocities induced by yaw

    # decay = eps ** 2 / (4 * nu * delta_x / Uinf + eps ** 2)   # This is the decay downstream
    yLocs = delta_y + NUM_EPS

    # top vortex
    # NOTE: this is the top of the grid, not the top of the rotor
    zT = z_i - (HH + D / 2) + NUM_EPS  # distance from the top of the grid
    # NOTE: This is (-) in the paper, but (+) is consistent with the
    # Martínez-Tossas et al. (2019) source.
    rT_squared = ne.evaluate("yLocs ** 2 + zT ** 2")
    # This looks like spanwise decay;
    # it defines the vortex profile in the spanwise directions
    core_shape = ne.evaluate("1 - exp(-rT_squared / (eps ** 2))")
    v_top = ne.evaluate("(Gamma_top * zT) / (2 * pi * rT_squared) * core_shape")
    v_top = np.mean( v_top, axis=(2,3) )
    # w_top = (-1 * Gamma_top * yLocs) / (2 * pi * rT) * core_shape * decay

    # bottom vortex
    zB = z_i - (HH - D / 2) + NUM_EPS
    rB_squared = ne.evaluate("yLocs ** 2 + zB ** 2")
    core_shape = ne.evaluate("1 - exp(-rB_squared / (eps ** 2))")
    v_bottom = ne.evaluate("(Gamma_bottom * zB) / (2 * pi * rB_squared) * core_shape")
    v_bottom = np.mean( v_bottom, axis=(2,3) )
    # w_bottom = (-1 * Gamma_bottom * yLocs) / (2 * pi * rB) * core_shape * decay

    # wake rotation vortex
    zC = z_i - HH + NUM_EPS
    rC_squared = ne.evaluate("yLocs ** 2 + zC ** 2")
    core_shape = ne.evaluate("1 - exp(-rC_squared / (eps ** 2))")
    v_core = ne.evaluate("(Gamma_wake_rotation * zC) / (2 * pi * rC_squared) * core_shape")
    v_core = np.mean( v_core, axis=(2,3) )
    # w_core = (-1 * Gamma_wake_rotation * yLocs) / (2 * pi * rC_squared) * core_shape * decay

    # Cap the effective yaw values between -45 and 45 degrees
    val = 2 * (avg_v - v_core) / (v_top + v_bottom)
    val = np.where(val < -1.0, -1.0, val)
    val = np.where(val > 1.0, 1.0, val)
    y = np.degrees(0.5 * np.arcsin(val))

    return y[:, :, None, None]

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
    wind_shear,
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
    Uinf = np.mean(u_initial, axis=(1, 2, 3))
    Uinf = Uinf[:, None, None, None]

    eps_gain = 0.2
    eps = eps_gain * D  # Use set value

    vel_top = ((HH + D / 2) / HH) ** wind_shear * np.ones((1, 1, 1, 1))
    Gamma_top = sind(yaw) * cosd(yaw) * gamma(
        D,
        vel_top,
        Uinf,
        Ct,
        scale,
    )

    vel_bottom = ((HH - D / 2) / HH) ** wind_shear * np.ones((1, 1, 1, 1))
    Gamma_bottom = -1 * sind(yaw) * cosd(yaw) * gamma(
        D,
        vel_bottom,
        Uinf,
        Ct,
        scale,
    )
    turbine_average_velocity = np.cbrt(np.mean(u_i ** 3, axis=(2,3), keepdims=True))
    Gamma_wake_rotation = 0.25 * 2 * pi * D * (aI - aI ** 2) * turbine_average_velocity / TSR

    ### compute the spanwise and vertical velocities induced by yaw

    # decay the vortices as they move downstream - using mixing length
    lmda = D / 8
    kappa = 0.41
    lm = kappa * z / (1 + kappa * z / lmda)
    nu = lm ** 2 * np.abs(dudz_initial)

    # This is the decay downstream
    decay = ne.evaluate("eps ** 2 / (4 * nu * delta_x / Uinf + eps ** 2)")
    yLocs = delta_y + NUM_EPS

    # top vortex
    zT = z - (HH + D / 2) + NUM_EPS
    # NOTE: This is (-) in the paper, but (+) is consistent with the
    # Martínez-Tossas et al. (2019) source.
    rT_squared = ne.evaluate("yLocs ** 2 + zT ** 2")
    # This looks like spanwise decay;
    # it defines the vortex profile in the spanwise directions
    core_shape = ne.evaluate("1 - exp(-rT_squared / (eps ** 2))")
    V1 = ne.evaluate("(Gamma_top * zT) / (2 * pi * rT_squared) * core_shape * decay")
    W1 = ne.evaluate("(-1 * Gamma_top * yLocs) / (2 * pi * rT_squared) * core_shape * decay")

    # bottom vortex
    zB = z - (HH - D / 2) + NUM_EPS
    rB_squared = ne.evaluate("yLocs ** 2 + zB ** 2")
    core_shape = ne.evaluate("1 - exp(-rB_squared / (eps ** 2))")
    V2 = ne.evaluate("(Gamma_bottom * zB) / (2 * pi * rB_squared) * core_shape * decay")
    W2 = ne.evaluate("(-1 * Gamma_bottom * yLocs) / (2 * pi * rB_squared) * core_shape * decay")

    # wake rotation vortex
    zC = z - HH + NUM_EPS
    rC_squared = ne.evaluate("yLocs ** 2 + zC ** 2")
    core_shape = ne.evaluate("1 - exp(-rC_squared / (eps ** 2))")
    V5 = ne.evaluate("(Gamma_wake_rotation * zC) / (2 * pi * rC_squared) * core_shape * decay")
    W5 = ne.evaluate(
        "(-1 * Gamma_wake_rotation * yLocs) / (2 * pi * rC_squared) * core_shape * decay"
    )

    ### Boundary condition - ground mirror vortex

    # top vortex - ground
    zTb = z + (HH + D / 2) + NUM_EPS
    rTb_squared = ne.evaluate("yLocs ** 2 + zTb ** 2")
    # This looks like spanwise decay;
    # it defines the vortex profile in the spanwise directions
    core_shape = ne.evaluate("1 - exp(-rTb_squared / (eps ** 2))")
    V3 = ne.evaluate("(-1 * Gamma_top * zTb) / (2 * pi * rTb_squared) * core_shape * decay")
    W3 = ne.evaluate("(Gamma_top * yLocs) / (2 * pi * rTb_squared) * core_shape * decay")

    # bottom vortex - ground
    zBb = z + (HH - D / 2) + NUM_EPS
    rBb_squared = ne.evaluate("yLocs ** 2 + zBb ** 2")
    core_shape = ne.evaluate("1 - exp(-rBb_squared / (eps ** 2))")
    V4 = ne.evaluate("(-1 * Gamma_bottom * zBb) / (2 * pi * rBb_squared) * core_shape * decay")
    W4 = ne.evaluate("(Gamma_bottom * yLocs) / (2 * pi * rBb_squared) * core_shape * decay")

    # wake rotation vortex - ground effect
    zCb = z + HH + NUM_EPS
    rCb_squared = ne.evaluate("yLocs ** 2 + zCb ** 2")
    core_shape = ne.evaluate("1 - exp(-rCb_squared / (eps ** 2))")
    V6 = ne.evaluate(
        "(-1 * Gamma_wake_rotation * zCb) / (2 * pi * rCb_squared) * core_shape * decay"
    )
    W6 = ne.evaluate(
        "(Gamma_wake_rotation * yLocs) / (2 * pi * rCb_squared) * core_shape * decay"
    )

    # total spanwise velocity
    V = V1 + V2 + V3 + V4 + V5 + V6
    W = W1 + W2 + W3 + W4 + W5 + W6

    # No spanwise and vertical velocity upstream of the turbine
    ### Original v3 implementation
    # V[delta_x < -1] = 0.0  # Subtract by 1 to avoid numerical issues on rotation
    # W[delta_x < -1] = 0.0  # Subtract by 1 to avoid numerical issues on rotation
    # TODO Should this be <= ? Shouldn't be adding V and W on the current turbine?
    ### Then we changed it to this
    # V[delta_x < 0.0] = 0.0  # Subtract by 1 to avoid numerical issues on rotation
    # W[delta_x < 0.0] = 0.0  # Subtract by 1 to avoid numerical issues on rotation
    ### Currently, here
    V = np.where(delta_x >= 0.0, V, 0.0)
    W = np.where(delta_x >= 0.0, W, 0.0)

    # TODO: Why would the say W cannot be negative?
    W = np.where(W >= 0, W, 0.0)

    return V, W

def yaw_added_turbulence_mixing(
    u_i,
    I_i,
    v_i,
    w_i,
    turb_v_i,
    turb_w_i
):
    # Since turbulence mixing is constant for the turbine,
    # use the left two dimensions only here and expand
    # before returning. Dimensions are (wd, ws).

    I_i = I_i[:, 0, 0, 0]

    average_u_i = np.cbrt(np.mean(u_i ** 3, axis=(1, 2, 3)))

    # Convert ambient turbulence intensity to TKE (eq 24)
    k = (average_u_i * I_i) ** 2 / (2 / 3)

    u_term = np.sqrt(2 * k)
    v_term = np.mean(v_i + turb_v_i, axis=(1, 2, 3))
    w_term = np.mean(w_i + turb_w_i, axis=(1, 2, 3))

    # Compute the new TKE (eq 23)
    k_total = 0.5 * (u_term ** 2 + v_term ** 2 + w_term ** 2)

    # Convert TKE back to TI
    I_total = np.sqrt((2 / 3) * k_total) / average_u_i

    # Remove ambient from total TI leaving only the TI due to mixing
    I_mixing = I_total - I_i

    return I_mixing[:, None, None, None]
