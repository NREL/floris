import logging
from typing import Any, Dict

import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np
from attrs import define, field
from scipy.integrate import solve_ivp

from floris.core import (
    BaseModel,
    Farm,
    FlowField,
    Grid,
    Turbine,
)
from floris.utilities import (
    cosd,
    sind,
    tand,
)


logger = logging.getLogger(name="floris")

@define
class EddyViscosityVelocity(BaseModel):

    k_l: float = field(default=0.015*np.sqrt(3.56))
    k_a: float = field(default=0.5)
    von_Karman_constant: float = field(default=0.41)

    i_const_1: float = field(default=0.05)
    i_const_2: float = field(default=16)
    i_const_3: float = field(default=0.5)
    i_const_4: float = field(default=10)

    # Below are likely not needed [or, I'll need to think more about it]
    filter_const_1: float = field(default=0.65)
    filter_const_2: float = field(default=4.5)
    filter_const_3: float = field(default=23.32)
    filter_const_4: float = field(default=1/3)
    filter_cutoff_x_: float = field(default=0.0)

    c_0: float = field(default=2.0)
    c_1: float = field(default=1.5)

    wd_std: float = field(default=3.0) # Also try with 0.0 for no meandering

    def prepare_function(
        self,
        grid: Grid,
        flow_field: FlowField,
    ) -> Dict[str, Any]:

        kwargs = {
            "x": grid.x_sorted,
            "y": grid.y_sorted,
            "z": grid.z_sorted,
            "u_initial": flow_field.u_initial_sorted,
            "wind_veer": flow_field.wind_veer
        }
        return kwargs

    # @profile
    def function(
        self,
        x_i: np.ndarray,
        turbulence_intensity_i: np.ndarray,
        ct_i: np.ndarray,
        hub_height_i: float,
        rotor_diameter_i: np.ndarray,
        # enforces the use of the below as keyword arguments and adherence to the
        # unpacking of the results from prepare_function()
        *,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        u_initial: np.ndarray,
        wind_veer: np.ndarray,
    ) -> np.ndarray:

        # Non-dimensionalize and center distances
        x_tilde = ((x.mean(axis=(2,3)) - x_i) / rotor_diameter_i)

        # Compute centerline velocities
        # TODO: This is using an "updated" TI. Is that appropriate?
        U_tilde_c_initial = initial_centerline_velocity(
            ct_i,
            turbulence_intensity_i,
            self.i_const_1,
            self.i_const_2,
            self.i_const_3,
            self.i_const_4
        )

        # Solve ODE to find centerline velocities at each x
        U_tilde_c = np.zeros_like(x_tilde)
        for findex in range(x_tilde.shape[0]):
            x_tilde_unique, unique_ind = np.unique(x_tilde[findex, :], return_inverse=True)
            sorting_indices = np.argsort(x_tilde_unique)
            x_tilde_sorted = x_tilde_unique[sorting_indices]
            valid_indices = x_tilde_sorted >= 2
            x_tilde_eval = x_tilde_sorted[valid_indices]
            if len(x_tilde_eval) == 0: # No downstream locations to fill
                U_tilde_c[findex, :] = U_tilde_c_initial[findex]
                continue
            sol = solve_ivp(
                fun=centerline_ode,
                t_span=[2, x_tilde_eval[-1]],
                y0=U_tilde_c_initial[findex,:],
                method='RK45',
                t_eval=x_tilde_eval,
                args=(
                    turbulence_intensity_i[findex,0],
                    ct_i[findex,0],
                    hub_height_i[findex,0],
                    rotor_diameter_i[findex,0],
                    self.k_a,
                    self.k_l,
                    self.von_Karman_constant
                )
            )

            if sol.status == -1:
                raise RuntimeError(
                    f"Eddy viscosity ODE solver failed to converge for findex={findex}.\n"
                    + "t_span: " + np.array2string(np.array([2, x_tilde_eval[-1]])) + "\n"
                    + "y0: " + np.array2string(U_tilde_c_initial[findex,:]) + "\n"
                    + "t_eval: " + np.array2string(x_tilde_eval) + "\n\n"
                    + "This may be caused by an initial condition for the ODE that is "
                    + "greater than 1."
                )

            # Extract the solution
            if (sol.t != x_tilde_eval).any():
                raise ValueError("ODE solver did not return requested values")
            U_tilde_c_eval = sol.y.flatten()

            U_tilde_c_fill = np.full_like(
                x_tilde_sorted[x_tilde_sorted < 2],
                U_tilde_c_initial[findex,:]
            )
            U_tilde_c_sorted = np.concatenate((U_tilde_c_fill, U_tilde_c_eval))

            # "Unsort", and broadcast back to shape of x_tilde
            U_tilde_c_unique = U_tilde_c_sorted[np.argsort(sorting_indices)]
            U_tilde_c_findex = U_tilde_c_unique[unique_ind]
            U_tilde_c[findex, :] = U_tilde_c_findex


        # Compute wake width
        w_tilde_sq = wake_width_squared(ct_i, U_tilde_c)

        # Correct for wake meandering
        U_tilde_c_meandering = wake_meandering_centerline_correction(
            U_tilde_c, w_tilde_sq, x_tilde, self.wd_std
        )

        # Recompute wake width
        w_tilde_sq_meandering = wake_width_squared(ct_i, U_tilde_c)

        # Set all upstream values (including current turbine's position) to no wake
        U_tilde_c_meandering[x_tilde < 0.1] = 1
        w_tilde_sq_meandering[x_tilde < 0.1] = 0

        # Return velocities NOT as deficits
        return U_tilde_c_meandering, w_tilde_sq_meandering

    def streamtube_expansion(
        self,
        x_i,
        x_from_i,
        y_from_i,
        z_from_i,
        ct_all,
        axial_induction_i,
        w_tilde_sq_tt,
        rotor_diameter_i,
        *,
        x,
        y,
        z,
        u_initial,
        wind_veer,
    ):
        # Non-dimensionalize and center distances
        x_tilde_points = (x.mean(axis=(2,3)) - x_i) / rotor_diameter_i
        x_tilde = x_from_i / rotor_diameter_i
        y_tilde = y_from_i / rotor_diameter_i
        z_tilde = z_from_i / rotor_diameter_i

        # Compute wake width
        e_tilde = wake_width_streamtube_correction_term(
            axial_induction_i,
            x_tilde_points,
            x_tilde,
            y_tilde,
            z_tilde,
            self.c_0,
            self.c_1
        )

        w_tilde_sq_tt = expanded_wake_width_squared(w_tilde_sq_tt, e_tilde)

        U_tilde_c_tt = expanded_wake_centerline_velocity(ct_all[:,:,None], w_tilde_sq_tt)

        return U_tilde_c_tt, w_tilde_sq_tt

    def evaluate_velocities(
        self,
        U_tilde_c_tt,
        ct_all,
        rotor_diameters,
        y_turbines,
        z_turbines,
        *,
        x,
        y,
        z,
        u_initial,
        wind_veer,
    ):
        # Non-dimensionalize and center distances
        y_tilde_rel = (
            (y[:,None,:,:,:] - y_turbines[:,:,None,None,None])
            / rotor_diameters[:,:,None,None,None]
        )
        z_tilde_rel = (
            (z[:,None,:,:,:] - z_turbines[:,:,None,None,None])
            / rotor_diameters[:,:,None,None,None]
        )
        # TODO: Check working as expected with correct D, hh being applied 
        # when there are multiple turbine types

        # Compute radial positions
        r_tilde_sq = y_tilde_rel**2 + z_tilde_rel**2

        U_tilde_r_tt = compute_off_center_velocities(
            U_tilde_c_tt,
            ct_all[:,:,None],
            r_tilde_sq
        )

        return U_tilde_r_tt


def compute_off_center_velocities(U_tilde_c, Ct, r_tilde_sq):
    """
    Compute the off-centerline velocities using the eddy viscosity model
    y_, z_ supposed to be defined from the center of the rotor.
    """
    w_tilde_sq = wake_width_squared(Ct, U_tilde_c)
    U_tilde_c_mask = U_tilde_c == 1
    # As long as U_tilde_c is 1, w_tilde_sq won't affect result, but this
    # silences a division by zero warning
    w_tilde_sq[U_tilde_c_mask] = 1
    U_tilde = (
        1
        - (1 - U_tilde_c[:,:,:,None,None])
        * np.exp(-r_tilde_sq/w_tilde_sq[:,:,:,None,None])
    )

    return U_tilde

def wake_width_squared(Ct, U_tilde_c):
    """
    Compute the wake width squared using the eddy viscosity model
    """
    U_tilde_c_mask = U_tilde_c < 1

    w_tilde_sq = np.zeros_like(U_tilde_c)
    Ct = _resize_Ct(Ct, U_tilde_c)
    w_tilde_sq[U_tilde_c_mask] = (
        Ct[U_tilde_c_mask] / (4 * (1 - U_tilde_c[U_tilde_c_mask]) * (1 + U_tilde_c[U_tilde_c_mask]))
    )
    w_tilde_sq.reshape(U_tilde_c.shape)
    return w_tilde_sq

def centerline_ode(x_tilde, U_tilde_c, ambient_ti, Ct, hh, D, k_a, k_l, von_Karman_constant):
    """
    Define the ODE for the centerline velocities
    """

    # Local component, nondimensionalized by U_inf*D (compared to Gunn 2019's K_l)
    K_l_tilde = k_l * np.sqrt(wake_width_squared(Ct, U_tilde_c)) * (1 - U_tilde_c)

    # Ambient component, nondimensionalized by U_inf*D (compared to Gunn 2019's K_a, eq. (9))
    K_a_tilde = k_a * ambient_ti * von_Karman_constant * (hh/D)

    def filter_function(x_tilde):
        """ Identity mapping (assumed by 'F=1') """

        # Following are not used in the current implementation
        # filter_const_1 = 0.65
        # filter_const_2 = 4.5
        # filter_const_3 = 23.32
        # filter_const_4 = 1/3
        # filter_cutoff_x_ = 0.0
        # if x_tilde < filter_cutoff_x_:
        #     return filter_const_1 * ((x_tilde - filter_const_2) / filter_const_3)**filter_const_4
        # else:
        #     return 1
        return 1

    eddy_viscosity_tilde = filter_function(x_tilde)*(K_l_tilde + K_a_tilde)

    dU_tilde_c_dx_tilde = (
        16 * eddy_viscosity_tilde
        * (U_tilde_c**3 - U_tilde_c**2 - U_tilde_c + 1)
        / (U_tilde_c * Ct)
    )

    return [dU_tilde_c_dx_tilde]

def initial_centerline_velocity(Ct, ambient_ti, i_const_1, i_const_2, i_const_3, i_const_4):

    # The below are from Ainslie (1988)
    initial_velocity_deficit = (
        Ct
        - i_const_1
        - (i_const_2 * Ct - i_const_3) * ambient_ti / i_const_4
    )

    if (initial_velocity_deficit < 0).any():
        logger.warning(
            "Initial velocity deficit is negative. Setting to 0."
        )
        initial_velocity_deficit[initial_velocity_deficit < 0] = 0

    U_c0_ = 1 - initial_velocity_deficit

    return U_c0_

def wake_meandering_centerline_correction(U_tilde_c, w_tilde_sq, x_tilde, wd_std):
    wd_std_rad = np.deg2rad(wd_std)

    m = np.sqrt(1 + 2*wd_std_rad**2 * x_tilde**2/w_tilde_sq)

    U_tilde_c_meandering = 1/m * U_tilde_c + (m-1)/m

    return U_tilde_c_meandering


def wake_width_streamtube_correction_term(ai_i, x_pts, x_ji_, y_ji_, z_ji_, c_0, c_1):
    e_i_ = np.sqrt(1-ai_i) * (1/np.sqrt(1-2*ai_i) - 1)

    e_ji_ = c_0 * e_i_ * np.exp(-(y_ji_**2 + z_ji_**2) / c_1**2)
    e_ji_[x_ji_ >= 0] = 0 # Does not affect wakes of self or downstream turbines
    downstream_mask = (x_pts > 0).astype(int)
    e_ji_ = e_ji_[:,:,None] * downstream_mask[:,None,:] # Affects only downstream locations

    return e_ji_

def expanded_wake_width_squared(w_tilde_sq, e_tilde):
    return (np.sqrt(w_tilde_sq) + e_tilde)**2

def expanded_wake_centerline_velocity(Ct, w_tilde_sq):
    w_tilde_sq_mask = w_tilde_sq > 0
    expanded_U_tilde_c = np.ones_like(w_tilde_sq)
    Ct = _resize_Ct(Ct, w_tilde_sq)
    expanded_U_tilde_c[w_tilde_sq_mask] = np.sqrt(
        1 - Ct[w_tilde_sq_mask]/(4*w_tilde_sq[w_tilde_sq_mask])
    )
    expanded_U_tilde_c.reshape(w_tilde_sq.shape)

    return expanded_U_tilde_c

def _resize_Ct(Ct, resize_like):
    if type(Ct) == np.ndarray:
        Ct = np.repeat(Ct, resize_like.shape[-1], axis=-1)
    else:
        Ct = Ct * np.ones_like(resize_like)
    return Ct
