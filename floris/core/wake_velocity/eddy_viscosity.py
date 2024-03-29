from typing import Any, Dict

import numexpr as ne
import numpy as np
from attrs import define, field

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

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


@define
class EddyViscosityVelocity(BaseModel):

    k_l: float = field(default=0.015*np.sqrt(3.56))
    k_a: float = field(default=0.5)
    von_Karman_constant: float = field(default=0.41)

    i_const_1 = 0.05
    i_const_2 = 16
    i_const_3 = 0.5
    i_const_4 = 10

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

        # # Solve ODE to find centerline velocities at each x
        # x_tilde_unique, unique_ind = np.unique(x_tilde, return_inverse=True)
        # sorting_indices = np.argsort(x_tilde_unique)
        # x_tilde_sorted = x_tilde_unique[sorting_indices]
        # valid_indices = x_tilde_sorted >= 2
        # x_tilde_eval = x_tilde_sorted[valid_indices]
        # import ipdb; ipdb.set_trace()
        # sol = solve_ivp(
        #     fun=centerline_ode,
        #     t_span=[2, x_tilde_eval[-1]],
        #     y0=[U_tilde_c_initial],
        #     method='RK45',
        #     t_eval=x_tilde_eval,
        #     args=(
        #         turbulence_intensity_i,
        #         ct_i,
        #         hub_height_i,
        #         rotor_diameter_i,
        #         self.k_a,
        #         self.k_l,
        #         self.von_Karman_constant
        #     )
        # )

        # # Extract the solution
        # if (sol.t != x_tilde_eval).any():
        #     raise ValueError("ODE solver did not return requested values")
        # U_tilde_c_eval = sol.y.flatten()
        
        # U_tilde_c_fill = np.full_like(x_tilde_sorted[x_tilde_sorted < 2], U_tilde_c_initial)
        # # TODO: I think concatenation will be along axis=1 finally
        # U_tilde_c_sorted = np.concatenate((U_tilde_c_fill, U_tilde_c_eval))

        # # "Unsort", and broadcast back to shape of x_tilde
        # U_tilde_c_unique = U_tilde_c_sorted[np.argsort(sorting_indices)]
        # U_tilde_c = U_tilde_c_unique[unique_ind]

        # Solve ODE to find centerline velocities at each x
        U_tilde_c = np.zeros_like(x_tilde)
        for findex in range(x_tilde.shape[0]):
            x_tilde_unique, unique_ind = np.unique(x_tilde[findex, :], return_inverse=True)
            sorting_indices = np.argsort(x_tilde_unique)
            x_tilde_sorted = x_tilde_unique[sorting_indices]
            valid_indices = x_tilde_sorted >= 2
            x_tilde_eval = x_tilde_sorted[valid_indices]
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

            # Extract the solution
            if (sol.t != x_tilde_eval).any():
                raise ValueError("ODE solver did not return requested values")
            U_tilde_c_eval = sol.y.flatten()
            
            U_tilde_c_fill = np.full_like(
                x_tilde_sorted[x_tilde_sorted < 2],
                U_tilde_c_initial[findex,:]
            )
            # TODO: I think concatenation will be along axis=1 finally
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


        # # Compute off-center velocities
        # U_tilde = compute_off_center_velocities(U_tilde_c_meandering, ct_i, y_tilde, z_tilde)

        # # Set all upstream values to one
        # U_tilde[x_tilde < 0] = 1 # Upstream

        # # Return velocities NOT as deficits
        return U_tilde_c_meandering, w_tilde_sq_meandering
    
    def streamtube_expansion(
        self,
        x_i,
        y_i,
        z_i,
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
        x_tilde = (x.mean(axis=(2,3)) - x_i) / rotor_diameter_i
        y_tilde = (y.mean(axis=(2,3)) - y_i) / rotor_diameter_i
        z_tilde = (z.mean(axis=(2,3)) - z_i) / rotor_diameter_i

        # Compute wake width
        e_tilde = wake_width_streamtube_correction_term(
            axial_induction_i,
            x_tilde,
            y_tilde,
            z_tilde,
            self.c_0,
            self.c_1
        )

        w_tilde_sq_tt = expanded_wake_width_squared(w_tilde_sq_tt, e_tilde)
        
        # Wait we don't need U_tilde_c_tt as an input? w is enough? Interesting, but OK.
        import ipdb; ipdb.set_trace()
        U_tilde_c_tt = expanded_wake_centerline_velocity(Ct, w_tilde_sq_tt)

        return U_tilde_c_tt, w_tilde_sq_tt


def compute_off_center_velocities(U_tilde_c, Ct, y_tilde, z_tilde):
    """
    Compute the off-centerline velocities using the eddy viscosity model
    y_, z_ supposed to be defined from the center of the rotor.
    """
    w_tilde_sq = wake_width_squared(Ct, U_tilde_c)
    U_tilde = 1 - (1 - U_tilde_c) * np.exp(-(y_tilde**2 + z_tilde**2)/w_tilde_sq)
    return U_tilde

def wake_width_squared(Ct, U_tilde_c):
    """
    Compute the wake width squared using the eddy viscosity model
    """
    return Ct / (4*(1-U_tilde_c)*(1+U_tilde_c))

def centerline_ode(x_tilde, U_tilde_c, ambient_ti, Ct, hh, D, k_a, k_l, von_Karman_constant):
    """
    Define the ODE for the centerline velocities
    """
    # Define constants (will later define these as class attribtues)

    # Local component, nondimensionalized by U_inf*D (compared to Gunn 2019's K_l)
    K_l_tilde = k_l * np.sqrt(wake_width_squared(Ct, U_tilde_c)) * (1 - U_tilde_c)

    # Ambient component, nondimensionalized by U_inf*D (compared to Gunn 2019's K_a, eq. (9))
    K_a_tilde = k_a * ambient_ti * von_Karman_constant * (hh/D)

    def filter_function(x_tilde):
        """ Identity mapping (assumed by 'F=1') """

        # Wait, is this just a multiplier? Seems to be?
        filter_const_1 = 0.65
        filter_const_2 = 4.5
        filter_const_3 = 23.32
        filter_const_4 = 1/3
        filter_cutoff_x_ = 0.0 # 5.5 doesn't seem to work; F negative, not good for EV model
        if x_tilde < filter_cutoff_x_: # How should this work? Is this smooth??
            return filter_const_1 * ((x_tilde - filter_const_2) / filter_const_3)**filter_const_4
        else:
            return 1

    #eddy_viscosity = filter_function(K_l + K_a)
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

    U_c0_ = 1 - initial_velocity_deficit

    return U_c0_

def wake_meandering_centerline_correction(U_tilde_c, w_tilde_sq, x_tilde, wd_std):
    wd_std_rad = np.deg2rad(wd_std)

    m = np.sqrt(1 + 2*wd_std_rad**2 * x_tilde**2/w_tilde_sq)

    U_tilde_c_meandering = 1/m * U_tilde_c + (m-1)/m

    return U_tilde_c_meandering


def wake_width_streamtube_correction_term(ai_j, x_ij_, y_ij_, z_ij_, c_0, c_1):
    e_j_ = np.sqrt(1-ai_j) * (1/np.sqrt(1-2*ai_j) - 1)

    # TODO: consider effect of different z also
    if (z_ij_ != 0).any():
        raise NotImplementedError("Only 2D for now")
    e_ij_ = c_0 * e_j_ * np.exp(-y_ij_**2 / c_1**2)

    # Expand and mask to only downstream locations for upstream turbines' wakes
    e_ij_ = np.repeat(e_ij_[:,:,None], e_ij_.shape[1], axis=2)
    e_ij_ = e_ij_ * np.triu(np.ones_like(e_ij_), k=2)

    return e_ij_

def expanded_wake_width_squared(w_tilde_sq, e_tilde):
    return (np.sqrt(w_tilde_sq) + e_tilde)**2

def expanded_wake_centerline_velocity(Ct, w_tilde_sq):

    return np.sqrt(1-Ct/(4*w_tilde_sq))


if __name__ == "__main__":

    plot_offcenter_velocities = True

    # Test inputs
    Ct = 0.8
    hh = 90.0
    D = 126.0
    ambient_ti = 0.06
    U_inf = 8.0

    EVDM = EddyViscosityVelocity()

    x_test = np.linspace(0*D, 20*D, 100)
    y_test = np.linspace(-2*D, 2*D, 9)
    x_test_m, y_test_m = np.meshgrid(x_test, y_test)
    x_test_m = x_test_m.flatten()
    y_test_m = y_test_m.flatten()
    vel_def = EVDM.function(
        x_i=0,
        y_i=0,
        z_i=hh,
        axial_induction_i=None,
        deflection_field_i=None,
        yaw_angle_i=None,
        turbulence_intensity_i=ambient_ti,
        ct_i=Ct,
        hub_height_i=hh,
        rotor_diameter_i=D,
        x=x_test_m,
        y=y_test_m,
        z=hh*np.ones_like(x_test_m),
        u_initial=None,
        wind_veer=None,
    )
    U_tilde = 1 - vel_def

    U_tilde_shaped = U_tilde.reshape((9, 100))

    fig, ax = plt.subplots(2,2)
    if plot_offcenter_velocities:
        for i in range(9):
            alpha = (3*D-abs(y_test[i]))/(3*D)
            ax[0,0].plot(x_test/D, U_tilde_shaped[i,:], color="lightgray", alpha=alpha)
    ax[0,0].plot(x_test/D, U_tilde_shaped[4,:], color="C0")
    ax[0,0].set_xlabel("x_tilde")
    ax[0,0].set_ylabel("U_c_tilde")
    ax[0,0].set_xlim([0, 20])
    ax[0,0].grid()

    if plot_offcenter_velocities:
        for i in range(9):
            alpha = (3*D-abs(y_test[i]))/(3*D)
            ax[0,1].plot(x_test, U_tilde_shaped[i,:]*U_inf, color="lightgray", alpha=alpha)
    ax[0,1].plot(x_test, U_tilde_shaped[4,:]*U_inf, color="C0")
    ax[0,1].plot([0, 20*D], [U_inf, U_inf], linestyle="dotted", color="black")
    ax[0,1].set_xlabel("x [m]")
    ax[0,1].set_ylabel("U_c [m/s]")
    ax[0,1].set_xlim([0, 20*D])
    ax[0,1].grid()

    ax[1,0].plot(x_test/D, np.sqrt(wake_width_squared(Ct, U_tilde_shaped[4,:])), color="C1")
    ax[1,0].set_xlabel("x_tilde")
    ax[1,0].set_ylabel("w_tilde")
    ax[1,0].set_xlim([0, 20])
    ax[1,0].grid()

    ax[1,1].plot(x_test, np.sqrt(wake_width_squared(Ct, U_tilde_shaped[4,:]))*D, color="C1")
    ax[1,1].set_xlabel("x [m]")
    ax[1,1].set_ylabel("w [m]")
    ax[1,1].set_xlim([0, 20*D])
    ax[1,1].grid()

    plt.show()
