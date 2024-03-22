import matplotlib.pyplot as plt
import numpy as np

import floris.core.wake_velocity.eddy_viscosity as evwv


"""
Based on https://dx.doi.org/10.1088/1742-6596/1222/1/012003
"""

def wake_width_streamtube_correction_term(ai_j, y_ij_): # Or, just pass in delta (y_i_ already 0, sort of)
    c_0 = 2.0
    c_1 = 1.5

    e_j_ = np.sqrt(1-ai_j) * (1/np.sqrt(1-2*ai_j) - 1)

    # TODO: consider effect of different z also
    e_ij_ = c_0 * e_j_ * np.exp(-y_ij_**2 / c_1**2)

    return e_ij_

def expanded_wake_width_squared(w_sq, e_ij_):
    return (np.sqrt(w_sq) + e_ij_)**2

def expanded_wake_centerline_velocity(Ct, w_sq):

    return np.sqrt(1-Ct/(4*w_sq))

### Possibly the above three will just go into the wake velocity model.
# not quite clear yet.

def combine_wake_velocities(U_v_):
    N = len(U_v_)
    U_comb_ = np.sqrt(1 - N + np.sum(U_v_**2, axis=0))
    if (U_comb_ < 0).any() or np.isnan(U_comb_).any():
        print("uh oh")
        U_comb_[U_comb_ < 0] = 0
        U_comb_[np.isnan(U_comb_)] = 0
    return U_comb_

#def combine_wake_velocities(U_v_, U_inf):
#    U_v = U_v_*U_inf
#    rhs = np.sum(U_inf**2 - U_v**2)
#    return np.sqrt(U_inf**2 - rhs)


if __name__ == "__main__":

    # Test inputs
    Ct = 0.8
    hh = 90.0
    D = 126.0
    ambient_ti = 0.06
    U_inf = 8.0

    # Second turbine's effect on first
    Ct_j = 0.8
    ai_j = 0.5*(1-np.sqrt(1-Ct_j))
    y_ij_ = 0.0 # 0 rotor diameters laterally
    x_ij_ = 5 # 5 rotor diameters downstream

    x_test = np.linspace(2, 20, 100)
    U_c__out, x__out = evwv.compute_centerline_velocities(x_test, U_inf, ambient_ti, Ct, hh, D)
    y_test = np.tile(np.linspace(-2, 2, 9), (100,1))
    z_test = np.zeros_like(y_test)
    U_r__out = evwv.compute_off_center_velocities(U_c__out, y_test, z_test, Ct)
    w_sq = evwv.wake_width_squared(Ct, U_c__out)


    # Correct first turbine wake for second turbine
    e_ij_ = wake_width_streamtube_correction_term(ai_j, y_ij_)
    w_sq_2 = w_sq.copy()
    w_sq_2[x_test >= x_ij_] = expanded_wake_width_squared(w_sq, e_ij_)[x_test >= x_ij_]
    U_c__out_2 = U_c__out.copy()
    U_c__out_2[x_test >= x_ij_] = expanded_wake_centerline_velocity(Ct, w_sq_2)[x_test >= x_ij_]

    # Compute the centerline velocity of the second wake
    U_c__out_j, x__out_j = evwv.compute_centerline_velocities(
        x_test,
        U_inf,
        ambient_ti,
        Ct_j,
        hh,
        D
    )
    U_c__out_j_2 = np.ones_like(U_c__out_j)
    n_valid = np.sum(x_test >= x_ij_ + x_test[0])
    U_c__out_j_2[x_test >= x_ij_ + x_test[0]] = U_c__out_j[:n_valid]

    U_c__out_comb = combine_wake_velocities(np.stack((U_c__out_2, U_c__out_j_2)))


    fig, ax = plt.subplots(2,2)
    ax[0,0].plot(x__out, U_c__out, color="C0", label="Single turbine center")
    ax[0,0].plot(x__out, U_c__out_2, color="C2", label="Upstream turbine center")
    ax[0,0].plot(x__out, U_c__out_j_2, color="C1",  label="Downstream turbine center")
    ax[0,0].plot(x__out, U_c__out_comb, color="black",  label="Combined center")
    ax[0,0].set_xlabel("x_ [D]")
    ax[0,0].set_ylabel("U_c_ [-]")
    ax[0,0].set_xlim([0, 20])
    ax[0,0].grid()
    ax[0,0].legend()

    ax[0,1].plot(x__out*D, U_c__out*U_inf, color="C0")
    ax[0,1].plot(x__out*D, U_c__out_2*U_inf, color="C2")
    ax[0,1].plot(x__out*D, U_c__out_j_2*U_inf, color="C1")
    ax[0,1].plot(x__out*D, U_c__out_comb*U_inf, color="black")
    ax[0,1].plot([0, 20*D], [U_inf, U_inf], linestyle="dotted", color="black")
    ax[0,1].set_xlabel("x [m]")
    ax[0,1].set_ylabel("U_c [m/s]")
    ax[0,1].set_xlim([0, 20*D])
    ax[0,1].grid()

    ax[1,0].plot(x__out, np.sqrt(w_sq), color="C0")
    ax[1,0].plot(x__out, np.sqrt(w_sq_2), color="C2")
    ax[1,0].set_xlabel("x_ [D]")
    ax[1,0].set_ylabel("w_ [-]")
    ax[1,0].set_xlim([0, 20])
    ax[1,0].grid()

    ax[1,1].plot(x__out*D, np.sqrt(w_sq)*D, color="C0")
    ax[1,1].plot(x__out*D, np.sqrt(w_sq_2)*D, color="C2")
    ax[1,1].set_xlabel("x [m]")
    ax[1,1].set_ylabel("w [m]")
    ax[1,1].set_xlim([0, 20*D])
    ax[1,1].grid()

    plt.show()
