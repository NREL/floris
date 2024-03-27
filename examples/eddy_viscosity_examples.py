import matplotlib.pyplot as plt
import numpy as np

import floris.core.wake_combination.streamtube_expansion as se
from floris.core.wake_velocity.eddy_viscosity import (
    EddyViscosityVelocityDeficit,
    wake_width_squared
)


plot_offcenter_velocities = True

# Test inputs
Ct = 0.8
hh = 90.0
D = 126.0
ambient_ti = 0.06
U_inf = 8.0

EVDM = EddyViscosityVelocityDeficit()

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
for i in range(9):
    alpha = (3*D-abs(y_test[i]))/(3*D)
    ax[0,0].plot(x_test/D, U_tilde_shaped[i,:], color="lightgray", alpha=alpha)
ax[0,0].plot(x_test/D, U_tilde_shaped[4,:], color="C0", label="With meandering")
ax[0,0].set_xlabel(r"$\tilde{x}$")
ax[0,0].set_ylabel(r"$\tilde{U}_c$")
ax[0,0].set_xlim([0, 20])
ax[0,0].grid()


for i in range(9):
    alpha = (3*D-abs(y_test[i]))/(3*D)
    ax[0,1].plot(x_test, U_tilde_shaped[i,:]*U_inf, color="lightgray", alpha=alpha)
ax[0,1].plot(x_test, U_tilde_shaped[4,:]*U_inf, color="C0")
ax[0,1].plot([0, 20*D], [U_inf, U_inf], linestyle="dotted", color="black")
ax[0,1].set_xlabel(r"$x$ [m]")
ax[0,1].set_ylabel(r"$U_c$ [m/s]")
ax[0,1].set_xlim([0, 20*D])
ax[0,1].grid()

ax[1,0].plot(x_test/D, np.sqrt(wake_width_squared(Ct, U_tilde_shaped[4,:])), color="C1")
ax[1,0].set_xlabel(r"$\tilde{x}$")
ax[1,0].set_ylabel(r"$\tilde{w}$")
ax[1,0].set_xlim([0, 20])
ax[1,0].grid()

ax[1,1].plot(x_test, np.sqrt(wake_width_squared(Ct, U_tilde_shaped[4,:]))*D, color="C1")
ax[1,1].set_xlabel(r"$x$ [m]")
ax[1,1].set_ylabel(r"$w$ [m]")
ax[1,1].set_xlim([0, 20*D])
ax[1,1].grid()

# Compute the equivalent centerline velocities without wake meandering
EVDM.wd_std = 0.0
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
ax[0,0].plot(x_test/D, U_tilde_shaped[4,:], color="C0", linestyle="dashed",
             label="Without meandering")
ax[0,1].plot(x_test, U_tilde_shaped[4,:]*U_inf, color="C0", linestyle="dashed")
ax[1,0].plot(x_test/D, np.sqrt(wake_width_squared(Ct, U_tilde_shaped[4,:])), color="C1",
             linestyle="dashed")
ax[1,1].plot(x_test, np.sqrt(wake_width_squared(Ct, U_tilde_shaped[4,:]))*D, color="C1",
             linestyle="dashed")
ax[0,0].legend()


plt.show()

## Look at multiple turbines

# Second turbine's effect on first
Ct_j = 0.8
ai_j = 0.5*(1-np.sqrt(1-Ct_j))
y_ij_ = 0.0 # 0 rotor diameters laterally
x_ij_ = 5 # 5 rotor diameters downstream

x_test = np.linspace(2, 20, 100)
U_c__out, x__out = ev.compute_centerline_velocities(x_test, U_inf, ambient_ti, Ct, hh, D)
y_test = np.tile(np.linspace(-2, 2, 9), (100,1))
z_test = np.zeros_like(y_test)
U_r__out = ev.compute_off_center_velocities(U_c__out, y_test, z_test, Ct)
w_sq = ev.wake_width_squared(Ct, U_c__out)


# Correct first turbine wake for second turbine
e_ij_ = se.wake_width_streamtube_correction_term(ai_j, y_ij_)
w_sq_2 = w_sq.copy()
w_sq_2[x_test >= x_ij_] = se.expanded_wake_width_squared(w_sq, e_ij_)[x_test >= x_ij_]
U_c__out_2 = U_c__out.copy()
U_c__out_2[x_test >= x_ij_] = se.expanded_wake_centerline_velocity(Ct, w_sq_2)[x_test >= x_ij_]

# Compute the centerline velocity of the second wake
U_c__out_j, x__out_j = ev.compute_centerline_velocities(
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

U_c__out_comb = se.combine_wake_velocities(np.stack((U_c__out_2, U_c__out_j_2)))


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
