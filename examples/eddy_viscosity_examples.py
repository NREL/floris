import matplotlib.pyplot as plt
import numpy as np

import floris.core.wake_combination.streamtube_expansion as se
import floris.core.wake_velocity.eddy_viscosity as ev


plot_offcenter_velocities = True

# Test inputs
Ct = 0.8
hh = 90.0
D = 126.0
ambient_ti = 0.06
U_inf = 8.0
wd_std = 3.0

x_test = np.linspace(2, 20, 100)
U_c__out, x__out = ev.compute_centerline_velocities(x_test, U_inf, ambient_ti, Ct, hh, D)
y_test = np.tile(np.linspace(-2, 2, 9), (100,1))
z_test = np.zeros_like(y_test)
U_r__out = ev.compute_off_center_velocities(U_c__out, y_test, z_test, Ct)
w_sq_ = ev.wake_width_squared(Ct, U_c__out)


fig, ax = plt.subplots(2,2)
fig.suptitle('Single turbine wake, no meandering', fontsize=16)
for i in range(9):
    alpha = (3-abs(y_test[0,i]))/3
    ax[0,0].plot(x__out, U_r__out[:,i], color="lightgray", alpha=alpha)
ax[0,0].plot(x__out, U_c__out, color="C0")
ax[0,0].set_xlabel("x_ [D]")
ax[0,0].set_ylabel("U_c_ [-]")
ax[0,0].set_xlim([0, 20])
ax[0,0].grid()

for i in range(9):
    alpha = (3-abs(y_test[0,i]))/3
    ax[0,1].plot(x__out*D, U_r__out[:,i]*U_inf, color="lightgray", alpha=alpha)
ax[0,1].plot(x__out*D, U_c__out*U_inf)
ax[0,1].plot([0, 20*D], [U_inf, U_inf], linestyle="dotted", color="black")
ax[0,1].set_xlabel("x [m]")
ax[0,1].set_ylabel("U_c [m/s]")
ax[0,1].set_xlim([0, 20*D])
ax[0,1].grid()

ax[1,0].plot(x__out, np.sqrt(w_sq_), color="C1")
ax[1,0].set_xlabel("x_ [D]")
ax[1,0].set_ylabel("w_ [-]")
ax[1,0].set_xlim([0, 20])
ax[1,0].grid()

ax[1,1].plot(x__out*D, np.sqrt(w_sq_)*D, color="C1")
ax[1,1].set_xlabel("x [m]")
ax[1,1].set_ylabel("w [m]")
ax[1,1].set_xlim([0, 20*D])
ax[1,1].grid()


U_c__out_mc = ev.wake_meandering_centerline_correction(U_c__out, w_sq_, x__out)
w_sq_mc = ev.wake_width_squared(Ct, U_c__out_mc)

fig, ax = plt.subplots(2,2)
fig.suptitle('Single turbine wake, meandering correction', fontsize=12)
for i in range(9):
    alpha = (3-abs(y_test[0,i]))/3
    ax[0,0].plot(x__out, U_r__out[:,i], color="lightgray", alpha=alpha, linestyle="dashed")
ax[0,0].plot(x__out, U_c__out, color="C0", linestyle="dashed", label="Without meandering")
ax[0,0].plot(x__out, U_c__out_mc, color="C0", linestyle="solid", label="With meandering")
ax[0,0].set_xlabel("x_ [D]")
ax[0,0].set_ylabel("U_c_ [-]")
ax[0,0].set_xlim([0, 20])
ax[0,0].grid()
ax[0,0].legend()

for i in range(9):
    alpha = (3-abs(y_test[0,i]))/3
    ax[0,1].plot(x__out*D, U_r__out[:,i]*U_inf, color="lightgray", alpha=alpha, linestyle="dashed")
ax[0,1].plot(x__out*D, U_c__out*U_inf, color="C0", linestyle="dashed")
ax[0,1].plot(x__out*D, U_c__out_mc*U_inf, color="C0", linestyle="solid")
ax[0,1].plot([0, 20*D], [U_inf, U_inf], linestyle="dotted", color="black")
ax[0,1].set_xlabel("x [m]")
ax[0,1].set_ylabel("U_c [m/s]")
ax[0,1].set_xlim([0, 20*D])
ax[0,1].grid()

ax[1,0].plot(x__out, np.sqrt(w_sq_), color="C1", linestyle="dashed")
ax[1,0].plot(x__out, np.sqrt(w_sq_mc), color="C1", linestyle="solid")
ax[1,0].set_xlabel("x_ [D]")
ax[1,0].set_ylabel("w_ [-]")
ax[1,0].set_xlim([0, 20])
ax[1,0].grid()

ax[1,1].plot(x__out*D, np.sqrt(w_sq_)*D, color="C1", linestyle="dashed")
ax[1,1].plot(x__out*D, np.sqrt(w_sq_mc)*D, color="C1", linestyle="solid")
ax[1,1].set_xlabel("x [m]")
ax[1,1].set_ylabel("w [m]")
ax[1,1].set_xlim([0, 20*D])
ax[1,1].grid()


## Look at mutliple turbines

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
