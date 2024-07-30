import matplotlib.pyplot as plt
import numpy as np
from eddy_viscosity import (
    compute_centerline_velocities,
    compute_off_center_velocities,
    wake_width_squared,
)


D_T1 = 1
Ct_T1 = 0.8
x_0_T1 = 0

# Suggested settings
D_T2 = 1.52
Ct_T2 = 0.34
x_0_T2 = 1.5

# Retuned settings
Ct_T3 = 0.39
D_T3 = 2.0


hh = 1.0
ambient_ti = 0.06
U_inf = 8.0

x_test_T1 = np.linspace(2*D_T1, 10*D_T1, 100)
x_test_T2 = np.linspace(2*D_T2, 10*D_T2, 100)

U_c_T1, x_out_T1 = compute_centerline_velocities(x_test_T1, U_inf, ambient_ti, Ct_T1, hh, D_T1)
U_c_T2, x_out_T2 = compute_centerline_velocities(x_test_T2, U_inf, ambient_ti, Ct_T2, hh, D_T2)
U_c_T3, x_out_T3 = compute_centerline_velocities(x_test_T2, U_inf, ambient_ti, Ct_T3, hh, D_T3)

fig, ax = plt.subplots(2,1)
ax[0].plot(x_out_T1+x_0_T1, U_c_T1, color="C0", linestyle="solid", label="T1")
ax[0].plot(x_out_T2+x_0_T2, U_c_T2, color="C2", linestyle="dashed", label="T2, suggested")
ax[0].plot(x_out_T3+x_0_T2, U_c_T3, color="C3", linestyle="dashed", label="T2, retuned")
ax[0].set_xlabel("x_ [D]")
ax[0].set_ylabel("U_c_ [-]")
ax[0].grid()
ax[0].set_xlim([0, 12])
ax[0].legend()

w_T1 = np.sqrt(wake_width_squared(Ct_T1, U_c_T1))
w_T2 = np.sqrt(wake_width_squared(Ct_T2, U_c_T2))
w_T3 = np.sqrt(wake_width_squared(Ct_T3, U_c_T3))

ax[1].plot(x_out_T1+x_0_T1, w_T1, color="C0", linestyle="solid", label="T1")
ax[1].plot(x_out_T2+x_0_T2, w_T2*D_T2, color="C2", linestyle="dashed", label="T2, suggested")
ax[1].plot(x_out_T3+x_0_T2, w_T3*D_T3, color="C3", linestyle="dashed", label="T2, retuned")
ax[1].set_xlabel("x_ [D]")
ax[1].set_ylabel("w [m]")
ax[1].grid()
ax[1].set_xlim([0, 12])
#ax[1].legend()

plt.show()
