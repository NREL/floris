"""Example: Reproduce published eddy viscosity results
This example attempts to reproduce the results of Ainslie (1988) and Gunn (2019)
using the FLORIS implementation of the eddy viscosity model.

Links:
- Ainslie (1988): https://doi.org/10.1016/0167-6105(88)90037-2
- Gunn (2019): https://dx.doi.org/10.1088/1742-6596/1222/1/012003
"""

import matplotlib.pyplot as plt
import numpy as np

from floris import FlorisModel
from floris.turbine_library import build_cosine_loss_turbine_dict


# Build a constant CT turbine model for use in comparisons (not realistic)
u_0 = 8.0 # wind speed [m/s]

# Load the EV model
fmodel = FlorisModel("../inputs/ev.yaml")

## First, attempt to reproduce Figs. 3 and 4 from Ainslie (1988).

# It is not clear exactly how the parametrization of Gunn, which is the one
# that is implemented in the EddyViscosityVelocity model, can be matched to
# the parameters given by Ainslie. Rather than try to do this, we simply
# generate plots similar to Figs. 3 and 4. using the default parameters following
# Gunn (2019).

# Generate figure to plot on
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
fig.set_size_inches(8, 4)

HH = 50
D = 50

for i, wd_std_ev in enumerate([0.0, np.rad2deg(0.1)]):
    fmodel.set_param(["wake", "wake_velocity_parameters", "eddy_viscosity", "wd_std_ev"], wd_std_ev)
    for C_T, ls in zip([0.5, 0.7, 0.9], ["-", "--", ":"]):
        const_CT_turb = build_cosine_loss_turbine_dict(
            turbine_data_dict={
                "wind_speed":[0.0, 30.0],
                "power":[0.0, 1.0], # Not realistic but won't be used here
                "thrust_coefficient":[C_T, C_T]
            },
            turbine_name="ConstantCT",
            rotor_diameter=D,
            hub_height=HH,
            ref_tilt=0.0,
        )

        # Load the EV model and set a constant CT turbine
        fmodel.set(
            layout_x=[0],
            layout_y=[0],
            turbine_type=[const_CT_turb],
            wind_speeds=[u_0],
            wind_directions=[270],
            turbulence_intensities=[0.14], # As specified by Ainslie
            wind_shear=0.0,
            reference_wind_height=HH
        )

        points_x = np.linspace(2*D, 10*D, 1000)
        points_y = np.zeros_like(points_x)
        points_z = HH * np.ones_like(points_x)

        u_at_points = fmodel.sample_flow_at_points(points_x, points_y, points_z)

        # Plot results (not different axis scales in Ainslie)
        ax[i].plot(
            points_x/D, 1-u_at_points[0, :]/u_0, color="k", linestyle=ls, label=rf"$C_T$ = {C_T}"
        )
        ax[i].set_title(r"WD std. dev. $\sigma_{\theta} = $"+"{:.1f} deg".format(wd_std_ev))

ax[0].set_xlabel("Downstream distance [D]")
ax[1].set_xlabel("Downstream distance [D]")
ax[0].set_ylabel("Centerline velocity deficit [-]")
ax[0].set_ylim([0, 1])
ax[0].legend()
ax[0].grid()
ax[1].legend()
ax[1].grid()


## Second, reproduce Figure 1b (right) from Gunn (2019)

# Reset to the default model parameters
fmodel = FlorisModel("../inputs/ev.yaml")

# Adjustments
C_T2 = 0.34
D2 = 1.52

y_offset = -20
HH = 0.8

# Match depends on ambient turbulence intensity. 7.5% appears close. Also, switch off meandering.
TI = 0.075
fmodel.set_param(["wake", "wake_velocity_parameters", "eddy_viscosity", "wd_std_ev"], 0.0)

turb_1 = build_cosine_loss_turbine_dict(
    turbine_data_dict={
        "wind_speed":[0.0, 30.0],
        "power":[0.0, 1.0], # Not realistic but won't be used here
        "thrust_coefficient":[0.8, 0.8]
    },
    turbine_name="ConstantCT1",
    rotor_diameter=1,
    hub_height=HH,
    ref_tilt=0.0,
)
turb_2 = build_cosine_loss_turbine_dict(
    turbine_data_dict={
        "wind_speed":[0.0, 30.0],
        "power":[0.0, 1.0], # Not realistic but won't be used here
        "thrust_coefficient":[C_T2, C_T2]
    },
    turbine_name="ConstantCT2",
    rotor_diameter=D2,
    hub_height=HH,
    ref_tilt=0.0,
)

fmodel.set(
    layout_x=[0, 1.5],
    layout_y=[0, y_offset],
    turbine_type=[turb_1, turb_2],
    wind_speeds=[u_0],
    wind_directions=[270.0],
    turbulence_intensities=[TI],
    wind_shear=0.0,
    reference_wind_height=HH
)

n_pts = 1000
points_x = np.concatenate((np.linspace(0, 10, n_pts), np.linspace(1.5, 11.5, n_pts)))
points_y = np.concatenate((np.zeros(n_pts), y_offset*np.ones(n_pts)))
points_z = HH * np.ones_like(points_x)
u_at_points = fmodel.sample_flow_at_points(points_x, points_y, points_z)

fig, ax = plt.subplots(1,1)
ax.plot([0.0, 0.0], [0.3, 1.1], color="black")
ax.plot([1.5, 1.5], [0.3, 1.1], color="black")
ax.plot(points_x[:n_pts], u_at_points[0, :n_pts]/u_0, color="C0")
ax.plot(points_x[n_pts:], u_at_points[0, n_pts:]/u_0, color="C2", linestyle="--")
ax.grid()
ax.set_xlabel(r"$X$")
ax.set_ylabel(r"$\tilde{U}_c$")
ax.set_ylim([0.3, 1.1])

plt.show()
