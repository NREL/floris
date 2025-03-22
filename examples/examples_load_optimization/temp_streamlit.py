import io

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from floris import FlorisModel, TimeSeries
from floris.optimization.load_optimization.load_optimization import (
    compute_farm_revenue,
    compute_farm_voc,
    compute_load_ti,
    compute_net_revenue,
    compute_turbine_voc,
)


MIN_POWER_SETPOINT = 0.00000001

def run_analysis(A_gain, exp_ws_std, exp_thrust):
    # Parameters
    D = 126.0
    N = 100

    # Declare a floris model with default configuration
    fmodel = FlorisModel(configuration="defaults")
    # fmodel = FlorisModel("../inputs/emgauss.yaml")
    fmodel.set_operation_model("simple-derating")

    # Set up a two turbine farm
    fmodel.set(layout_x=[0, D * 5], layout_y=[0.0, 0.0])

    # Simple fixed input
    wind_directions = np.ones(N) * 270.0
    values = np.ones(N)
    load_ambient_tis = np.ones(N) * 0.231818
    time_series = TimeSeries(
        wind_directions=wind_directions, wind_speeds=8.0, turbulence_intensities=0.06, values=values
    )

    fmodel.set(wind_data=time_series)
    fmodel.run()

    # Calculate the A which would put the farm at a 10x revenue to VOC ratio
    # A_initial = find_A_to_satisfy_rev_voc_ratio(fmodel, 4.0, load_ambient_tis)

    # Only downstream
    A = A_gain  # * A_initial

    results = []

    # Plot everything
    fig, axarr = plt.subplots(7, 2, sharex=True, figsize=(8, 12))

    for t, t_i in zip(["Front", "Back"], [0, 1]):
        # st.write(f"### Optimizing derate for {t} turbine")

        # Set the setpoints to ramp down the upstream turbine
        power_setpoints = np.ones((N, 2)) * 5.0e6
        up_setpoints = np.linspace(2e6, MIN_POWER_SETPOINT, N)
        power_setpoints[:, t_i] = up_setpoints

        # Run this
        fmodel.set(power_setpoints=power_setpoints)
        fmodel.run()

        # Get all values
        turbine_powers = fmodel.get_turbine_powers()
        farm_power = fmodel.get_farm_power()
        farm_revenue = compute_farm_revenue(fmodel)
        turbine_load_tis = compute_load_ti(fmodel, load_ambient_tis)
        turbine_voc = compute_turbine_voc(
            fmodel, A, load_ambient_tis, exp_ws_std=exp_ws_std, exp_thrust=exp_thrust
        )
        farm_voc = compute_farm_voc(
            fmodel, A, load_ambient_tis, exp_ws_std=exp_ws_std, exp_thrust=exp_thrust
        )
        net_revenue = compute_net_revenue(
            fmodel, A, load_ambient_tis, exp_ws_std=exp_ws_std, exp_thrust=exp_thrust
        )

        # Find index of peak net revenue
        peak_index = np.argmax(net_revenue)

        # Show the TI values
        ax = axarr[0, t_i]
        for i in range(2):
            ax.plot(up_setpoints, turbine_load_tis[:, i], label=f"Turbine {i}")
        ax.set_ylabel("Turbine Load TI")
        ax.legend()
        ax.set_title(f"{t} Turbine")

        # Turbine powers
        ax = axarr[1, t_i]
        for i in range(2):
            ax.plot(up_setpoints, turbine_powers[:, i], label=f"Turbine {i}")
        ax.set_ylabel("Turbine Power (W)")
        ax.legend()

        # Farm Power
        ax = axarr[2, t_i]
        ax.plot(up_setpoints, np.sum(turbine_powers, axis=1), label="Farm Power (Turbine Sum)")
        ax.plot(up_setpoints, farm_power, label="Farm Power (Farm Model)")
        ax.set_ylabel("Farm Power (W)")
        ax.legend()

        # Farm Revenue
        ax = axarr[3, t_i]
        ax.plot(up_setpoints, farm_revenue, label="Farm Revenue")
        ax.set_ylabel("Farm Revenue ($)")
        ax.legend()

        # Turbine VOC
        ax = axarr[4, t_i]
        for i in range(2):
            ax.plot(up_setpoints, turbine_voc[:, i], label=f"Turbine {i} VOC")
        ax.set_ylabel("Turbine VOC")
        ax.legend()

        # Farm VOC and Farm Revenue
        ax = axarr[5, t_i]
        ax.plot(up_setpoints, farm_voc, label="Farm VOC")
        # ax.plot(up_setpoints, farm_revenue, label="Farm Revenue")
        ax.set_ylabel("Farm VOC")
        ax.legend()

        # Net revenue
        ax = axarr[6, t_i]
        ax.plot(up_setpoints, net_revenue, label="Net Revenue")
        ax.set_ylabel("Net Revenue ($)")

        axarr[-1, t_i].set_xlabel("Power Setpoint")

        for ax in axarr[:, t_i]:
            ax.grid(True)
            ax.axvline(up_setpoints[peak_index], color="r", lw=2)

        # fig.suptitle(f"Optimizing derate for {t} turbine (A_gain = {A_gain:.2f})")

        # Store results for display
        results.append(
            {
                "turbine": t,
                "peak_index": peak_index,
                "peak_power_setpoint": up_setpoints[peak_index],
                "peak_net_revenue": net_revenue[peak_index],
            }
        )
    st.pyplot(fig)

    return results


# Streamlit app
st.title("FLORIS Wind Farm Analysis")
st.write("Adjust the A_gain parameter to see how it affects the optimization")

# Slider for A_gain
with st.sidebar:
    A_gain = st.slider(
        "A_gain", min_value=0.001, max_value=3.0, value=0.42997704388655983, step=0.001
    )
    exp_ws_std = st.slider(
        "Exponent WS Std Dev", min_value=-4.00, max_value=4.0, value=1.0, step=0.01
    )
    exp_thrust = st.slider("Exponent Thrust", min_value=-4.00, max_value=4.0, value=1.0, step=0.01)

# Run the analysis when the slider is adjusted
results = run_analysis(A_gain, exp_ws_std, exp_thrust)

# Display summary
st.write("### Summary of Results")
for result in results:
    st.write(f"**{result['turbine']} Turbine**")
    st.write(f"Peak Net Revenue: ${result['peak_net_revenue']:.2f}")
    st.write(f"Optimal Power Setpoint: {result['peak_power_setpoint']:.2f} W")
    st.write("")
