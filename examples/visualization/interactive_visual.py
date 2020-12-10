# TO USE THIS FROM COMMAND TYPE:
#    streamlit run interactive_visual.py

import streamlit as st
import floris.tools as wfct
import matplotlib.pyplot as plt


# Fixed parameters
minSpeed = 4
maxSpeed = 8.0

# Options
wd = st.sidebar.slider("Wind Direction", 250, 290, 270, step=2)
yaw_1 = st.sidebar.slider("Yaw angle T1", -30, 30, 0, step=1)
x_loc = st.sidebar.slider("x normal plane intercept", 0, 3000, 500, step=10)
y_loc = st.sidebar.slider("y normal plane intercept", -100, 100, 0, step=5)


# Initialize the FLORIS interface fi
fi = wfct.floris_interface.FlorisInterface("../example_input.json")
fi.reinitialize_flow_field(
    wind_direction=wd, layout_array=((0, 126 * 7, 126 * 14), (0, 0, 0))
)
fi.calculate_wake(yaw_angles=[yaw_1, 0, 0])

st.write("# Results with GCH")

# Horizontal plane
hor_plane = fi.get_hor_plane()
fig, ax = plt.subplots()
wfct.visualization.visualize_cut_plane(
    hor_plane, ax=ax, minSpeed=minSpeed, maxSpeed=maxSpeed
)
ax.axhline(y_loc, color="w", ls="--", lw=1)
ax.axvline(x_loc, color="w", ls="--", lw=1)
st.write("## Horizontal Cut Plane")
st.write(fig)

# Cross (x-normal) plane
cross_plane = fi.get_cross_plane(x_loc=x_loc)
fig, ax = plt.subplots()
wfct.visualization.visualize_cut_plane(
    cross_plane, ax=ax, minSpeed=minSpeed, maxSpeed=maxSpeed
)
st.write("## Cross (X-Normal) Cut Plane")
wfct.visualization.reverse_cut_plane_x_axis_in_plot(ax)
st.write(fig)

# Cross (y-normal) plane
cross_plane = fi.get_y_plane(y_loc=y_loc)
fig, ax = plt.subplots()
wfct.visualization.visualize_cut_plane(
    cross_plane, ax=ax, minSpeed=minSpeed, maxSpeed=maxSpeed
)
st.write("## Cross (Y-Normal) Cut Plane")
st.write(fig)


# NO GCH RESULTS
st.write("# Results without GCH")
fi = wfct.floris_interface.FlorisInterface("../example_input.json")
fi.set_gch(False)
fi.reinitialize_flow_field(
    wind_direction=wd, layout_array=((0, 126 * 7, 126 * 14), (0, 0, 0))
)
fi.calculate_wake(yaw_angles=[yaw_1, 0, 0])


# Horizontal plane
hor_plane = fi.get_hor_plane()
fig, ax = plt.subplots()
wfct.visualization.visualize_cut_plane(
    hor_plane, ax=ax, minSpeed=minSpeed, maxSpeed=maxSpeed
)
ax.axhline(y_loc, color="w", ls="--", lw=1)
ax.axvline(x_loc, color="w", ls="--", lw=1)
st.write("## Horizontal Cut Plane")
st.write(fig)

# Cross (x-normal) plane
cross_plane = fi.get_cross_plane(x_loc=x_loc)
fig, ax = plt.subplots()
wfct.visualization.visualize_cut_plane(
    cross_plane, ax=ax, minSpeed=minSpeed, maxSpeed=maxSpeed
)
st.write("## Cross (X-Normal) Cut Plane")
wfct.visualization.reverse_cut_plane_x_axis_in_plot(ax)
st.write(fig)

# Cross (y-normal) plane
cross_plane = fi.get_y_plane(y_loc=y_loc)
fig, ax = plt.subplots()
wfct.visualization.visualize_cut_plane(
    cross_plane, ax=ax, minSpeed=minSpeed, maxSpeed=maxSpeed
)
st.write("## Cross (Y-Normal) Cut Plane")
st.write(fig)
