
# Floating Wind Turbine Modeling

Here are the things to consider when modeling floating wind turbines.
As a turbine tilts, its operation can shift similar to how performance changes
when you yaw a turbine. The turbine is no longer operating on its defined Cp/Ct curve
and a vertical wake deflection can occur. As such, corrections for the effect of tilt
can be made to account for these changes.

Currently, FLORIS can correct the user-supplied Cp/Ct values for the effect of average tilt of
a turbine. This is accomplished by defining a `floating_tilt_table` in the turbine
input yaml which sets the steady tilt angle of the turbine based on wind speed. An interpolation
is created and this tilt angle is computed for each turbine based on effective velocity.
Taking into account the turbine rotor's built-in tilt, the absolute tilt change can then be used
to correct Cp and Ct. This tilt angle is then passed on to the respective wake model.

**NOTE** No wake models currently use the tilt for vertical wake deflection,
but it will be available with the inclusion of an upcoming wake model.
