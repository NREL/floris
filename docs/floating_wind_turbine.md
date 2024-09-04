
# Floating Wind Turbine Modeling

The FLORIS wind turbine description includes a definition of the performance curves
(`power` and `thrust_coefficient`) as a function of wind speed, and this lookup table is used
directly in
the calculation of power production for a steady-state atmospheric condition
(wind speed and wind direction). The power curve definition typically assumes a
fixed-bottom wind turbine with no active or controllable tilt. However, floating
wind turbines have additional rotational degrees of freedom including pitch which
adds a tilt angle to the rotor. As the turbine tilts, its performance is affected
similar to a yawed condition. The turbine is no longer operating on its defined
performance curve, and corrections must be included to accurately predict the power
production.

Support for modeling this impact on a floating wind turbine were added in
[PR#518](https://github.com/NREL/floris/pull/518/files) and allow for correcting the
user-supplied performance curve for the average tilt. This is accomplished by including
an additional input, `floating_tilt_table`, in the turbine definition which sets the
steady tilt angle of the turbine based on wind speed. An interpolation is created and
the tilt angle is computed for each turbine based on effective velocity. Taking into
account the turbine rotor's built-in tilt, the absolute tilt change can then be used
to correct the power and thrust coefficient.
This tilt angle is then used directly in the selected wake models.
