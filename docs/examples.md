(examples)=
# Examples Index

The FLORIS software repository includes a set of
[examples/](https://github.com/NREL/floris/tree/main/examples)
intended to describe most features as well as provide a starting point
for various analysis methods. These are generally ordered from simplest
to most complex. The examples and their content are described below.
Prior to exploring the examples, it is highly recommended to review
[](background_concepts).


## Basic setup and pre and post processing

These examples are primarily for demonstration and explanation purposes.
They build up data for a simulation, execute the calculations, and do various
post processing steps to analyse the results.

### 01_opening_floris_computing_power.py
This example script loads an input file and makes changes to the turbine layout
and atmospheric conditions. It then configures wind turbine yaw settings and
executes the simulation. Finally, individual turbine powers are reported.
It demonstrates the vectorization capabilities of FLORIS by first creating
a simulation with a single wind condition, and then creating another
simulation with multiple wind conditions.

### 02_visualizations.py
Create visualizations for x, y, and z planes in the whole farm as well as plots of the grid points
on each turbine rotor.

### 03_making_adjustments.py
Make various changes to an initial configuration and plot results on a single figure.
- Change atmospheric conditions including wind speed, wind direction, and shear
- Create a new layout
- Configure yaw settings

### 04_sweep_wind_directions.py
Simulate a wind farm over multiple wind directions and one wind speed.
Evaluate the individual turbine powers.
- Setting up a problem considering the vectorization of the calculations
  - Data structures
  - Broadcasted mathematical operations

### 05_sweep_wind_speeds.py
Same as above except multiple wind speeds and one wind direction.
Evaluate the individual turbine powers.
- Setting up a problem considering the vectorization of the calculations
  - Data structures
  - Broadcasted mathematical operations

### 06_sweep_wind_conditions.py
Simulate a wind farm with multiple wind speeds and wind directions.
- Setting up a problem considering the vectorization of the calculations
  - Data structures
  - Broadcasted mathematical operations

### 07_calc_aep_from_rose.py
Load wind rose information from a .csv file and calculate the AEP of
a wind farm.
- Create a new layout
- Arrange the wind rose data into arrays
- Create the frequency information from the wind condition data

### 08_calc_aep_from_rose_use_class.py
Do the above but use the included WindRose class.

### 09_compare_farm_power_with_neighbor.py
Consider the affects of one wind farm on another wind farm's AEP.

### 20_calculate_farm_power_with_uncertainty.py
Calculate the farm power with a consideration of uncertainty
with the default gaussian probability distribution.

### 21_demo_time_series.py
Simulate a time-series of wind condition data and generate plots
of turbine power over time.

### 22_get_wind_speed_at_turbines.py
Similar to the "Getting Started" tutorial. Sets up a simulation and
prints the wind speeds at all turbines.

### 16_heterogeneous_inflow.py
Define non-uniform (heterogeneous) atmospheric conditions by specifying
speedups at locations throughout the farm. Show plots of the
impact on wind turbine wakes.

### 16b_heterogeneity_multiple_ws_wd.py
Illustrate usage of heterogeneity with multiple wind speeds and directions.

## 16c_optimize_layout_with_heterogeneity.py
This example shows a layout optimization using the geometric yaw option. It
combines elements of examples 15 (layout optimization) and 16 (heterogeneous
inflow) for demonstrative purposes. If you haven't yet run those examples,
we recommend you try them first.

Heterogeneity in the inflow provides the necessary driver for coupled yaw
and layout optimization to be worthwhile. First, a layout optimization is
run without coupled yaw optimization; then a coupled optimization is run to
show the benefits of coupled optimization when flows are heterogeneous.

### 17_multiple_turbine_types.py
Load an input file that describes a wind farm with two turbines
of different types and plot the wake profiles.

### 23_visualize_layout.py
Use the visualize_layout function to provide diagram visualization
of a turbine layout within FLORIS.

### 24_floating_turbine_models.py
Demonstrates the definition of a floating turbine and how to enable the effects of tilt
on Cp and Ct.

For further examples on floating wind turbines, see also examples
25 (vertical wake deflection by a forced tilt angle) and 29 (comparison between
a fixed-bottom and floating wind farm).

### 25_tilt_driven_vertical_wake_deflection.py

This example demonstrates vertical wake deflections due to the tilt angle when running
with the Empirical Gauss model. Note that only the Empirical Gauss model implements
vertical deflections at this time. Also be aware that this example uses a potentially
unrealistic tilt angle, 15 degrees, to highlight the wake deflection. Moreover, the magnitude
of vertical deflections due to tilt has not been validated.

For further examples on floating wind turbines, see also examples
24 (effects of tilt on turbine power and thrust coefficients) and 29
(comparison between a fixed-bottom and floating wind farm).

### 26_empirical_gauss_velocity_deficit_parameters.py

This example illustrates the main parameters of the Empirical Gaussian
velocity deficit model and their effects on the wind turbine wake.

### 27_empirical_gauss_deflection_parameters.py
This example illustrates the main parameters of the Empirical Gaussian
deflection model and their effects on the wind turbine wake.

### 28_extract_wind_speed_at_points.py
This example demonstrates the use of the `FlorisInterface.sample_flow_at_points` method
to extract the wind speed information at user-specified locations in the flow.

Specifically, this example gets the wind speed at a single x, y location and four different
heights over a sweep of wind directions. This mimics the wind speed measurements of a met
mast across all wind directions (at a fixed free stream wind speed).

Try different values for met_mast_option to vary the location of the met mast within
the two-turbine farm.

### 32_plot_velocity_deficit_profiles.py
This example illustrates how to plot velocity deficit profiles at several locations
downstream of a turbine. Here we use the following definition:

    velocity_deficit = (homogeneous_wind_speed - u) / homogeneous_wind_speed
        , where u is the wake velocity obtained when the incoming wind speed is the
        same at all heights and equal to `homogeneous_wind_speed`.

### 29_floating_vs_fixedbottom_farm.py

Compares a fixed-bottom wind farm (with a gridded layout) to a floating
wind farm with the same layout. Includes:
- Turbine-by-turbine power comparison for a single wind speed and direction
- Flow visualizations for a single wind speed and direction
- AEP calculations based on an example wind rose.

For further examples on floating wind turbines, see also examples
24 (effects of tilt on turbine power and thrust coefficients) and 25
(vertical wake deflection by a forced tilt angle).

### 30_multi_dimensional_cp_ct.py

This example showcases the capability of using multi-dimensional Cp/Ct data in turbine defintions
dependent on external conditions. Specifically, fictional data for varying Cp/Ct values based on
wave period, Ts, and wave height, Hs, is used, showing the user how to setup the turbine
definition and input file. Also demonstrated is the different method for getting turbine
powers when using multi-dimensional Cp/Ct data.

### 31_multi_dimensional_cp_ct_2Hs.py

This example builds on example 30. Specifically, fictional data for varying Cp/Ct values based on
wave period, Ts, and wave height, Hs, is used to show the difference in power performance for
different wave heights.

### 32_specify_turbine_power_curve.py

This example demonstrates how to generate a turbine dictionary or yaml input file based on
a specified power and thrust curve. The power and thrust curves may be specified as power
and thrust coefficients or as absolute values.

## Optimization

These examples demonstrate use of the optimization routines
included in FLORIS through {py:mod}`floris.tools.optimization`. These
focus on yaw settings and wind farm layout, but the concepts
are general and can be used for other optimizations.

### 10_opt_yaw_single_ws.py
Using included yaw optimization routines, run a yaw optimization for a single wind speed
and plot yaw settings and performance.

### 11_opt_yaw_multiple_ws.py
Using included yaw optimization routines, run a yaw optimization for multiple wind
conditions including multiple wind speeds and wind directions.
Similar to above but with extra steps for post processing.

### 12_optimize_yaw.py
Construct wind farm yaw settings for a full wind rose based on the
optimized yaw settings at a single wind speed. Then, compare
results to the baseline no-yaw configuration.

### 12_optimize_yaw_in_parallel.py
Comparable to the above but perform all the computations using
parallel processing. In the current example, use 16 cores
simultaneously to calculate the AEP and perform a wake steering
yaw angle optimization for multiple wind speeds.

### 13_optimize_yaw_with_neighboring_farm.py
Same as above but considering the effects of a nearby wind farm.

### 14_compare_yaw_optimizers.py
Show the difference in optimization results for
- SerialRefine
- SciPy

### 15_optimize_layout.py
Optimize a wind farm layout for AEP within a square boundary and a
random wind resource using the SciPy optimization routines.


## Gallery

The examples listed here are fun and interesting. If you're doing something
cool with FLORIS and want to share, create a pull request with your example
listed here!

### 18_check_turbine.py
Plot power and thrust curves for each turbine type included in the
turbine library. Additionally, plot the losses due to yaw.

### 19_streamlit_demo.py
Creates a Streamlit dashboard to quickly modify the layout and
atmospheric conditions of a wind farm.
