# Solver Descriptions

FLORIS includes a collection of solver algorithms to support different
types of simulations and details for all wake models. The solver
is selected and configured in the `solver` block of the main input file
by specifying a type of grid, and each solver type has specific
configuration options.

The following solver-types are available:
- `sequential_solver`
    - Required grid: `TurbineGrid`
    - Primary use: AEP
    - This is the general purpose solver for any wake model that doesn't
        have a corresponding solver. It is often used in other solver
        algorithms to initialize the velocities at the turbines.
- `full_flow_sequential_solver`:
    - Required grid: `FullFlowGrid`, `FullFlowPlanarGrid`
    - Primary use: Visualization
    - This is a widely used solver typically for visualization of a plane
        of points within the fluid domain. It is compatible with any wake
        model that doesn't have a corresponding solver.
- `cc_solver`:
    - Required grid: `TurbineGrid`
    - Primary use: AEP with Cumulative-Curl model
    - This is a version of the `sequential_solver` specific to the
        Cumulative-Curl wake model.
- `full_flow_cc_solver`:
    - Required grid: `FullFlowGrid`, `FullFlowPlanarGrid`
    - Primary use: Visualization with Cumulative-Curl model
    - This is a version of the `full_flow_sequential_solver` specific to the
        Cumulative-Curl wake model.
- `turbopark_solver`:
    - Required grid: `TurbineGrid`
    - Primary use: AEP with TurbOPark model
    - This is a version of the `sequential_solver` specific to the
        TurbOPark wake model.
- `full_flow_turbopark_solver`:
    - Required grid: `FullFlowGrid`, `FullFlowPlanarGrid`
    - Primary use: Visualization with TurbOPark model
    - This is a version of the `full_flow_sequential_solver` specific to the
        TurbOPark wake model.



## Sequential Solvers

This collection of solvers iterates over each turbine in order from
upstream to downstream and applies the current turbine's wake impacts
onto the downstream grid points. The grid points are typically
associated with a turbine's rotor grid. The wake effect is calculated
for the entire downstream domain, and masks are used to apply the wake
only to points within a box of influence. The velocity deficit due to the
wake is combined with the freestream velocity via the chosen wake
combination model.

## Full Flow Solvers

These solvers are typically used for visualization of a 3d or 2D
collection of points. First, a sequential solver is used to calculate
the wake as described above. Then, another loop over all turbines
allows to add the impact of each turbine onto the points throughout
the fluid domain.
