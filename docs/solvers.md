# Solver Descriptions

This page will explain the different solvers available in FLORIS. For each of the solvers (except TurbOPark) there is the regular solver, which computes the wake effects at each turbine, an a full flow solver, which calculates the wake effects across the entire layout, including non-turbine positions. The full flow solver is used primarily for visualizing the flow field because it's more computationally intensive, so runtimes are much slower for larger wind power plants.

## Sequential

For each turbine in the layout (sorted from upstream to downstream), we calculate its effect on every downstream turbine. This is accomplished by calculating the deficit that each turbine adds to downstream turbines, then integrating it into the main data structure.

## CC Solver

TODO

## TurbOPark Solver

TODO
