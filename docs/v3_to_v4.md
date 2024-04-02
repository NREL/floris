# Switching from FLORIS v3 to FLORIS v4

There are several major changes introduced in FLORIS v4. The largest underlying change is that, 
where FLORIS v3 had a "wind directions" and a "wind speeds" dimension to its internal data
structures, FLORIS v4 collapses these into a single dimension, which we refer to as the `findex`
dimension. This dimension contains each "configuration" or "condition" to be run, and is
conceptually similar to running FLORIS v3 in `time_series` mode. At the user interface level, the 
largest implication of this change is that users must specify `wind_directions`, `wind_speeds`, and 
`turbulence_intensities` (new) as arrays of equal length; and these are "zipped" to create the 
conditions for FLORIS to run, rather than creating a grid of all combinations. This is discussed 
further in [Setting and Running](#setting-and-running).

- FlorisInterface -> FlorisModel
- reinitialize, calculate_wake -> set, run
- arguments to each
- setpoints STORED; can use reset_operation
- Time series mode everywhere
- Stricter about consistent inputs
- wind data objects
- operation models

## Setting and running

In FLORIS v3, users interacted with FLORIS by instantiating a `FlorisInterface` object, nominally 
called `fi`. The notion here is that the users "interface" with the underlying FLORIS code using `fi`. For FLORIS v4, we acknowledge that to most users, this main "interface" object, for all 
intents and purposes, _is FLORIS_. We therefore have renamed the `FlorisInterface` the
`FlorisModel`, nominally instantiated as `fmodel`. To instantiate a `FlorisModel`, the code is 
very similar to before, i.e.
```python
from floris import FlorisModel

fmodel = FlorisModel("input_file.yaml")
```

Previously, to set the atmospheric conditions on `fi`, users called the `reinitialize()` method; 
and to run the calculations, as well as provide any control setpoints such as yaw angles, users 
generally called `calculate_wake()`. Some of the other methods on `FlorisInterface` also called 
`calculate_wake()` internally, most notably `get_farm_AEP()`.

For FLORIS v4, we have changed from the (`reinitialize()`, `calculate_wake()`) paradigm to a new 
pair of methods (`set()`, `run()`). `set()` is similar to the retired `reinitialize()` method, and 
`run()` is similar to the retired `calculate_wake()` method. However, there are some important 
differences:
- `FlorisModel.set()` accepts both atmospheric conditions _and_ control setpoints.
- `FlorisModel.run()` accept no arguments. Its sole function is to run the FLORIS calculation.
- Control setpoints are now "remembered". Previously, if control setpoints (`yaw_angles`) were
passed to `calculate_wake()`, they were discarded at the end of the calculation. In FLORIS v4, the 
control setpoints passed to `set()` are stored, and invoking `run()` multiple times will continue to
use those control setpoints. 
- To "forget" previously provided control setpoints, use the new method
`FlorisModel.reset_operation()`.
- When providing arguments to `set()`, all arguments much have the same length, as they will be 
"paired" (rather than gridded) for the computation. For instance, if the user provides `n`
wind directions, they _must_ provide `n` wind speeds and `n` turbulence intensities; as well as 
`n`x`n_turbines` yaw angles, if yaw angles are being used.
- Providing varying `turbulence_intensities` is new for FLORIS v4.
- To facilitate "easier" use of the `set()` method (for instance, to run all combinations of 
wind directions and wind speeds), we now provide `WindData` objects that can be passed directly to 
`set()`'s `wind_data` keyword argument. See [Wind data](#wind-data) as well as 
[Wind Data Objects](wind_data_user) for more information.
- `calculate_no_wake()` has been replaced with `run_no_wake()`

An example workflow for using `set` and `run` is:
```python
import numpy as np
from floris import FlorisModel

fmodel = FlorisModel("input_file.yaml") # Input file with 3 turbines


fmodel.set(
    wind_directions=np.array([270., 270.]),
    wind_speeds=np.array([8.0, 8.0]),
    turbulence_intensities=np.array([0.06, 0.06])
)
fmodel.run()
turbine_powers_base = fmodel.get_turbine_powers()

# Provide yaw angles
fmodel.set(
    yaw_angles=np.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]) # n_findex x n_turbines
)
fmodel.run()
turbine_powers_yawed = fmodel.get_turbine_powers()

# If we run again, this time with no wake, the provided yaw angles will still be used
fmodel.run_no_wake()
turbine_powers_yawed_nowake = fmodel.get_turbine_powers()

# To "forget" the yaw angles, we use the reset_operation method
fmodel.reset_operation()
fmodel.run_no_wake()
turbine_powers_base_nowake = fmodel.get_turbine_powers()
```

## Input files
- main input yaml and turbine yaml, as before
- changes to each (describe)
- utilities for converting between them.


## Quick Start

FLORIS is a Python package run on the command line typically by providing
an input file with an initial configuration. It can be installed with
```pip install floris``` (see {ref}`installation`). The typical entry point is
{py:class}`.FlorisInterface` which accepts the path to the
input file as an argument. From there, changes can be made to the initial
configuration through the {py:meth}`.FlorisInterface.reinitialize`
routine, and the simulation is executed with
{py:meth}`.FlorisInterface.calculate_wake`.

```python
from floris.tools import FlorisInterface
fi = FlorisInterface("path/to/input.yaml")
fi.reinitialize(wind_directions=[i for i in range(10)])
fi.calculate_wake()
```

Finally, results can be analyzed via post-processing functions available within
{py:class}`.FlorisInterface` such as
{py:meth}`.FlorisInterface.get_turbine_layout`,
{py:meth}`.FlorisInterface.get_turbine_powers` and
{py:meth}`.FlorisInterface.get_farm_AEP`, and
a visualization package is available in {py:mod}`floris.tools.visualization`.
A collection of examples are included in the [repository](https://github.com/NREL/floris/tree/main/examples)
and described in detail in {ref}`examples`.

## Engaging on GitHub

FLORIS leverages the following GitHub features to coordinate support and development efforts:

- [Discussions](https://github.com/NREL/floris/discussions): Collaborate to develop ideas for new use cases, features, and software designs, and get support for usage questions
- [Issues](https://github.com/NREL/floris/issues): Report potential bugs and well-developed feature requests
- [Projects](https://github.com/orgs/NREL/projects/18/): Include current and future work on a timeline and assign a person to "own" it

Generally, the first entry point for the community will be within one of the
categories in Discussions.
[Ideas](https://github.com/NREL/floris/discussions/categories/ideas) is a great spot to develop the
details for a feature request. [Q&A](https://github.com/NREL/floris/discussions/categories/q-a)
is where to get usage support.
[Show and tell](https://github.com/NREL/floris/discussions/categories/show-and-tell) is a free-form
space to show off the things you are doing with FLORIS.
