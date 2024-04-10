# Switching from FLORIS v3 to v4

There are several major changes introduced in FLORIS v4. The largest underlying change is that,
where FLORIS v3 had a "wind directions" and a "wind speeds" dimension to its internal data
structures, FLORIS v4 collapses these into a single dimension, which we refer to as the `findex`
dimension. This dimension contains each "configuration" or "condition" to be run, and is
conceptually similar to running FLORIS v3 in `time_series` mode. At the user interface level, the
largest implication of this change is that users must specify `wind_directions`, `wind_speeds`, and
`turbulence_intensities` (new) as arrays of equal length; and these are "zipped" to create the
conditions for FLORIS to run, rather than creating a grid of all combinations. This is discussed
further in [Setting and Running](#setting-and-running).

## Setting and running

In FLORIS v3, users interacted with FLORIS by instantiating a `FlorisInterface` object, nominally
called `fi`. The notion here is that the users "interface" with the underlying FLORIS code using
`fi`. For FLORIS v4, we acknowledge that to most users, this main "interface" object, for all
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
"paired" (rather than gridded) for the computation. For instance, if the user provides `n_findex`
wind directions, they _must_ provide `n_findex` wind speeds and `n_findex` turbulence intensities;
as well as `n_findex`x`n_turbines` yaw angles, if yaw angles are being used.
- Providing varying `turbulence_intensities` is new for FLORIS v4.
- To facilitate "easier" use of the `set()` method (for instance, to run all combinations of
wind directions and wind speeds), we now provide `WindData` objects that can be passed directly to
`set()`'s `wind_data` keyword argument. See [Wind data](#wind-data) as well as
[Wind Data Objects](wind_data_user) for more information.
- `calculate_no_wake()` has been replaced with `run_no_wake()`
- `get_farm_AEP()` no longer calls `run()`; to compute the farm AEP, users should `run()` the
`fmodel` themselves before calling `get_farm_AEP()`.

An example workflow for using `set` and `run` is:
```python
import numpy as np
from floris import FlorisModel

fmodel = FlorisModel("input_file.yaml") # Input file with 3 turbines

# Set up a base case and run
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

For more advanced users, it is best to group many conditions into single calls of `set` and `run`
than to step through various conditions individually, as this will make the best use of FLORIS's
vectorization capabilities.

## Input files
As in FLORIS v3, there are two main input files to FLORIS v4:
1. The "main" FLORIS input yaml, which contains wake model parameters and wind farm data
2. The "turbine" input yaml, which contains data about the wind turbines

Examples for main FLORIS input yamls are in examples/inputs/. Default turbine yamls, which many
users
may use if they do not have their own turbine models to use, can be found in
floris/turbine_library/.
See also [Turbine Library Interface](input_reference_turbine) and
[Main Input File Reference](input_reference_main).

Conceptually, both the main FLORIS input yaml and the turbine input yaml is much the same in v4 as
in v3. However, there are a few changes to the fields on each that mean that existing yamls for v3
will not run in v4 as is.

#### Main FLORIS input yaml
On the main FLORIS input file, the `turbulence_intensity` field (on`flow_field`),
which was specified as a scalar in FLORIS v3, has been changed to `turbulence_intensities`, and
should now contain a list of turbulence intensities that is of the same length as `wind_directions`
and `wind_speeds`. Additionally, the length of the lists for `wind_directions` and `wind_speeds`
_must_ now be of equal length.

In addition, a new field `enable_active_wake_mixing` has been added to the `wake` field,
which users may set to `false` unless they would like to use active wake mixing strategies such
as [Helix](empirical_gauss_model.md#Added-mixing-by-active-wake-control).

#### Turbine input yaml
To reflect the transition to more flexible [operation models](#operation-model), there are a
number of changes to the fields on the turbine yaml. The changes are mostly regrouping and
renaming of the existing fields.
- The `power_thrust_table` field now has `wind_speed` and `power` fields, as before; however,
the `thrust` field has been renamed `thrust_coefficient` for clarity, and the `power` field now
specifies the turbine _absolute_ power (in kW) rather than the _power coefficient_.
- Additionally, any extra parameters and data required by operation models to evaluate the power
and thrust curves have been moved onto the `power_thrust_table` field. This includes
`ref_density_cp_ct` (renamed `ref_air_density` and moved onto the `power_thrust_table`);
`ref_tilt_cp_ct` (renamed `ref_tilt` and moved onto the `power_thrust_table`); and `pP` and `pT`
(renamed `cosine_loss_exponent_yaw` and `cosine_loss_exponent_tilt`, respectively, and moved onto
the `power_thrust_table`).
- The `generator_efficiency` field has been removed. The `power` field on `power_thrust_table`
should reflect the electrical power produced by the turbine, including any losses.
- A new field `operation_model` has been added, whose value should be a string that selects the
operation model the user would like to evaluate. The default is `"cosine-loss"`,
which recovers FLORIS v3-type turbine operation. See [Operation model](#operation-model) and
[Turbine Operation Models](operation_models_user) for details.

### Converting v3 yamls to v4
To aid users in converting their existing v3 main FLORIS input yamls and turbine input, we provide
two utilities:
- floris/tools/convert_floris_input_v3_to_v4.py
- floris/tools/convert_turbine_v3_to_v4.py

These can be executed from the command line and expect to be passed the exiting v3 yaml as an input;
the will then write a new v4-compatible yaml of the same name but appended _v4.
```bash
python convert_floris_input_v3_to_v4.py your_v3_input_file.yaml
python convert_floris_turbine_v3_to_v4.py your_v3_turbine_file.yaml
```

Additionally, a function for building a turbine dictionary that can be passed directly to the
`turbine_type` argument of `FlorisModel.set()` is provided:
```python
from floris.turbine_library.turbine_utilities import build_cosine_loss_turbine_dict
```

### Reference turbine updates
The power and thrust curves for the NREL 5MW, IEA 10MW, and IEA 15MW turbines have been updated
slightly do reflect publicly available data. The x_20MW reference turbine has been removed, as data
was not readily available. See [Turbine Library Interface](turbine_interaction).

## Wind data
To aid users in setting the wind conditions they are interested in running, we provide "wind data"
classes, which can be passed directly to `FlorisModel.set()`'s `wind_data` keyword argument in place
of `wind_directions`, `wind_speeds`, and `turbulence_intensities`. The wind data objects enable,
for example, gridding inputs (`WindRose` and `WindTIRose`) and broadcasting a scalar-valued
turbulence intensity (`TimeSeries`).
```python
import numpy as np
from floris import FlorisModel
from floris import TimeSeries

fmodel = FlorisModel("input_file.yaml") # Input file with 3 turbines

time_series = TimeSeries(
    wind_directions=np.array([270.0, 270.0]),
    wind_speeds=8.0,
    turbulence_intensities=0.06
)
fmodel.set(wind_data=time_series)
fmodel.set(wind_data=time_series)turbine_powers_base = fmodel.get_turbine_powers()
turbine_powers = fmodel.get_turbine_powers()
```

More information about the various wind data classes can be found at
[Wind Data Objects](wind_data_user).

## Operation model
FLORIS v4 allows for significantly more flexible turbine operation via
[Turbine Operation Models](operation_models_user). These allow users to specify how a turbine loses
power when yaw misaligned; how a turbine operates when derated; and how turbines produce power
and thrust when operating with active wake mixing strategies. The default operation model is the
`"cosine-loss"` model, which models a turbine's power loss when in yaw misalignment using the same
cosine model as was hardcoded in FLORIS v3.
