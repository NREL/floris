import copy
from pathlib import Path
from typing import (
    Any,
    List,
)

import attrs
import numpy as np
from attrs import (
    define,
    field,
    setters,
)

from floris.core import (
    BaseClass,
    State,
    Turbine,
)
from floris.core.turbine.operation_models import POWER_SETPOINT_DEFAULT
from floris.type_dec import (
    convert_to_path,
    floris_array_converter,
    iter_validator,
    NDArrayFloat,
    NDArrayInt,
    NDArrayObject,
    NDArrayStr,
)
from floris.utilities import load_yaml


default_turbine_library_path = Path(__file__).parents[1] / "turbine_library"


@define
class Farm(BaseClass):
    """Farm is where wind power plants should be instantiated from a YAML configuration
    file. The Farm will create a heterogeneous set of turbines that compose a wind farm,
    validate the inputs, and then create a vectorized representation of the the turbine
    data.

    Farm is the container class of the FLORIS package. It brings
    together all of the component objects after input (i.e., Turbine,
    Wake, FlowField) and packages everything into the appropriate data
    type. Farm should also be used as an entry point to probe objects
    for generating output.

    Args:
        layout_x (NDArrayFloat): A sequence of x-axis locations for the turbines that can be
            converted to a 1-D :py:obj:`numpy.ndarray`.
        layout_y (NDArrayFloat): A sequence of y-axis locations for the turbines that can be
            converted to a 1-D :py:obj:`numpy.ndarray`.
        turbine_type (list[dict | str]): A list of turbine definition dictionaries, or string
            references to the filename of the turbine type in either the FLORIS-provided turbine
            library (.../floris/turbine_library/), or a user-provided
            :py:attr:`external_turbine_library_path`.
        external_turbine_library_path (:obj:`str`): Either an absolute file path to the turbine
            library, or a path relative to the file that is running the analysis.
    """

    layout_x: NDArrayFloat = field(init=True, converter=floris_array_converter)
    layout_y: NDArrayFloat = field(init=True, converter=floris_array_converter)

    turbine_type: List = field(
        init=True,
        validator=iter_validator(list, (dict, str)),
        on_setattr=setters.frozen
    )

    external_turbine_library_path: Path = field(
        init=True,
        default=default_turbine_library_path,
        converter=convert_to_path
    )

    # Generated after initialization
    internal_turbine_library_path: Path = field(init=False, default=default_turbine_library_path)

    turbines: List[Turbine] = field(init=False, factory=list)
    turbine_type_map_sorted: NDArrayObject = field(init=False, factory=list)

    # TODO (later): Collect into a ControlSetpoint class
    yaw_angles: NDArrayFloat = field(init=False)
    power_setpoints: NDArrayFloat = field(init=False)
    awc_modes: NDArrayStr = field(init=False)
    awc_amplitudes: NDArrayFloat = field(init=False)
    awc_frequencies: NDArrayFloat = field(init=False)
    # TODO: Are TSRs control_setpoints? What models need them, and when/how? What is the eventual
    # use? Perhaps these are the "optimal" TSRs, which could be considered control setpoints, but
    # dont affect power.
    TSRs: NDArrayFloat = field(init=False, factory=list)

    # Convenience attributes extracted from the Turbine objects.
    hub_heights: NDArrayFloat = field(init=False)
    rotor_diameters: NDArrayFloat = field(init=False, factory=list)

    # Post-turbine solve attributes
    turbine_powers: NDArrayFloat = field(init=False, factory=list)
    turbine_thrust_coefficients: NDArrayFloat = field(init=False, factory=list)
    turbine_axial_inductions: NDArrayFloat = field(init=False, factory=list)
    turbine_rotor_average_velocities: NDArrayFloat = field(init=False, factory=list)

    # Private attributes
    # Private attributes.
    _turbine_types: List = field(init=False, validator=iter_validator(list, str), factory=list)
    _turbine_definition_cache: dict = field(init=False, factory=dict)
    _sorted_indices: NDArrayInt = field(init=False, factory=list)

    def __attrs_post_init__(self) -> None:
        # Turbine definitions can be supplied in three ways:
        # - A string selecting a turbine in the floris turbine library
        # - A Python dict representation of a turbine definition
        #   - There's an option to use the yaml keyword "!include" which results in the yaml
        #     library preprocessing the inputs and loading the specified file directly into
        #     the main input file. The result is that floris sees the turbine definition as a dict.
        # - A string selecting an turbine that exists in an external turbine library
        #   specified in `external_turbine_library_path`

        # Load all the turbine types into a cache to be mapped to specific turbine indices later.
        # This allows to read the yaml input files once rather than every time they're given.
        # In other words, if the turbine type is already in the cache, skip that iteration of
        # the for-loop.

        for t in self.turbine_type:
            # If a turbine type is a dict, then it was either preprocessed by the yaml
            # library to resolve the "!include" or it was set in a script as a dict. In either case,
            # add an entry to the cache
            if isinstance(t, dict):
                if t["turbine_type"] in self._turbine_definition_cache:
                    if self._turbine_definition_cache[t["turbine_type"]] == t:
                        continue # Skip t if already loaded
                    else:
                        raise ValueError(
                            "Two different turbine definitions have the same name: "\
                            f"'{t['turbine_type']}'. "\
                            "Please specify a unique 'turbine_type' for each turbine definition."
                        )
                self._turbine_definition_cache[t["turbine_type"]] = t
                self._turbine_definition_cache[t["turbine_type"]]["turbine_library_path"] = (
                    self.external_turbine_library_path
                )

            # If a turbine type is a string, then it is expected in the internal or external
            # turbine library
            if isinstance(t, str):
                if t in self._turbine_definition_cache:
                    continue # Skip t if already loaded

                # Check if the file exists in the internal and/or external library
                internal_fn = (self.internal_turbine_library_path / t).with_suffix(".yaml")
                external_fn = (self.external_turbine_library_path / t).with_suffix(".yaml")
                in_internal = internal_fn.exists()
                in_external = external_fn.exists()

                # If an external library is used and there's a duplicate of an internal
                # definition, then raise an error
                is_unique_path = self.external_turbine_library_path != default_turbine_library_path
                if is_unique_path and in_external and in_internal:
                    raise ValueError(
                        f"The turbine type: {t} exists in both the internal and external"
                        " turbine library."
                    )

                if in_internal:
                    full_path = internal_fn
                elif in_external:
                    full_path = external_fn
                else:
                    raise FileNotFoundError(
                        f"The turbine type: {t} does not exist in either the internal or"
                        " external turbine library."
                    )
                self._turbine_definition_cache[t] = load_yaml(full_path)
                self._turbine_definition_cache[t]["turbine_library_path"] = (
                    self.external_turbine_library_path
                )

        # Convert any dict entries in the turbine_type list to the type string. Since the
        # definition is saved above, we can make the whole list consistent now to use it
        # for mapping turbines later.
        # We use a private variable here instead of self.turbine_type because self.turbine_type
        # should always retain the input data. When this class is exported as_dict, the input
        # types must be used. If we modify that directly and change its shape, recreating this
        # class with a different layout but not a new self.turbine_type could cause the data
        # to be out of sync.
        self._turbine_types = [
            copy.deepcopy(t["turbine_type"]) if isinstance(t, dict) else t
            for t in self.turbine_type
        ]

        # If 1 turbine definition is given, expand to N turbines; this covers a 1-turbine
        # farm and 1 definition for multiple turbines
        if len(self._turbine_types) == 1:
            self._turbine_types *= self.n_turbines

        self.construct_turbines()

    @layout_x.validator
    def _check_x(self, attribute: attrs.Attribute, value: Any) -> None:
        if len(value) != len(self.layout_y):
            raise ValueError("layout_x and layout_y must have the same number of entries.")

    @layout_y.validator
    def _check_y(self, attribute: attrs.Attribute, value: Any) -> None:
        if len(value) != len(self.layout_x):
            raise ValueError("layout_x and layout_y must have the same number of entries.")

    @turbine_type.validator
    def _check_turbine_type(self, attribute: attrs.Attribute, value: Any) -> None:
        # Check that the list of turbines is either of length 1 or N turbines
        if len(value) != 1 and len(value) != self.n_turbines:
            raise ValueError(
                "turbine_type must have the same number of entries as layout_x/layout_y or have "
                "a single turbine_type value. This error can arise if you set the turbine_type or "
                "alter the operation model before setting the layout."
            )

    @external_turbine_library_path.validator
    def _check_library_path(self, attribute: attrs.Attribute, value: Path) -> None:
        """Ensures that the input to `library_path` exists and is a directory."""
        if not value.is_dir():
            raise FileExistsError(f"The input file path: {str(value)} is not a valid directory.")

    def initialize(self):
        # Create structures for storing the turbine outputs
        if not hasattr(self, "_sorted_indices"):
            raise ValueError(
                "The Farm object must be initialized with the sorted indices from a Grid object "
                "before it can be used. Please call Farm.set_sorted_indices() first."
            )

        self.turbine_powers = np.full((self._sorted_indices.shape[0], self.n_turbines), np.nan)
        self.turbine_thrust_coefficients = np.full(
            (self._sorted_indices.shape[0], self.n_turbines), np.nan
        )
        self.turbine_axial_inductions = np.full(
            (self._sorted_indices.shape[0], self.n_turbines), np.nan
        )
        self.turbine_rotor_average_velocities = np.full(
            (self._sorted_indices.shape[0], self.n_turbines), np.nan
        )

        self.state = State.INITIALIZED

    def construct_turbines(self):
        turbines_unique = {
            k: Turbine.from_dict(v) for k, v in self._turbine_definition_cache.items()
        }
        self.turbines = [turbines_unique[k] for k in self._turbine_types]

        # Extract various attributes for convenience.
        self.hub_heights = np.array([t.hub_height for t in self.turbines])
        self.rotor_diameters = np.array([t.rotor_diameter for t in self.turbines])
        self.TSRs = np.array([t.TSR for t in self.turbines])

    def set_sorted_indices(self, sorted_indices: NDArrayInt):
        self._sorted_indices = sorted_indices

    def construct_turbine_type_map(self):
        self.turbine_type_map_sorted = np.take_along_axis(
            np.reshape(
                [t.turbine_type for t in self.turbines] * self._sorted_indices.shape[0],
                np.shape(self._sorted_indices)
            ),
            self._sorted_indices,
            axis=1
        )

    def set_yaw_angles(self, yaw_angles: NDArrayFloat | list[float]):
        self.yaw_angles = np.array(yaw_angles)

    def set_yaw_angles_to_ref_yaw(self, n_findex: int):
        yaw_angles = np.zeros((n_findex, self.n_turbines))
        self.set_yaw_angles(yaw_angles)

    def set_power_setpoints(self, power_setpoints: NDArrayFloat):
        self.power_setpoints = np.array(power_setpoints)

    def set_power_setpoints_to_ref_power(self, n_findex: int):
        power_setpoints = POWER_SETPOINT_DEFAULT * np.ones((n_findex, self.n_turbines))
        self.set_power_setpoints(power_setpoints)

    def set_awc_modes(self, awc_modes: NDArrayStr):
        self.awc_modes = np.array(awc_modes)

    def set_awc_modes_to_ref_mode(self, n_findex: int):
        awc_modes = np.array([["baseline"]*self.n_turbines]*n_findex)
        self.set_awc_modes(awc_modes)

    def set_awc_amplitudes(self, awc_amplitudes: NDArrayFloat):
        self.awc_amplitudes = np.array(awc_amplitudes)

    def set_awc_amplitudes_to_ref_amp(self, n_findex: int):
        awc_amplitudes = np.zeros((n_findex, self.n_turbines))
        self.set_awc_amplitudes(awc_amplitudes)

    def set_awc_frequencies(self, awc_frequencies: NDArrayFloat):
        self.awc_frequencies = np.array(awc_frequencies)

    def set_awc_frequencies_to_ref_freq(self, n_findex: int):
        awc_frequencies = np.zeros((n_findex, self.n_turbines))
        self.set_awc_frequencies(awc_frequencies)

    def set_control_setpoints_to_reference(self, n_findex: int):
        self.set_yaw_angles_to_ref_yaw(n_findex)
        self.set_power_setpoints_to_ref_power(n_findex)
        self.set_awc_modes_to_ref_mode(n_findex)
        self.set_awc_amplitudes_to_ref_amp(n_findex)
        self.set_awc_frequencies_to_ref_freq(n_findex)

    def finalize(self):
        self.state.USED

    @property
    def coordinates(self):
        return np.array([
            np.array([x, y, z]) for x, y, z in zip(
                self.layout_x,
                self.layout_y,
                self.hub_heights if len(self.hub_heights.shape) == 1 else self.hub_heights[0]
            )
        ])

    @property
    def n_turbines(self):
        return len(self.layout_x)

    @property
    def rotor_diameters_sorted(self):
        return _sort_by_coord_indices(self.rotor_diameters, self._sorted_indices)

    @property
    def hub_heights_sorted(self):
        return _sort_by_coord_indices(self.hub_heights, self._sorted_indices)

    @property
    def TSRs_sorted(self):
        return _sort_by_coord_indices(self.TSRs, self._sorted_indices)

    @property
    def yaw_angles_sorted(self):
        return _sort_by_coord_indices(self.yaw_angles, self._sorted_indices)

    @property
    def power_setpoints_sorted(self):
        return _sort_by_coord_indices(self.power_setpoints, self._sorted_indices)

    @property
    def awc_modes_sorted(self):
        return _sort_by_coord_indices(self.awc_modes, self._sorted_indices)

    @property
    def awc_amplitudes_sorted(self):
        return _sort_by_coord_indices(self.awc_amplitudes, self._sorted_indices)

    @property
    def awc_frequencies_sorted(self):
        return _sort_by_coord_indices(self.awc_frequencies, self._sorted_indices)

    @property
    def turbine_type_map(self):
        return np.broadcast_to(
            np.array([t.turbine_type for t in self.turbines]),
            (self._sorted_indices.shape[0], self.n_turbines)
        )

def _sort_by_coord_indices(array, sorted_indices):
    if array.ndim != 2:
        template_shape = np.ones_like(sorted_indices)
        return np.take_along_axis(
            array * template_shape,
            sorted_indices,
            axis=1
        )
    elif array.ndim == 2:
        return np.take_along_axis(
            array,
            sorted_indices,
            axis=1
        )
    else:
        raise ValueError("Array must be 1-dimensional or 2-dimensional to sort.")
