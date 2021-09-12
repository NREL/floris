# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from typing import Any, Dict, List, Union

import copy
from typing import Dict, List, Union

import attr
import numpy as np
import xarray as xr

from .turbine import Turbine
from .utilities import Vec3, FromDictMixin, iter_validator, attrs_array_converter


def create_turbines(mapping: Dict[str, dict]) -> Dict[str, Turbine]:
    return {t_id: Turbine.from_dict(config) for t_id, config in mapping.items()}


def generate_turbine_tuple(turbine: Turbine) -> tuple:
    exclusions = ("power_thrust_table", "model_string")
    return attr.astuple(
        turbine, filter=lambda attribute, value: attribute.name not in exclusions
    )


def generate_turbine_attribute_order(turbine: Turbine) -> List[str]:
    exclusions = ("power_thrust_table", "model_string")
    mapping = attr.asdict(
        turbine, filter=lambda attribute, value: attribute.name not in exclusions
    )
    return list(mapping.keys())


@attr.s(auto_attribs=True)
class FarmGenerator(FromDictMixin):
    """NewFarm is where wind power plants should be instantiated from a YAML configuration
    file. The NewFarm will create a heterogenous set of turbines that compose a windfarm,
    validate the inputs, and then create a vectorized representation of the the turbine
    data.

    Farm is the container class of the FLORIS package. It brings
    together all of the component objects after input (i.e., Turbine,
    Wake, FlowField) and packages everything into the appropriate data
    type. Farm should also be used as an entry point to probe objects
    for generating output.

    Args:
        turbine_id (List[str]): The turbine identifiers to map each turbine to one of
            the turbine classifications in `turbine_map`.
        turbine_map (Dict[str, Union[dict, Turbine]]): The dictionary mapping of unique
            turbines at the wind power plant. Takes either a pre-generated `Turbine`
            object, or a dictionary that will be used to generate the `Turbine` object.
        layout_x (Union[List[float], np.ndarray]): The x-coordinates for the turbines at
            the wind power plant.
        layout_y (Union[List[float], np.ndarray]): The y-coordinates for the turbines at
            the wind power plant.
        wtg (List[str]): The WTG ID values for each turbine. This field acts as metadata
            only.

    Raises:
        ValueError: [description]
        ValueError: [description]

    Returns:
        [type]: [description]
    """

    # TODO: Create the mapping of turbines to a farm
    # TODO: Add an ID filed to the turbines
    # TODO: create a vectorized implementation of the turbine data

    turbine_id: List[str] = attr.ib(validator=iter_validator(list, str))
    turbine_map: Dict[str, Union[dict, Turbine]] = attr.ib(converter=create_turbines)
    layout_x: Union[List[float], np.ndarray] = attr.ib(converter=attrs_array_converter)
    layout_y: Union[List[float], np.ndarray] = attr.ib(converter=attrs_array_converter)
    wtg_id: List[str] = attr.ib(
        factory=list,
        on_setattr=attr.setters.validate,
        validator=iter_validator(list, str),
    )

    coordinates: List[Vec3] = attr.ib(init=False)

    rotor_diameter: np.ndarray = attr.ib(init=False)
    hub_height: np.ndarray = attr.ib(init=False)
    pP: np.ndarray = attr.ib(init=False)
    pT: np.ndarray = attr.ib(init=False)
    generator_efficiency: np.ndarray = attr.ib(init=False)
    # power_thrust_table: np.ndarray = attr.ib(init=False)  # NOTE: Is this only necessary for the creation of the interpolations?
    fCp_interp: np.ndarray = attr.ib(init=False)
    fCt_interp: np.ndarray = attr.ib(init=False)
    power_interp: np.ndarray = attr.ib(init=False)
    rotor_radius: np.ndarray = attr.ib(init=False)
    rotor_area: np.ndarray = attr.ib(init=False)
    array_data: xr.DataArray = attr.ib(init=False)

    # Pre multi-turbine
    # i  j  k  l  m
    # wd ws x  y  z

    # With multiple turbines per floris ez (aka Chris)
    # i  j  k    l  m  n
    # wd ws t_ix x  y  z

    def __attrs_post_init__(self) -> None:
        self.coordinates = [
            Vec3([x, y, self.turbine_map[t_id].hub_height])
            for x, y, t_id in zip(self.layout_x, self.layout_y, self.turbine_id)
        ]

        self.generate_farm_points()

    @layout_x.validator
    def check_x_len(self, instance: str, value: Union[List[float], np.ndarray]) -> None:
        if len(value) < len(self.turbine_id):
            raise ValueError("Not enough `layout_x` values to match the `turbine_id`s")
        if len(value) > len(self.turbine_id):
            raise ValueError("Too many `layout_x` values to match the `turbine_id`s")

    @layout_y.validator
    def check_y_len(self, instance: str, value: Union[List[float], np.ndarray]) -> None:
        if len(value) < len(self.turbine_id):
            raise ValueError("Not enough `layout_y` values to match the `turbine_id`s")
        if len(value) > len(self.turbine_id):
            raise ValueError("Too many `layout_y` values to match the `turbine_id`s")

    @wtg_id.validator
    def check_wtg_id(self, instance: str, value: Union[list, List[str]]) -> None:
        if len(value) == 0:
            self.wtg_id = [
                f"t{str(i).zfill(3)}" for i in 1 + np.arange(len(self.turbine_id))
            ]
        elif len(value) < len(self.turbine_id):
            raise ValueError("There are too few `wtg_id` values")
        elif len(value) > len(self.turbine_id):
            raise ValueError("There are too many `wtg_id` values")

    def generate_farm_points(self) -> None:
        # Create an array of turbine values and the column ordering
        arbitrary_turbine = self.turbine_map[self.turbine_id[0]]
        column_order = generate_turbine_attribute_order(arbitrary_turbine)
        turbine_array = np.array(
            [generate_turbine_tuple(self.turbine_map[t_id]) for t_id in self.turbine_id]
        )

        column_ix = {col: i for i, col in enumerate(column_order)}
        self.rotor_diameter = turbine_array[:, column_ix["rotor_diameter"]]
        self.rotor_radius = turbine_array[:, column_ix["rotor_radius"]]
        self.rotor_area = turbine_array[:, column_ix["rotor_area"]]
        self.hub_height = turbine_array[:, column_ix["hub_height"]]
        self.pP = turbine_array[:, column_ix["pP"]]
        self.pT = turbine_array[:, column_ix["pT"]]
        self.generator_efficiency = turbine_array[:, column_ix["generator_efficiency"]]
        self.fCp_interp = turbine_array[:, column_ix["fCp_interp"]]
        self.fCt_interp = turbine_array[:, column_ix["fCt_interp"]]
        self.power_interp = turbine_array[:, column_ix["power_interp"]]

        self.data_array = xr.DataArray(
            turbine_array,
            coords=dict(wtg_id=self.wtg_id, turbine_attributes=column_order),
        )


class FarmController:
    def __init__(self, n_wind_speeds: int, n_wind_directions: int) -> None:
        # TODO: This should hold the yaw settings for each turbine for each wind speed and wind direction

        # Initialize the yaw settings to an empty array
        # self.set_yaw_angles(np.array((0,)))
        pass
    
    def set_yaw_angles(self, yaw_angles: Union[list, np.ndarray]) -> None:
        self.yaw_angles = yaw_angles


class Farm:
    """
    Farm is a class containing the objects that make up a FLORIS model.

    Farm is the container class of the FLORIS package. It brings
    together all of the component objects after input (i.e., Turbine,
    Wake, FlowField) and packages everything into the appropriate data
    type. Farm should also be used as an entry point to probe objects
    for generating output.
    """

    def __init__(self, input_dictionary: dict, turbine: Turbine):
        """
        Args:
            input_dictionary (dict): The required keys in this dictionary
                are:

                    -   **wind_speed** (*list*): The wind speed measurements at
                        hub height (m/s).
                    -   **wind_x** (*list*): The x-coordinates of the wind
                        speed measurements.
                    -   **wind_y** (*list*): The y-coordinates of the wind
                        speed measurements.
                    -   **wind_direction** (*list*): The wind direction
                        measurements (deg).
                    -   **turbulence_intensity** (*list*): Turbulence intensity
                        measurements at hub height (as a decimal fraction).
                    -   **wind_shear** (*float*): The power law wind shear
                        exponent.
                    -   **wind_veer** (*float*): The vertical change in wind
                        direction across the rotor.
                    -   **air_density** (*float*): The air density (kg/m^3).
                    -   **layout_x** (*list*): The x-coordinates of the
                        turbines.
                    -   **layout_y** (*list*): The y-coordinates of the
                        turbines.

            turbine (:py:obj:`~.turbine.Turbine`): The turbine models used
                throughout the farm.
            wake (:py:obj:`~.wake.Wake`): The wake model used to simulate the
                freestream flow and wakes.
        """
        """
        Converts input coordinates into :py:class:`~.utilities.Vec3` and
        constructs the underlying mapping to :py:class:`~.turbine.Turbine`.
        It is assumed that all arguments are of the same length and that the
        Turbine at a particular index corresponds to the coordinate at the same
        index in the layout arguments.

        Args:
            layout_x ( list(float) ): X-coordinate of the turbine locations.
            layout_y ( list(float) ): Y-coordinate of the turbine locations.
            turbines ( list(float) ): Turbine objects corresponding to
                the locations given in layout_x and layout_y.
        """
        layout_x = input_dictionary["layout_x"]
        layout_y = input_dictionary["layout_y"]

        # check if the length of x and y coordinates are equal
        if len(layout_x) != len(layout_y):
            err_msg = (
                "The number of turbine x locations ({0}) is "
                + "not equal to the number of turbine y locations "
                + "({1}). Please check your layout array."
            ).format(len(layout_x), len(layout_y))
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)

        coordinates = [
            Vec3([x1, x2, turbine.hub_height])
            for x1, x2 in list(zip(layout_x, layout_y))
        ]
        self.turbine_map_dict = {c: copy.deepcopy(turbine) for c in coordinates}

        # Turbine control settings indexed by the turbine ID
        self.farm_controller = FarmController(len(input_dictionary["wind_speeds"]), 1)
        self.farm_controller.set_yaw_angles( np.zeros( ( len(self.turbine_map_dict)) ) )

    def sorted_in_x_as_list(self):
        """
        Sorts the turbines based on their x-coordinates in ascending order.

        Returns:
            list((:py:class:`~.utilities.Vec3`, :py:class:`~.turbine.Turbine`)):
            The sorted coordinates and corresponding turbines. This is a
            list of tuples where each tuple contains the coordinate
            and turbine in the first and last element, respectively.
        """
        coords = sorted(self.turbine_map_dict, key=lambda coord: coord.x1)
        return [(c, self.turbine_map_dict[c]) for c in coords]

    @property
    def turbines(self):
        """
        Turbines contained in the :py:class:`~.turbine_map.TurbineMap`.

        Returns:
            list(:py:class:`floris.simulation.turbine.Turbine`)
        """
        return [turbine for _, turbine in self.items]

    @property
    def coords(self):
        """
        Coordinates of the turbines contained in the
        :py:class:`~.turbine_map.TurbineMap`.

        Returns:
            list(:py:class:`~.utilities.Vec3`)
        """
        return [coord for coord, _ in self.items]

    @property
    def items(self):
        """
        Contents of the internal Python dictionary mapping of the turbine
        and coordinates.

        Returns:
            dict_items: Iterable object containing tuples of key-value pairs
            where the first index is the coordinate
            (:py:class:`~.utilities.Vec3`) and the second index is the
            :py:class:`~.turbine.Turbine`.
        """
        return self.turbine_map_dict.items()

    def set_yaw_angles(
        self, yaw_angles: list, n_wind_speeds: int, n_wind_directions: int
    ) -> None:
        if len(yaw_angles) != len(self.items):
            raise ValueError("Farm.set_yaw_angles: a yaw angle must be given for each turbine.")

        # TODO: support a user-given yaw angle setting for each wind speed and wind direction
        # self.farm_controller.set_yaw_angles(
        #     np.reshape(
        #         np.array([yaw_angles] * n_wind_speeds),  # broadcast
        #         (len(self.items), n_wind_speeds)  # reshape
        #     )
        # )
        self.farm_controller.set_yaw_angles(np.array(yaw_angles))
