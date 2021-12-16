from typing import List, Union

import attr
import numpy as np

from floris.utilities import Vec3, float_attrib, model_attrib
from floris.base_class import BaseClass


@attr.s(auto_attribs=True)
class Curl(BaseClass):
    """The Curl model class computes the wake velocity deficit based on the curled
    wake model developed in
    :cite:`cvm-martinez2019aerodynamics`. The curled wake
    model includes the change in the shape of the wake profile under yawed
    conditions due to vortices that are shed from the rotor plane of a yawed
    turbine. For more information about the curled wake model theory, see
    :cite:`cvm-martinez2019aerodynamics`. For more
    information about the impact of the curled wake behavior on wake steering,
    see :cite:`cvm-fleming2018simulation`.

    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: cvm-

    Args:
        model_grid_resolution(:py:obj:`Union[List[float], Vec3]`): A list of three
            floats, or `Vec3` object that define the flow field grid resolution in the
            x, y, and z directions used for the curl wake model calculations. The grid
            resolution is specified as the number of grid points in the flow field
            domain in the x, y, and z directions, by default [250, 100, 75].
        initial_deficit (:py:obj:`float`): Parameter that, along with the freestream
            velocity and the turbine's induction factor, is used to determine the
            initial wake velocity deficit immediately downstream of the rotor, by
            default 2.0.
        dissipation (:py:obj:`float`): A scaling parameter that determines the amount of
            dissipation of the vortices with downstream distance, by default 0.06.
        veer_linear (:py:obj:`float`): Describes the amount of linear wind veer. This
            parameter defines the linear change in the V velocity between the ground and
            hub height; therefore, determines the slope of the change in the V velocity
            with height, by default 0.0.
        ti_initial (:py:obj:`float`): Initial ambient turbulence intensity, expressed as
            a decimal fraction, by default 0.1.
        ti_constant (:py:obj:`float`): The constant used to scale the wake-added
            turbulence intensity, by default 0.73.
        ti_ai (:py:obj:`float`): The axial induction factor exponent used in in the
            calculation of wake-added turbulence, by default 0.8.
        ti_downstream (:py:obj:`float`): The exponent applied to the distance downstream
            of an upstream turbine normalized by the rotor diameter used in the
            calculation of wake-added turbulence, by default 0.275.
    """

    model_grid_resolution: Union[List[float], Vec3] = attr.ib(
        default=[250, 100, 75],
        # converter=convert_to_Vec3,
        on_setattr=attr.setters.validate,
        kw_only=True,
    )
    initial_deficit: float = float_attrib(default=2.0)
    dissipation: float = float_attrib(default=0.06)
    veer_linear: float = float_attrib(default=0.0)
    ti_initial: float = float_attrib(default=0.1)
    ti_constant: float = float_attrib(default=0.73)
    ti_ai: float = float_attrib(default=0.8)
    ti_downstream: float = float_attrib(default=-0.275)
    requires_resolution: bool = True
    model_string: str = model_attrib(default="curl")

    def function(
        self,
        x_locations: np.ndarray,
        y_locations: np.ndarray,
        z_locations: np.ndarray,
        turbine: np.ndarray,  # we'll see here
        turbine_coord: Vec3,
        deflection_field: np.ndarray,
        # flow_filed shall be deprecated for the following
        grid_wind_sped: np.ndarray,  # flow_field.wind_map.grid_wind_speed
        u: float,  # from flow_field
        v: float,  # from flow_field
        w: float,  # from flow_field
        air_density: float,  # flow_field
        x: float,  # flow_field
        y: float,  # flow_field
        z: float,  # flow_field
        grid_turbulence_intensity: np.ndarray,  # flow_field.wind_map_grid_turbulence_intensity
    ) -> None:
        # Probably picked the wrong model to start with, but I think I'll move over to
        # others and see where this goes and update as I see fit.
        pass


if __name__ == "__main__":
    # Run some demonstrations of the Curl model setup

    # Demonstrate the model works and show the __repr__
    c = Curl()
    print(c)

    # Demonstrate that a Vec3 object can also be passed in
    v = Vec3([25, 10, 7.5])
    c = Curl(model_grid_resolution=v)
    print(c)

    # Demonstrate the default extraction helper classmethod works (from FromDictMixin)
    print(Curl.get_model_defaults())

    # Demonstrate that you the from_dict classmethod works (from FromDictMixin)
    # This raises an error message because we've attempted to set the model_string
    # parameter as anything but the default
    input = {
        "model_grid_resolution": [250, 100, 75],
        "initial_deficit": 2.0,
        "dissipation": 0.06,
        "veer_linear": 0.0,
        "initial": 0.1,
        "constant": 0.73,
        "ai": 0.8,
        "downstream": -0.275,
        "model_string": "anything",
    }
    c = Curl.from_dict(input)
    print(c)
