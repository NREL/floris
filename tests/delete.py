import math
from copy import deepcopy
from typing import List

import attr
import numpy as np
import xarray as xr

from floris.turbine import Turbine, PowerThrustTable
from floris.utilities import float_attrib, iter_validator


turbine_data = {
    "rotor_diameter": 126.0,
    "hub_height": 90.0,
    "pP": 1.88,
    "pT": 1.88,
    "generator_efficiency": 1.0,
    "power_thrust_table": {
        "power": [
            0.0,
            0.0,
            0.1780851,
            0.28907459,
            0.34902166,
            0.3847278,
            0.40605878,
            0.4202279,
            0.42882274,
            0.43387274,
            0.43622267,
            0.43684468,
            0.43657497,
            0.43651053,
            0.4365612,
            0.43651728,
            0.43590309,
            0.43467276,
            0.43322955,
            0.43003137,
            0.37655587,
            0.33328466,
            0.29700574,
            0.26420779,
            0.23839379,
            0.21459275,
            0.19382354,
            0.1756635,
            0.15970926,
            0.14561785,
            0.13287856,
            0.12130194,
            0.11219941,
            0.10311631,
            0.09545392,
            0.08813781,
            0.08186763,
            0.07585005,
            0.07071926,
            0.06557558,
            0.06148104,
            0.05755207,
            0.05413366,
            0.05097969,
            0.04806545,
            0.04536883,
            0.04287006,
            0.04055141,
        ],
        "thrust": [
            1.19187945,
            1.17284634,
            1.09860817,
            1.02889592,
            0.97373036,
            0.92826162,
            0.89210543,
            0.86100905,
            0.835423,
            0.81237673,
            0.79225789,
            0.77584769,
            0.7629228,
            0.76156073,
            0.76261984,
            0.76169723,
            0.75232027,
            0.74026851,
            0.72987175,
            0.70701647,
            0.54054532,
            0.45509459,
            0.39343381,
            0.34250785,
            0.30487242,
            0.27164979,
            0.24361964,
            0.21973831,
            0.19918151,
            0.18131868,
            0.16537679,
            0.15103727,
            0.13998636,
            0.1289037,
            0.11970413,
            0.11087113,
            0.10339901,
            0.09617888,
            0.09009926,
            0.08395078,
            0.0791188,
            0.07448356,
            0.07050731,
            0.06684119,
            0.06345518,
            0.06032267,
            0.05741999,
            0.05472609,
        ],
        "wind_speed": [
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            5.5,
            6.0,
            6.5,
            7.0,
            7.5,
            8.0,
            8.5,
            9.0,
            9.5,
            10.0,
            10.5,
            11.0,
            11.5,
            12.0,
            12.5,
            13.0,
            13.5,
            14.0,
            14.5,
            15.0,
            15.5,
            16.0,
            16.5,
            17.0,
            17.5,
            18.0,
            18.5,
            19.0,
            19.5,
            20.0,
            20.5,
            21.0,
            21.5,
            22.0,
            22.5,
            23.0,
            23.5,
            24.0,
            24.5,
            25.0,
            25.5,
        ],
    },
}
# turbine = Turbine.from_dict(turbine_data)
# turbine_list = [deepcopy(turbine) for _ in range(5)]

# print(attr.asdict(turbine, filter=attr.filters.exclude(attr.fields(Turbine).power_thrust_table, PowerThrustTable)))

# arr = np.array([
#     attr.astuple(t, filter=lambda attribute, value: attribute.name not in ("power_thrust_table", "model_string"))
#     for t in turbine_list
# ])
# print(arr.shape)
# print(arr)
# # print()
# # print(attr.astuple(turbine, filter=lambda attribute, value: attribute.name not in ("power_thrust_table", "model_string")))

# cols = attr.asdict(turbine, filter=lambda attribute, value: attribute.name not in ("power_thrust_table", "model_string"))
# cols = [*cols]
# wtg_id = range(arr.shape[0])
# new_arr = xr.DataArray(arr, coords=dict(wtg_id=wtg_id, turbine_attributes=cols))
# print(new_arr)


@attr.s(auto_attribs=True)
class A:
    x: float = float_attrib()
    y: float = float_attrib(init=False)
    z: float = float_attrib(init=False)
    wtg_id: List[str] = attr.ib(factory=list, validator=iter_validator(list, str))

    def __attrs_post_init__(self) -> None:
        self.y = self.x / 2
        self.z = np.pi * self.y ** 2

    @wtg_id.validator
    def wtg_id_if_none(self, instance, value):
        if len(value) == 0:
            self.wtg_id = [f"t{str(i).zfill(3)}" for i in range(5)]

    @x.validator
    def set_dependencies(self, instance, value) -> None:
        attr.set_run_validators(False)
        self.y = value / 2
        self.z = np.pi * self.y ** 2
        attr.set_run_validators(True)

    @y.validator
    def set_y(self, instance, value) -> None:
        self.x = value * 2

    @z.validator
    def set_z(self, instance, value) -> None:
        self.y = math.sqrt(value / np.pi)


a = A(x=2.0)
print(a)

a.x = 4.0
print(a)

a.y = 3.0
print(a)

a.z = 49
print(a)

a = A(x=15, wtg_id=["DAW_001", "DJT_001"])
print(a)

a = A(x=15, wtg_id=[1, 2, 3, 4])
print(a)

print(attr.asdict(a, filter=attr.filters.include(a.y, float)))
