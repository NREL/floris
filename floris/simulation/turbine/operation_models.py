# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

from __future__ import annotations

import copy
from abc import abstractmethod
from typing import (
    Any,
    Dict,
    Final,
)

import numpy as np
from attrs import define, field
from scipy.interpolate import interp1d

from floris.simulation import BaseClass
from floris.simulation.rotor_velocity import (
    average_velocity,
    compute_tilt_angles_for_floating_turbines,
    rotor_velocity_tilt_correction,
    rotor_velocity_yaw_correction,
)
from floris.type_dec import (
    NDArrayFloat,
    NDArrayObject,
)
from floris.utilities import cosd


POWER_SETPOINT_DEFAULT = 1e12

def rotor_velocity_air_density_correction(
    velocities: NDArrayFloat,
    air_density: float,
    ref_air_density: float,
) -> NDArrayFloat:
    # Produce equivalent velocities at the reference air density
    # TODO: This could go on BaseTurbineModel
    return (air_density/ref_air_density)**(1/3) * velocities


@define
class BaseOperationModel(BaseClass):
    """
    Base class for turbine operation models. All turbine operation models must implement static
    power(), thrust_coefficient(), and axial_induction() methods, which are called by power() and
    thrust_coefficient() through the interface in the turbine.py module.

    Args:
        BaseClass (_type_): _description_

    Raises:
        NotImplementedError: _description_
        NotImplementedError: _description_
    """
    @staticmethod
    @abstractmethod
    def power() -> None:
        raise NotImplementedError("BaseOperationModel.power")

    @staticmethod
    @abstractmethod
    def thrust_coefficient() -> None:
        raise NotImplementedError("BaseOperationModel.thrust_coefficient")

    @staticmethod
    @abstractmethod
    def axial_induction() -> None:
        raise NotImplementedError("BaseOperationModel.axial_induction")

@define
class SimpleTurbine(BaseOperationModel):
    """
    Static class defining an actuator disk turbine model that is fully aligned with the flow. No
    handling for yaw or tilt angles.

    As with all turbine submodules, implements only static power() and thrust_coefficient() methods,
    which are called by power() and thrust_coefficient() on turbine.py, respectively. This class is
    not intended to be instantiated; it simply defines a library of static methods.

    TODO: Should the turbine submodels each implement axial_induction()?
    """

    def power(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        air_density: float,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        **_ # <- Allows other models to accept other keyword arguments
    ):
        # Construct power interpolant
        power_interpolator = interp1d(
            power_thrust_table["wind_speed"],
            power_thrust_table["power"],
            fill_value=0.0,
            bounds_error=False,
        )

        # Compute the power-effective wind speed across the rotor
        rotor_average_velocities = average_velocity(
            velocities=velocities,
            method=average_method,
            cubature_weights=cubature_weights,
        )

        rotor_effective_velocities = rotor_velocity_air_density_correction(
            velocities=rotor_average_velocities,
            air_density=air_density,
            ref_air_density=power_thrust_table["ref_air_density"]
        )

        # Compute power
        power = power_interpolator(rotor_effective_velocities) * 1e3 # Convert to W

        return power

    def thrust_coefficient(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        **_ # <- Allows other models to accept other keyword arguments
    ):
        # Construct thrust coefficient interpolant
        thrust_coefficient_interpolator = interp1d(
            power_thrust_table["wind_speed"],
            power_thrust_table["thrust_coefficient"],
            fill_value=0.0001,
            bounds_error=False,
        )

        # Compute the effective wind speed across the rotor
        rotor_average_velocities = average_velocity(
            velocities=velocities,
            method=average_method,
            cubature_weights=cubature_weights,
        )

        # TODO: Do we need an air density correction here?

        thrust_coefficient = thrust_coefficient_interpolator(rotor_average_velocities)
        thrust_coefficient = np.clip(thrust_coefficient, 0.0001, 0.9999)

        return thrust_coefficient

    def axial_induction(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        **_ # <- Allows other models to accept other keyword arguments
    ):

        thrust_coefficient = SimpleTurbine.thrust_coefficient(
            power_thrust_table=power_thrust_table,
            velocities=velocities,
            average_method=average_method,
            cubature_weights=cubature_weights,
        )

        return (1 - np.sqrt(1 - thrust_coefficient))/2


@define
class CosineLossTurbine(BaseOperationModel):
    """
    Static class defining an actuator disk turbine model that may be misaligned with the flow.
    Nonzero tilt and yaw angles are handled via cosine relationships, with the power lost to yawing
    defined by the pP exponent. This turbine submodel is the default, and matches the turbine
    model in FLORIS v3.

    As with all turbine submodules, implements only static power() and thrust_coefficient() methods,
    which are called by power() and thrust_coefficient() on turbine.py, respectively. This class is
    not intended to be instantiated; it simply defines a library of static methods.

    TODO: Should the turbine submodels each implement axial_induction()?
    """

    def power(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        air_density: float,
        yaw_angles: NDArrayFloat,
        tilt_angles: NDArrayFloat,
        tilt_interp: NDArrayObject,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        correct_cp_ct_for_tilt: bool = False,
        **_ # <- Allows other models to accept other keyword arguments
    ):
        # Construct power interpolant
        power_interpolator = interp1d(
            power_thrust_table["wind_speed"],
            power_thrust_table["power"],
            fill_value=0.0,
            bounds_error=False,
        )

        # Compute the power-effective wind speed across the rotor
        rotor_average_velocities = average_velocity(
            velocities=velocities,
            method=average_method,
            cubature_weights=cubature_weights,
        )

        rotor_effective_velocities = rotor_velocity_air_density_correction(
            velocities=rotor_average_velocities,
            air_density=air_density,
            ref_air_density=power_thrust_table["ref_air_density"]
        )

        rotor_effective_velocities = rotor_velocity_yaw_correction(
            pP=power_thrust_table["pP"],
            yaw_angles=yaw_angles,
            rotor_effective_velocities=rotor_effective_velocities,
        )

        rotor_effective_velocities = rotor_velocity_tilt_correction(
            tilt_angles=tilt_angles,
            ref_tilt=power_thrust_table["ref_tilt"],
            pT=power_thrust_table["pT"],
            tilt_interp=tilt_interp,
            correct_cp_ct_for_tilt=correct_cp_ct_for_tilt,
            rotor_effective_velocities=rotor_effective_velocities,
        )

        # Compute power
        power = power_interpolator(rotor_effective_velocities) * 1e3 # Convert to W

        return power

    def thrust_coefficient(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        yaw_angles: NDArrayFloat,
        tilt_angles: NDArrayFloat,
        tilt_interp: NDArrayObject,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        correct_cp_ct_for_tilt: bool = False,
        **_ # <- Allows other models to accept other keyword arguments
    ):
        # Construct thrust coefficient interpolant
        thrust_coefficient_interpolator = interp1d(
            power_thrust_table["wind_speed"],
            power_thrust_table["thrust_coefficient"],
            fill_value=0.0001,
            bounds_error=False,
        )

        # Compute the effective wind speed across the rotor
        rotor_average_velocities = average_velocity(
            velocities=velocities,
            method=average_method,
            cubature_weights=cubature_weights,
        )

        # TODO: Do we need an air density correction here?
        thrust_coefficient = thrust_coefficient_interpolator(rotor_average_velocities)
        thrust_coefficient = np.clip(thrust_coefficient, 0.0001, 0.9999)

        # Apply tilt and yaw corrections
        # Compute the tilt, if using floating turbines
        old_tilt_angles = copy.deepcopy(tilt_angles)
        tilt_angles = compute_tilt_angles_for_floating_turbines(
            tilt_angles=tilt_angles,
            tilt_interp=tilt_interp,
            rotor_effective_velocities=rotor_average_velocities,
        )
        # Only update tilt angle if requested (if the tilt isn't accounted for in the Ct curve)
        tilt_angles = np.where(correct_cp_ct_for_tilt, tilt_angles, old_tilt_angles)

        thrust_coefficient = (
            thrust_coefficient
            * cosd(yaw_angles)
            * cosd(tilt_angles - power_thrust_table["ref_tilt"])
        )

        return thrust_coefficient

    def axial_induction(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        yaw_angles: NDArrayFloat,
        tilt_angles: NDArrayFloat,
        tilt_interp: NDArrayObject,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        correct_cp_ct_for_tilt: bool = False,
        **_ # <- Allows other models to accept other keyword arguments
    ):

        thrust_coefficient = CosineLossTurbine.thrust_coefficient(
            power_thrust_table=power_thrust_table,
            velocities=velocities,
            yaw_angles=yaw_angles,
            tilt_angles=tilt_angles,
            tilt_interp=tilt_interp,
            average_method=average_method,
            cubature_weights=cubature_weights,
            correct_cp_ct_for_tilt=correct_cp_ct_for_tilt
        )

        misalignment_loss = cosd(yaw_angles) * cosd(tilt_angles - power_thrust_table["ref_tilt"])
        return 0.5 / misalignment_loss * (1 - np.sqrt(1 - thrust_coefficient * misalignment_loss))

@define
class SimpleDeratingTurbine(BaseOperationModel):
    """
    power_thrust_table is a dictionary (normally defined on the turbine input yaml)
    that contains the parameters necessary to evaluate power(), thrust(), and axial_induction().
    Any specific parameters for derating can be placed here. (they can be added to the turbine
    yaml). For this operation model to receive those arguements, they'll need to be
    added to the kwargs dictionaries in the respective functions on turbine.py. They won't affect
    the other operation models.
    """
    def power(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        air_density: float,
        power_setpoints: NDArrayFloat | None,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        **_ # <- Allows other models to accept other keyword arguments
    ):
        base_powers = SimpleTurbine.power(
            power_thrust_table=power_thrust_table,
            velocities=velocities,
            air_density=air_density,
            average_method=average_method,
            cubature_weights=cubature_weights
        )
        if power_setpoints is None:
            return base_powers
        else:
            return np.minimum(base_powers, power_setpoints)

        # TODO: would we like special handling of zero power setpoints
        # (mixed with non-zero values) to speed up computation in that case?

    def thrust_coefficient(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        air_density: float,
        power_setpoints: NDArrayFloat,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        **_ # <- Allows other models to accept other keyword arguments
    ):
        base_thrust_coefficients = SimpleTurbine.thrust_coefficient(
            power_thrust_table=power_thrust_table,
            velocities=velocities,
            average_method=average_method,
            cubature_weights=cubature_weights
        )
        if power_setpoints is None:
            return base_thrust_coefficients
        else:
            # Assume thrust coefficient scales directly with power
            base_powers = SimpleTurbine.power(
                power_thrust_table=power_thrust_table,
                velocities=velocities,
                air_density=air_density
            )
            power_fractions = power_setpoints / base_powers
            thrust_coefficients = power_fractions * base_thrust_coefficients
            return np.minimum(base_thrust_coefficients, thrust_coefficients)

    def axial_induction(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        air_density: float,
        power_setpoints: NDArrayFloat,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        **_ # <- Allows other models to accept other keyword arguments
    ):
        thrust_coefficient = SimpleDeratingTurbine.thrust_coefficient(
            power_thrust_table=power_thrust_table,
            velocities=velocities,
            air_density=air_density,
            power_setpoints=power_setpoints,
            average_method=average_method,
            cubature_weights=cubature_weights,
        )

        return (1 - np.sqrt(1 - thrust_coefficient))/2

@define
class MixedOperationTurbine(BaseOperationModel):

    def power(
        yaw_angles: NDArrayFloat,
        power_setpoints: NDArrayFloat,
        **kwargs
    ):
        yaw_angles_mask = yaw_angles > 0
        power_setpoints_mask = power_setpoints < POWER_SETPOINT_DEFAULT
        neither_mask = np.logical_not(yaw_angles_mask) & np.logical_not(power_setpoints_mask)

        if (power_setpoints_mask & yaw_angles_mask).any():
            raise ValueError((
                "Power setpoints and yaw angles are incompatible."
                "If yaw_angles entry is nonzero, power_setpoints must be greater than"
                " or equal to {0}.".format(POWER_SETPOINT_DEFAULT)
            ))

        powers = np.zeros_like(power_setpoints)
        powers[yaw_angles_mask] += CosineLossTurbine.power(
            yaw_angles=yaw_angles,
            **kwargs
        )[yaw_angles_mask]
        powers[power_setpoints_mask] += SimpleDeratingTurbine.power(
            power_setpoints=power_setpoints,
            **kwargs
        )[power_setpoints_mask]
        powers[neither_mask] += SimpleTurbine.power(
            **kwargs
        )[neither_mask]

        return powers

    def thrust_coefficient(
        yaw_angles: NDArrayFloat,
        power_setpoints: NDArrayFloat,
        **kwargs
    ):
        yaw_angles_mask = yaw_angles > 0
        power_setpoints_mask = power_setpoints < POWER_SETPOINT_DEFAULT
        neither_mask = np.logical_not(yaw_angles_mask) & np.logical_not(power_setpoints_mask)

        if (power_setpoints_mask & yaw_angles_mask).any():
            raise ValueError((
                "Power setpoints and yaw angles are incompatible."
                "If yaw_angles entry is nonzero, power_setpoints must be greater than"
                " or equal to {0}.".format(POWER_SETPOINT_DEFAULT)
            ))

        thrust_coefficients = np.zeros_like(power_setpoints)
        thrust_coefficients[yaw_angles_mask] += CosineLossTurbine.thrust_coefficient(
            yaw_angles=yaw_angles,
            **kwargs
        )[yaw_angles_mask]
        thrust_coefficients[power_setpoints_mask] += SimpleDeratingTurbine.thrust_coefficient(
            power_setpoints=power_setpoints,
            **kwargs
        )[power_setpoints_mask]
        thrust_coefficients[neither_mask] += SimpleTurbine.thrust_coefficient(
            **kwargs
        )[neither_mask]

        return thrust_coefficients

    def axial_induction(
        yaw_angles: NDArrayFloat,
        power_setpoints: NDArrayFloat,
        **kwargs
    ):
        yaw_angles_mask = yaw_angles > 0
        power_setpoints_mask = power_setpoints < POWER_SETPOINT_DEFAULT
        neither_mask = np.logical_not(yaw_angles_mask) & np.logical_not(power_setpoints_mask)

        if (power_setpoints_mask & yaw_angles_mask).any():
            raise ValueError((
                "Power setpoints and yaw angles are incompatible."
                "If yaw_angles entry is nonzero, power_setpoints must be greater than"
                " or equal to {0}.".format(POWER_SETPOINT_DEFAULT)
            ))

        axial_inductions = np.zeros_like(power_setpoints)
        axial_inductions[yaw_angles_mask] += CosineLossTurbine.axial_induction(
            yaw_angles=yaw_angles,
            **kwargs
        )[yaw_angles_mask]
        axial_inductions[power_setpoints_mask] += SimpleDeratingTurbine.axial_induction(
            power_setpoints=power_setpoints,
            **kwargs
        )[power_setpoints_mask]
        axial_inductions[neither_mask] += SimpleTurbine.axial_induction(
            **kwargs
        )[neither_mask]

        return axial_inductions
