
import copy
from pathlib import Path

import numpy as np

from floris.tools import FlorisInterface


class UncertaintyInterface(FlorisInterface):
    def __init__(
        self,
        configuration: dict | str | Path,
    ):
        # Call base init function
        super().__init__(configuration)  # Call the parent's __init__

    def set_uncertain(
        self, wd_resolution=1, wd_std=3.0, wd_sample_points=np.array([-6, -3, 3, 6.0])
    ):
        # Save these values
        self.wd_resolution = wd_resolution
        self.wd_std = wd_std
        self.wd_sample_points = wd_sample_points

        # Grab the unexpanded values of all arrays
        self.yaw_angles_unexpanded = self.floris.farm.yaw_angles
        self.power_setpoints_unexpanded = self.floris.farm.power_setpoints

        self.wind_directions_unexpanded = self.floris.flow_field.wind_directions
        self.wind_speeds_unexpanded = self.floris.flow_field.wind_speeds
        self.turbulence_intensities_unexpanded = self.floris.flow_field.turbulence_intensities

        self.n_unexpanded = len(self.wind_directions_unexpanded)

        # self.floris.farm.n_turbines
        print(self.n_unexpanded)

        # Combine into the complete unexpanded input
        self.unexpanded_input = np.hstack(
            (
                self.wind_directions_unexpanded[:, np.newaxis],
                self.wind_speeds_unexpanded[:, np.newaxis],
                self.turbulence_intensities_unexpanded[:, np.newaxis],
                self.yaw_angles_unexpanded,
                self.power_setpoints_unexpanded,
            )
        )

        # Determine the expanded input

        # floris_dict = self.floris.as_dict()
        # flow_field_dict = floris_dict["flow_field"]

        # self.yaw_angles_unexpanded
        # if wind_speeds is not None:
        #     flow_field_dict["wind_speeds"] = wind_speeds
        # if wind_directions is not None:
        #     flow_field_dict["wind_directions"] = wind_directions
        # if wind_shear is not None:
        #     flow_field_dict["wind_shear"] = wind_shear
        # if wind_veer is not None:
        #     flow_field_dict["wind_veer"] = wind_veer
        # if reference_wind_height is not None:
        #     flow_field_dict["reference_wind_height"] = reference_wind_height
        # if turbulence_intensities is not None:
        #     flow_field_dict["turbulence_intensities"] = turbulence_intensities
        # if air_density is not None:
        #     flow_field_dict["air_density"] = air_density
        # if heterogenous_inflow_config is not None:
        #     flow_field_dict["heterogenous_inflow_config"] = heterogenous_inflow_config

    def _expand_wind_directions(self):
        pass
