from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import (
    LineString,
    Point,
    Polygon,
)

from .layout_optimization_base import LayoutOptimization
from .layout_optimization_random_search import test_point_in_bounds


class LayoutOptimizationGridded(LayoutOptimization):
    def __init__(
        self,
        fmodel,
        boundaries,
        spacing: float | None = None,
        spacing_D: float | None = -1,
        rotation_step: float = 5.0,
        rotation_range: tuple[float, float] = (0.0, 360.0),
        translation_step: float | None = None,
        translation_step_D: float | None = -1,
        translation_range: tuple[float, float] | None = None,
        hexagonal_packing: bool = False,
        enable_geometric_yaw: bool = False,
        use_value: bool=False,

    ):
        # Save boundaries
        # Handle spacing information
        if spacing is not None and spacing_D is not None and spacing_D >= 0:
            raise ValueError("Only one of spacing and spacing_D can be defined.")
        if spacing is None and spacing_D is None:
            raise ValueError("Either spacing or spacing_D must be defined.")
        if spacing_D is not None and spacing is None:
            if spacing_D < 0: # Default to 5D
                spacing_D = 5.0
            spacing = spacing_D * fmodel.core.farm.rotor_diameters[0]
            if len(np.unique(fmodel.core.farm.rotor_diameters)) > 1:
                self.logger.warning((
                    "Found multiple turbine diameters. Using diameter of first turbine to set"
                    f" spacing to {spacing}m ({spacing_D} diameters)."
                ))

        # Similar for translation step
        if (translation_step is not None
            and translation_step_D is not None
            and translation_step_D >= 0
           ):
            raise ValueError("Only one of translation_step and translation_step_D can be defined.")
        if translation_step is None and translation_step_D is None:
            raise ValueError("Either translation_step or translation_step_D must be defined.")
        if translation_step_D is not None and translation_step is None:
            if translation_step_D < 0: # Default to 1D
                translation_step_D = 1.0
            translation_step = translation_step_D * fmodel.core.farm.rotor_diameters[0]
            if len(np.unique(fmodel.core.farm.rotor_diameters)) > 1:
                self.logger.warning((
                    "Found multiple turbine diameters. Using diameter of first turbine to set"
                    f" translation step to {translation_step}m ({translation_step_D} diameters)."
                ))

        # Initialize the base class
        super().__init__(
            fmodel,
            boundaries,
            min_dist=spacing,
            enable_geometric_yaw=enable_geometric_yaw,
            use_value=use_value,
        )

        # Create the default grid

        # use spacing, hexagonal packing, and boundaries to create a grid.
        d = 1.1 * np.sqrt((self.xmax**2 - self.xmin**2) + (self.ymax**2 - self.ymin**2))
        grid_1D = np.arange(0, d+spacing, spacing)
        if hexagonal_packing:
            x_locs = np.tile(grid_1D.reshape(1,-1), (len(grid_1D), 1))
            x_locs[np.arange(1, len(grid_1D), 2), :] += 0.5 * spacing
            y_locs = np.tile(np.sqrt(3) / 2 * grid_1D.reshape(-1,1), (1, len(grid_1D)))
        else:
            x_locs, y_locs = np.meshgrid(grid_1D, grid_1D)
        x_locs = x_locs.flatten() - np.mean(x_locs) + 0.5*(self.xmax + self.xmin)
        y_locs = y_locs.flatten() - np.mean(y_locs) + 0.5*(self.ymax + self.ymin)

        # Trim to a circle to minimize wasted computation
        x_locs_grid, y_locs_grid = self.trim_to_circle(
            x_locs,
            y_locs,
            (grid_1D.max()-grid_1D.min()+spacing)/2
        )
        self.xy_grid = np.concatenate(
            [x_locs_grid.reshape(-1,1), y_locs_grid.reshape(-1,1)],
            axis=1
        )

        # Limit the rotation range if grid has symmetry
        if hexagonal_packing:
            # Hexagonal packing has 60 degree symmetry
            rotation_range = (
                rotation_range[0],
                np.minimum(rotation_range[1], rotation_range[0]+60)
            )
        else:
            # Square grid has 90 degree symmetry
            rotation_range = (
                rotation_range[0],
                np.minimum(rotation_range[1], rotation_range[0]+90)
            )

        # Deal with None translation_range
        if translation_range is None:
            translation_range = (0.0, spacing)

        # Create test rotations and translations
        self.rotations = np.arange(rotation_range[0], rotation_range[1], rotation_step)
        self.translations = np.arange(translation_range[0], translation_range[1], translation_step)
        self.translations = np.concatenate([-np.flip(self.translations), self.translations])

    def optimize(self):

        # Sweep over rotations and translations to find the best layout
        n_rots = len(self.rotations)
        n_trans = len(self.translations)
        n_tot = n_rots * n_trans**2

        # There are a total of n_rots x n_trans x n_trans layouts to test
        rots_rad = np.radians(self.rotations)
        rotation_matrices = np.array(
            [
                [np.cos(rots_rad), -np.sin(rots_rad)],
                [np.sin(rots_rad), np.cos(rots_rad)]
            ]
        ).transpose(2,0,1)

        translations_x, translations_y = np.meshgrid(self.translations, self.translations)
        translation_matrices = np.concatenate(
            [translations_x.reshape(-1,1), translations_y.reshape(-1,1)],
            axis=1
        )

        rotations_all = np.tile(rotation_matrices, (n_trans**2, 1, 1))
        translations_all = np.repeat(translation_matrices, n_rots, axis=0)[:,None,:]

        # Create candidate layouts [(n_rots x n_trans x n_trans) x n_turbines x 2]
        candidate_layouts = np.einsum('ijk,lk->ilj', rotations_all, self.xy_grid) + translations_all

        # For each candidate layout, check how many turbines are in bounds
        turbines_in_bounds = np.zeros(n_tot)
        for i in range(n_tot):
            turbines_in_bounds[i] = np.sum(
                [test_point_in_bounds(xy[0], xy[1], self._boundary_polygon) for
                 xy in candidate_layouts[i, :, :]]
            )
        idx_max = np.argmax(turbines_in_bounds) # FIRST maximizing index returned

        # Get the best layout
        x_opt_all = candidate_layouts[idx_max, :, 0]
        y_opt_all = candidate_layouts[idx_max, :, 1]
        mask_in_bounds = [test_point_in_bounds(x, y, self._boundary_polygon) for
                          x, y in zip(x_opt_all, y_opt_all)]

        # Return the best layout, along with the number of turbines in bounds
        return turbines_in_bounds[idx_max], x_opt_all[mask_in_bounds], y_opt_all[mask_in_bounds]

    @staticmethod
    def trim_to_circle(x_locs, y_locs, radius):
        center = np.array([0.5*(x_locs.max() + x_locs.min()), 0.5*(y_locs.max() + y_locs.min())])
        xy = np.concatenate([x_locs.reshape(-1,1), y_locs.reshape(-1,1)], axis=1)
        mask = np.linalg.norm(xy - center, axis=1) <= radius
        return x_locs[mask], y_locs[mask]
