from pathlib import Path

import attrs
import matplotlib.pyplot as plt
import numpy as np
from attrs import define, field

from floris.simulation import power, Turbine
from floris.type_dec import NDArrayFloat
from floris.utilities import load_yaml


INTERNAL_LIBRARY = Path(__file__).parent
DEFAULT_WIND_SPEEDS = np.linspace(0, 40, 81)


@define(auto_attribs=True)
class TurbineInterface:
    turbine: Turbine = field(validator=attrs.validators.instance_of(Turbine))

    @classmethod
    def from_yaml(cls, file_path: str | Path):
        """Loads the turbine defintion from a YAML configuration file.

        Parameters
        ----------
        file_path : str | Path
            The full path and file name of the turbine configuration file.

        Returns
        -------
        TurbineInterface
            Creates a new ``TurbineInterface`` object.
        """
        return cls(turbine=Turbine.from_dict(load_yaml(file_path)))

    @classmethod
    def from_turbine_dict(cls, config_dict: dict):
        """Loads the turbine defintion from a dictionary.

        Args:
            config_dict : dict
                The ``Turbine`` configuration dictionary.

        Returns:
            (`TurbineInterface`): Returns a ``TurbineInterface`` object.
        """
        return cls(turbine=Turbine.from_dict(config_dict))

    def power_curve(
            self,
            wind_speed: NDArrayFloat | None = None
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Produces a plot-ready power curve for the turbine for wind speed vs power (MW), assuming
        no tilt or yaw effects.

        Parameters
        ----------
        wind_speed : NDArrayFloat | None, optional
            The wind speed conditions to produce the power curve for. If None, then the default wind
            conditions are used (0 m/s -> 40 m/s, every 0.5 m/s), by default None.

        Returns
        -------
        tuple[NDArrayFloat, NDArrayFloat]
            Returns the wind speed array and the power array.
        """
        if wind_speed is None:
            wind_speed = DEFAULT_WIND_SPEEDS
        shape = (1, wind_speed.size, 1)
        power_mw = power(
            np.full(shape, self.turbine.ref_density_cp_ct),
            wind_speed.reshape(shape),
            {self.turbine.turbine_type: self.turbine.power_interp},
            np.full(shape, self.turbine.turbine_type)
        ).flatten() / 1e6
        return wind_speed, power_mw


@define(auto_attribs=True)
class TurbineLibrary:
    turbine_map: dict[str: TurbineInterface] = field(factory=dict)
    power_curves: dict[str, tuple[NDArrayFloat, NDArrayFloat]] = field(factory=dict)

    def load_internal_library(self, exclude: list[str] = []) -> None:
        """Loads all of the turbine configurations from ``floris/floris/turbine_libary``,
        except any turbines defined in :py:attr:`exclude`.

        Parameters
        ----------
        exclude : list[str], optional
            A list of file names to exclude from loading, by default [].
        """
        include = [el for el in INTERNAL_LIBRARY.iterdir() if el.suffix in (".yaml", ".yml")]
        exclude = [INTERNAL_LIBRARY / el for el in exclude]
        include = set(include).difference(exclude)
        for fn in include:
            turbine_dict = load_yaml(fn)
            self.turbine_map.update({
                turbine_dict["turbine_type"]: TurbineInterface.from_turbine_dict(turbine_dict)
            })

    def load_external_library(self, library_path: str | Path, exclude: list[str] = []) -> None:
        """Loads all the turbine configurations from :py:attr:`library_path`, except the file names
        defined in :py:attr:`exclude`, and adds each to ``turbine_map`` via a dictionary
        update.

        Args:
            library_path : str | Path
                The external turbine library that should be used for loading the turbines.
            exclude : list[str], optional
                A list of file names to exclude from loading, by default [].
        """
        library_path = Path(library_path).resolve()
        include = [el for el in INTERNAL_LIBRARY.iterdir() if el.suffix in (".yaml", ".yml")]
        exclude = [INTERNAL_LIBRARY / el for el in exclude]
        include = set(include).difference(exclude)
        for fn in include:
            turbine_dict = load_yaml(fn)
            self.turbine_map.update({turbine_dict["name"]: turbine_dict})

    def compute_power_curves(self, wind_speed: NDArrayFloat | None = None) -> None:
        """Computes the power curves for each turbine in ``turbine_map`` and sets the
        ``power_curves`` attribute.

        Args:
            wind_speed (`NDArrayFloat`): A 1-D array of wind speeds, in m/s, to compute the
                power curve for, for each turbine in ``turbine_map``.
        """
        if wind_speed is None:
            wind_speed = DEFAULT_WIND_SPEEDS
        self.power_curves = {
            name: t.power_curve(wind_speed) for name, t in self.turbine_map.items()
        }

    def plot_power_curves(
            self,
            wind_speed: NDArrayFloat | None = None,
            fig_kwargs: dict = {},
            plot_kwargs = {},
            return_fig: bool = False
        ) -> None | tuple[plt.Figure, plt.Axes]:
        # TODO: docstring
        if self.power_curves == {}:
            self.compute_power_curves(wind_speed)

        # Set the figure defaults if none are provided
        fig_kwargs.setdefault("dpi", 200)
        fig_kwargs.setdefault("figsize", (10, 8))

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for name, (ws, p) in self.power_curves.items():
            ax.plot(ws, p, label=name)

        ax.grid()
        ax.set_axisbelow(True)
        ax.legend()

        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("Power (MW)")

        if return_fig:
            return fig, ax

        fig.tight_layout()
