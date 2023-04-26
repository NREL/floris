import re
from math import ceil
from pathlib import Path

import attrs
import matplotlib.pyplot as plt
import numpy as np
from attrs import define, field

from floris.simulation import (
    Ct,
    power,
    Turbine,
)
from floris.type_dec import NDArrayFloat
from floris.utilities import load_yaml


INTERNAL_LIBRARY = Path(__file__).parent
DEFAULT_WIND_SPEEDS = np.linspace(0, 40, 81)


def round_nearest_2_or_5(x: int | float) -> int:
    """Rounds a number (with a 0.5 buffer) up to the nearest integer divisible by 2 or 5.

    Args:
        x (int | float): The number to be rounded.

    Returns:
        int: The rounded number.
    """
    base_2 = 2
    base_5 = 5
    return min(base_2 * ceil((x + 0.5) / base_2), base_5 * ceil((x + 0.5) / base_5))


def round_nearest_5(x: int | float) -> int:
    """Rounds a number (with a 0.5 buffer) up to the nearest integer divisible by 5.

    Args:
        x (int | float): The number to be rounded.

    Returns:
        int: The rounded number.
    """
    base_5 = 5
    return base_5 * ceil((x + 0.5) / base_5)


def round_scientific(x: float) -> float:
    """Rounds a number in scientific notation up to the nearest 2 or 5 on the same decimal
    scale.

    Args:
        x (float): The number in scientific notation to round.

    Returns:
        float: The rounded scientific notation number.
    """
    x_str = f"{x:.20f}"
    n_zeros = len(re.search("\d+\.(0*)", x_str).group(1))  # noqa: disable=W605
    x_crop = x_str.replace("1".zfill(n_zeros + 1)[:-1], "")
    x_round = round_nearest_2_or_5(float(x_crop) * 10)
    x_round = float(f"0.{str(x_round).zfill(n_zeros + 1)}")
    return x_round


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
            ref_density_cp_ct=np.full(shape, self.turbine.ref_density_cp_ct),
            rotor_effective_velocities=wind_speed.reshape(shape),
            power_interp={self.turbine.turbine_type: self.turbine.power_interp},
            turbine_type_map=np.full(shape, self.turbine.turbine_type)
        ).flatten() / 1e6
        return wind_speed, power_mw

    def thrust_curve(
        self,
        wind_speed: NDArrayFloat | None = None
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Produces a plot-ready thrust curve for the turbine for wind speed vs thrust coefficient
        assuming no tilt or yaw effects.

        Parameters
        ----------
        wind_speed : NDArrayFloat | None, optional
            The wind speed conditions to produce the power curve for. If None, then the default wind
            conditions are used (0 m/s -> 40 m/s, every 0.5 m/s), by default None.

        Returns
        -------
        tuple[NDArrayFloat, NDArrayFloat]
            Returns the wind speed array and the thrust array.
        """
        if wind_speed is None:
            wind_speed = DEFAULT_WIND_SPEEDS
        shape = (1, wind_speed.size, 1)
        ct_curve = Ct(
            velocities=wind_speed.reshape(shape),
            yaw_angle=np.zeros(shape),
            tilt_angle=np.full(shape, self.turbine.ref_tilt_cp_ct),
            ref_tilt_cp_ct=np.full(shape, self.turbine.ref_tilt_cp_ct),
            fCt={self.turbine.turbine_type: self.turbine.fCt_interp},
            tilt_interp=[(self.turbine.turbine_type, self.turbine.fTilt_interp)],
            correct_cp_ct_for_tilt=np.zeros(shape, dtype=bool),
            turbine_type_map=np.full(shape, self.turbine.turbine_type),
        ).flatten() / 1e6
        return wind_speed, ct_curve


@define(auto_attribs=True)
class TurbineLibrary:
    turbine_map: dict[str: TurbineInterface] = field(factory=dict)
    power_curves: dict[str, tuple[NDArrayFloat, NDArrayFloat]] = field(factory=dict)
    thrust_curves: dict[str, tuple[NDArrayFloat, NDArrayFloat]] = field(factory=dict)

    def load_internal_library(self, which: list[str] = [], exclude: list[str] = []) -> None:
        """Loads all of the turbine configurations from ``floris/floris/turbine_libary``,
        except any turbines defined in :py:attr:`exclude`.

        Args:
            which (list[str], optional): A list of which file names to include from loading.
                Defaults to [].
            exclude (list[str], optional): A list of file names to exclude from loading. Defaults to
                [].
        """
        include = [el for el in INTERNAL_LIBRARY.iterdir() if el.suffix in (".yaml", ".yml")]
        which = [INTERNAL_LIBRARY / el for el in which] if which != [] else include
        exclude = [INTERNAL_LIBRARY / el for el in exclude]
        include = set(which).intersection(include).difference(exclude)
        for fn in include:
            turbine_dict = load_yaml(fn)
            turbine_dict.setdefault("ref_density_cp_ct", 1.225)
            self.turbine_map.update({
                turbine_dict["turbine_type"]: TurbineInterface.from_turbine_dict(turbine_dict)
            })

    def load_external_library(
            self,
            library_path: str | Path,
            which: list[str] = [],
            exclude: list[str] = [],
        ) -> None:
        """Loads all the turbine configurations from :py:attr:`library_path`, except the file names
        defined in :py:attr:`exclude`, and adds each to ``turbine_map`` via a dictionary
        update.

        Args:
            library_path : str | Path
                The external turbine library that should be used for loading the turbines.
            which (list[str], optional): A list of which file names to include from loading.
                Defaults to [].
            exclude (list[str], optional): A list of file names to exclude from loading. Defaults to
                [].
        """
        library_path = Path(library_path).resolve()
        include = [el for el in library_path.iterdir() if el.suffix in (".yaml", ".yml")]
        which = [library_path / el for el in which] if which != [] else include
        exclude = [library_path / el for el in exclude]
        include = set(which).intersection(include).difference(exclude)
        for fn in include:
            turbine_dict = load_yaml(fn)
            turbine_dict.setdefault("ref_density_cp_ct", 1.225)
            self.turbine_map.update({
                turbine_dict["turbine_type"]: TurbineInterface.from_turbine_dict(turbine_dict)
            })

    def compute_power_curves(
            self,
            which: list[str] = [],
            exclude: list[str] = [],
            wind_speed: NDArrayFloat | None = None,
        ) -> None:
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

    def compute_thrust_curves(
            self,
            which: list[str] = [],
            exclude: list[str] = [],
            wind_speed: NDArrayFloat | None = None,
        ) -> None:
        """Computes the thrust curves for each turbine in ``turbine_map`` and sets the
        ``thrust_curves`` attribute.

        Args:
            wind_speed (`NDArrayFloat`): A 1-D array of wind speeds, in m/s, to compute the
                thrust curve for, for each turbine in ``turbine_map``.
        """
        if wind_speed is None:
            wind_speed = DEFAULT_WIND_SPEEDS
        self.thrust_curves = {
            name: t.thrust_curve(wind_speed) for name, t in self.turbine_map.items()
        }

    def plot_power_curves(
        self,
        which: list[str] = [],
        exclude: list[str] = [],
        wind_speed: NDArrayFloat | None = None,
        fig_kwargs: dict = {},
        plot_kwargs = {},
        return_fig: bool = False
    ) -> None | tuple[plt.Figure, plt.Axes]:
        """Plots each power curve in ``turbine_map`` in a single plot.

        Args:
            which (list[str], optional): A list of which turbine types/names to include. Defaults to
                [].
            exclude (list[str], optional): A list of turbine types/names names to exclude. Defaults
                to [].
            wind_speed (NDArrayFloat | None, optional): A 1-D array of wind speeds, in m/s. Defaults
                to None.
            fig_kwargs (dict, optional): Any keywords arguments to be passed to ``plt.Figure()``.
                Defaults to {}.
            plot_kwargs (dict, optional): Any keyword arguments to be passed to ``plt.plot()``.
                Defaults to {}.
            return_fig (bool, optional): Indicator if the ``Figure`` and ``Axes`` objects should be
                returned. Defaults to False.

        Returns:
            None | tuple[plt.Figure, plt.Axes]: None, if :py:attr:`return_fig` is False, otherwise
                a tuple of the Figure and Axes objects are returned.
        """
        if self.power_curves == {} or wind_speed is not None:
            self.compute_power_curves(which=which, exclude=exclude, wind_speed=wind_speed)

        which = [*self.turbine_map] if which == [] else which

        # Set the figure defaults if none are provided
        fig_kwargs.setdefault("dpi", 200)
        fig_kwargs.setdefault("figsize", (10, 8))

        fig = plt.figure()
        ax = fig.add_subplot(111)

        min_windspeed = 0
        max_windspeed = 0
        min_power = 0
        max_power = 0
        for name, (ws, p) in self.power_curves.items():
            if name in exclude or name not in which:
                continue
            max_power = max(p.max(), max_power)
            max_windspeed = max(ws.max(), max_windspeed)
            ax.plot(ws, p, label=name, **plot_kwargs)

        ax.grid()
        ax.set_axisbelow(True)
        ax.legend()

        max_power = round_nearest_2_or_5(max_power)
        ax.set_xlim(min_windspeed, max_windspeed)
        ax.set_ylim(min_power, max_power)

        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("Power (MW)")

        if return_fig:
            return fig, ax

        fig.tight_layout()

    def plot_thrust_curves(
        self,
        which: list[str] = [],
        exclude: list[str] = [],
        wind_speed: NDArrayFloat | None = None,
        fig_kwargs: dict = {},
        plot_kwargs = {},
        return_fig: bool = False
    ) -> None | tuple[plt.Figure, plt.Axes]:
        """Plots each thrust curve in ``turbine_map`` in a single plot.

        Args:
            which (list[str], optional): A list of which turbine types/names to include. Defaults to
                [].
            exclude (list[str], optional): A list of turbine types/names names to exclude. Defaults
                to [].
            wind_speed (NDArrayFloat | None, optional): A 1-D array of wind speeds, in m/s. Defaults
                to None.
            fig_kwargs (dict, optional): Any keywords arguments to be passed to ``plt.Figure()``.
                Defaults to {}.
            plot_kwargs (dict, optional): Any keyword arguments to be passed to ``plt.plot()``.
                Defaults to {}.
            return_fig (bool, optional): Indicator if the ``Figure`` and ``Axes`` objects should be
                returned. Defaults to False.

        Returns:
            None | tuple[plt.Figure, plt.Axes]: None, if :py:attr:`return_fig` is False, otherwise
                a tuple of the Figure and Axes objects are returned.
        """
        if self.thrust_curves == {} or wind_speed is None:
            self.compute_thrust_curves(which=which, exclude=exclude, wind_speed=wind_speed)

        which = [*self.turbine_map] if which == [] else which

        # Set the figure defaults if none are provided
        fig_kwargs.setdefault("dpi", 200)
        fig_kwargs.setdefault("figsize", (10, 8))

        fig = plt.figure()
        ax = fig.add_subplot(111)

        min_windspeed = 0
        max_windspeed = 0
        min_power = 0
        max_thrust = 0
        for name, (ws, p) in self.thrust_curves.items():
            if name in exclude or name not in which:
                continue
            max_thrust = max(p.max(), max_thrust)
            max_windspeed = max(ws.max(), max_windspeed)
            ax.plot(ws, p, label=name, **plot_kwargs)

        ax.grid()
        ax.set_axisbelow(True)
        ax.legend()

        max_thrust = round_scientific(max_thrust)
        ax.set_xlim(min_windspeed, max_windspeed)
        ax.set_ylim(min_power, max_thrust)

        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("Thrust Coefficient")

        if return_fig:
            return fig, ax

        fig.tight_layout()

    def plot_rotor_diameters(
        self,
        which: list[str] = [],
        exclude: list[str] = [],
        fig_kwargs: dict = {},
        bar_kwargs = {},
        return_fig: bool = False,
    ) -> None | tuple[plt.Figure, plt.Axes]:
        which = [*self.turbine_map] if which == [] else which

        # Set the figure defaults if none are provided
        fig_kwargs.setdefault("dpi", 200)
        fig_kwargs.setdefault("figsize", (10, 8))

        fig = plt.figure()
        ax = fig.add_subplot(111)

        subset_map = {
            name: t for name, t in self.turbine_map.items()
            if name not in exclude or name in which
        }
        x = np.arange(len(subset_map))
        y = [ti.turbine.rotor_diameter for ti in subset_map.values()]
        ix_sort = np.argsort(y)
        y_sorted = np.array(y)[ix_sort]
        ax.bar(x, y_sorted, **bar_kwargs)

        ax.grid(axis="y")
        ax.set_axisbelow(True)

        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_ylim(0, round_nearest_5(max(y) / 10) * 10)

        ax.set_xticks(x)
        ax.set_xticklabels(np.array([*subset_map])[ix_sort])
        ax.set_ylabel("Rotor Diameter (m)")

        if return_fig:
            return fig, ax

        fig.tight_layout()
