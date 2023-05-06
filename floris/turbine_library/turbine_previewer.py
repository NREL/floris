# Copyright 2023 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

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
from floris.utilities import (
    load_yaml,
    round_nearest,
    round_nearest_2_or_5,
)


INTERNAL_LIBRARY = Path(__file__).parent
DEFAULT_WIND_SPEEDS = np.linspace(0, 40, 81)


@define(auto_attribs=True)
class TurbineInterface:
    turbine: Turbine = field(validator=attrs.validators.instance_of(Turbine))

    @classmethod
    def from_internal_library(cls, file_name: str):
        """Loads the turbine definition from a YAML configuration file located in
        ``floris/floris/turbine_library/``.

        Args:
            file_`name : str | Path
                T`he file name of the turbine configuration file.

        Returns:
            (TurbineInterface): Creates a new ``TurbineInterface`` object.
        """
        return cls(turbine=Turbine.from_dict(load_yaml(INTERNAL_LIBRARY / file_name)))

    @classmethod
    def from_yaml(cls, file_path: str | Path):
        """Loads the turbine definition from a YAML configuration file.

        Args:
            file_path : str | Path
                The full path and file name of the turbine configuration file.

        Returns:
            (TurbineInterface): Creates a new ``TurbineInterface`` object.
        """
        return cls(turbine=Turbine.from_dict(load_yaml(file_path)))

    @classmethod
    def from_turbine_dict(cls, config_dict: dict):
        """Loads the turbine definition from a dictionary.

        Args:
            config_dict : dict
                The ``Turbine`` configuration dictionary.

        Returns:
            (`TurbineInterface`): Returns a ``TurbineInterface`` object.
        """
        return cls(turbine=Turbine.from_dict(config_dict))

    def power_curve(
        self,
        wind_speeds: NDArrayFloat = DEFAULT_WIND_SPEEDS,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Produces a plot-ready power curve for the turbine for wind speed vs power (MW), assuming
        no tilt or yaw effects.

        Args:
            wind_speeds (NDArrayFloat, optional): A 1-D array of wind speeds, in m/s. Defaults to
                0 m/s -> 40 m/s, every 0.5 m/s.

        Returns:
            (tuple[NDArrayFloat, NDArrayFloat]): Returns the wind speed array and the power array.
        """
        shape = (1, wind_speeds.size, 1)
        power_mw = power(
            ref_density_cp_ct=np.full(shape, self.turbine.ref_density_cp_ct),
            rotor_effective_velocities=wind_speeds.reshape(shape),
            power_interp={self.turbine.turbine_type: self.turbine.power_interp},
            turbine_type_map=np.full(shape, self.turbine.turbine_type)
        ).flatten() / 1e6
        return wind_speeds, power_mw

    def Cp_curve(
        self,
        wind_speeds: NDArrayFloat = DEFAULT_WIND_SPEEDS,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Produces a plot-ready thrust curve for the turbine for wind speed vs power coefficient
        assuming no tilt or yaw effects.

        Args:
        wind_speeds : NDArrayFloat, optional
            The wind speed conditions to produce the power curve for, by default 0 m/s -> 40 m/s,
            every 0.5 m/s.

        Returns:
            tuple[NDArrayFloat, NDArrayFloat]
                Returns the wind speed array and the power coefficient array.
        """
        cp_curve = self.turbine.fCp_interp(wind_speeds)
        return wind_speeds, cp_curve

    def Ct_curve(
        self,
        wind_speeds: NDArrayFloat = DEFAULT_WIND_SPEEDS,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Produces a plot-ready thrust curve for the turbine for wind speed vs thrust coefficient
        assuming no tilt or yaw effects.

        Args:
            wind_speeds (NDArrayFloat, optional): A 1-D array of wind speeds, in m/s. Defaults to
                0 m/s -> 40 m/s, every 0.5 m/s.

        Returns:
            tuple[NDArrayFloat, NDArrayFloat]
                Returns the wind speed array and the thrust coefficient array.
        """
        shape = (1, wind_speeds.size, 1)
        ct_curve = Ct(
            velocities=wind_speeds.reshape(shape),
            yaw_angle=np.zeros(shape),
            tilt_angle=np.full(shape, self.turbine.ref_tilt_cp_ct),
            ref_tilt_cp_ct=np.full(shape, self.turbine.ref_tilt_cp_ct),
            fCt={self.turbine.turbine_type: self.turbine.fCt_interp},
            tilt_interp=[(self.turbine.turbine_type, self.turbine.fTilt_interp)],
            correct_cp_ct_for_tilt=np.zeros(shape, dtype=bool),
            turbine_type_map=np.full(shape, self.turbine.turbine_type),
        ).flatten()
        return wind_speeds, ct_curve

    def plot_power_curve(
        self,
        wind_speeds: NDArrayFloat = DEFAULT_WIND_SPEEDS,
        fig_kwargs: dict = {},
        plot_kwargs = {},
        return_fig: bool = False
    ) -> None | tuple[plt.Figure, plt.Axes]:
        """Plots the power curve for a given set of wind speeds.

        Args:
            wind_speeds (NDArrayFloat, optional): A 1-D array of wind speeds, in m/s.
                Defaults to 0 m/s -> 40 m/s, every 0.5 m/s.
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
        wind_speeds, power_mw = self.power_curve(wind_speeds=wind_speeds)

        # Set the figure defaults if none are provided
        fig_kwargs.setdefault("dpi", 200)
        fig_kwargs.setdefault("figsize", (4, 3))

        fig = plt.figure(**fig_kwargs)
        ax = fig.add_subplot(111)

        min_windspeed = 0
        max_windspeed = max(wind_speeds)
        min_power = 0
        max_power = max(power_mw)
        ax.plot(wind_speeds, power_mw, label=self.turbine.turbine_type, **plot_kwargs)

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

    def plot_Cp_curve(
        self,
        wind_speeds: NDArrayFloat = DEFAULT_WIND_SPEEDS,
        fig_kwargs: dict = {},
        plot_kwargs = {},
        return_fig: bool = False
    ) -> None | tuple[plt.Figure, plt.Axes]:
        """Plots the power coefficient curve for a given set of wind speeds.

        Args:
            wind_speeds (NDArrayFloat, optional): A 1-D array of wind speeds, in m/s. Defaults to
                0 m/s -> 40 m/s, every 0.5 m/s.
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
        wind_speeds, power_c = self.Cp_curve(wind_speeds=wind_speeds)

        # Set the figure defaults if none are provided
        fig_kwargs.setdefault("dpi", 200)
        fig_kwargs.setdefault("figsize", (4, 3))

        fig = plt.figure(**fig_kwargs)
        ax = fig.add_subplot(111)

        min_windspeed = 0
        max_windspeed = max(wind_speeds)
        ax.plot(wind_speeds, power_c, label=self.turbine.turbine_type, **plot_kwargs)

        ax.grid()
        ax.set_axisbelow(True)
        ax.legend()

        ax.set_xlim(min_windspeed, max_windspeed)
        ax.set_ylim(0, round_nearest(max(power_c) * 100, base=10) / 100)

        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("Power Coefficient")

        if return_fig:
            return fig, ax

        fig.tight_layout()

    def plot_Ct_curve(
        self,
        wind_speeds: NDArrayFloat = DEFAULT_WIND_SPEEDS,
        fig_kwargs: dict = {},
        plot_kwargs = {},
        return_fig: bool = False
    ) -> None | tuple[plt.Figure, plt.Axes]:
        """Plots the thrust coefficient curve for a given set of wind speeds.

        Args:
            wind_speeds (NDArrayFloat, optional): A 1-D array of wind speeds, in m/s. Defaults to
                0 m/s -> 40 m/s, every 0.5 m/s.
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
        wind_speeds, thrust = self.Ct_curve(wind_speeds=wind_speeds)

        # Set the figure defaults if none are provided
        fig_kwargs.setdefault("dpi", 200)
        fig_kwargs.setdefault("figsize", (4, 3))

        fig = plt.figure(**fig_kwargs)
        ax = fig.add_subplot(111)

        min_windspeed = 0
        max_windspeed = max(wind_speeds)
        ax.plot(wind_speeds, thrust, label=self.turbine.turbine_type, **plot_kwargs)

        ax.grid()
        ax.set_axisbelow(True)
        ax.legend()

        ax.set_xlim(min_windspeed, max_windspeed)
        ax.set_ylim(0, round_nearest(max(thrust) * 100, base=10) / 100)

        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("Thrust Coefficient")

        if return_fig:
            return fig, ax

        fig.tight_layout()


@define(auto_attribs=True)
class TurbineLibrary:
    turbine_map: dict[str: TurbineInterface] = field(factory=dict)
    power_curves: dict[str, tuple[NDArrayFloat, NDArrayFloat]] = field(factory=dict)
    Cp_curves: dict[str, tuple[NDArrayFloat, NDArrayFloat]] = field(factory=dict)
    Ct_curves: dict[str, tuple[NDArrayFloat, NDArrayFloat]] = field(factory=dict)

    def load_internal_library(self, which: list[str] = [], exclude: list[str] = []) -> None:
        """Loads all of the turbine configurations from ``floris/floris/turbine_libary``,
        except any turbines defined in :py:attr:`exclude`.

        Args:
            which (list[str], optional): A list of which file names to include from loading.
                Defaults to [].
            exclude (list[str], optional): A list of file names to exclude from loading.
                Defaults to [].
        """
        include = [el for el in INTERNAL_LIBRARY.iterdir() if el.suffix in (".yaml", ".yml")]
        which = [INTERNAL_LIBRARY / el for el in which] if which != [] else include
        exclude = [INTERNAL_LIBRARY / el for el in exclude]
        include = set(which).intersection(include).difference(exclude)
        for fn in include:
            turbine_dict = load_yaml(fn)
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
            exclude (list[str], optional): A list of file names to exclude from loading.
                Defaults to [].
        """
        library_path = Path(library_path).resolve()
        include = [el for el in library_path.iterdir() if el.suffix in (".yaml", ".yml")]
        which = [library_path / el for el in which] if which != [] else include
        exclude = [library_path / el for el in exclude]
        include = set(which).intersection(include).difference(exclude)
        for fn in include:
            turbine_dict = load_yaml(fn)
            self.turbine_map.update({
                turbine_dict["turbine_type"]: TurbineInterface.from_turbine_dict(turbine_dict)
            })

    def compute_power_curves(
            self,
            wind_speeds: NDArrayFloat = DEFAULT_WIND_SPEEDS,
        ) -> None:
        """Computes the power curves for each turbine in ``turbine_map`` and sets the
        ``power_curves`` attribute.

        Args:
            wind_speeds (NDArrayFloat, optional): A 1-D array of wind speeds, in m/s. Defaults to
                0 m/s -> 40 m/s, every 0.5 m/s.
        """
        self.power_curves = {
            name: t.power_curve(wind_speeds) for name, t in self.turbine_map.items()
        }

    def compute_Cp_curves(
            self,
            wind_speeds: NDArrayFloat = DEFAULT_WIND_SPEEDS,
        ) -> None:
        """Computes the power coefficient curves for each turbine in ``turbine_map`` and sets the
        ``Ct_curves`` attribute.

        Args:
            wind_speeds (NDArrayFloat, optional): A 1-D array of wind speeds, in m/s. Defaults to
                0 m/s -> 40 m/s, every 0.5 m/s.
        """
        self.Cp_curves = {
            name: t.Cp_curve(wind_speeds) for name, t in self.turbine_map.items()
        }

    def compute_Ct_curves(
            self,
            wind_speeds: NDArrayFloat = DEFAULT_WIND_SPEEDS,
        ) -> None:
        """Computes the thrust curves for each turbine in ``turbine_map`` and sets the
        ``Ct_curves`` attribute.

        Args:
            wind_speeds (NDArrayFloat, optional): A 1-D array of wind speeds, in m/s. Defaults to
                0 m/s -> 40 m/s, every 0.5 m/s.
        """
        self.Ct_curves = {
            name: t.Ct_curve(wind_speeds) for name, t in self.turbine_map.items()
        }

    def plot_power_curves(
        self,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        which: list[str] = [],
        exclude: list[str] = [],
        wind_speeds: NDArrayFloat = DEFAULT_WIND_SPEEDS,
        fig_kwargs: dict = {},
        plot_kwargs = {},
        return_fig: bool = False,
        show: bool = False,
    ) -> None | tuple[plt.Figure, plt.Axes]:
        """Plots each power curve in ``turbine_map`` in a single plot.

        Args:
            fig (plt.figure, optional): A pre-made figure where the plot should exist.
            ax (plt.Axes, optional): A pre-initialized axes object that should be used for the plot.
            which (list[str], optional): A list of which turbine types/names to include. Defaults to
                [].
            exclude (list[str], optional): A list of turbine types/names names to exclude. Defaults
                to [].
            wind_speeds (NDArrayFloat, optional): A 1-D array of wind speeds, in m/s. Defaults to
                0 m/s -> 40 m/s, every 0.5 m/s.
            fig_kwargs (dict, optional): Any keywords arguments to be passed to ``plt.Figure()``.
                Defaults to {}.
            plot_kwargs (dict, optional): Any keyword arguments to be passed to ``plt.plot()``.
                Defaults to {}.
            return_fig (bool, optional): Indicator if the ``Figure`` and ``Axes`` objects should be
                returned. Defaults to False.
            show (bool, optional): Indicator if the figure should be automatically displayed.
                Defaults to False.

        Returns:
            None | tuple[plt.Figure, plt.Axes]: None, if :py:attr:`return_fig` is False, otherwise
                a tuple of the Figure and Axes objects are returned.
        """
        if self.power_curves == {} or wind_speeds is not None:
            self.compute_power_curves(wind_speeds=wind_speeds)

        which = [*self.turbine_map] if which == [] else which

        # Set the figure defaults if none are provided
        if fig is None:
            fig_kwargs.setdefault("dpi", 200)
            fig_kwargs.setdefault("figsize", (4, 3))

            fig = plt.figure(**fig_kwargs)
        if ax is None:
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

        max_power = round_nearest(max_power, base=5)
        ax.set_xlim(min_windspeed, max_windspeed)
        ax.set_ylim(min_power, max_power)

        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("Power (MW)")

        if return_fig:
            return fig, ax

        if show:
            fig.tight_layout()

    def plot_Cp_curves(
        self,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        which: list[str] = [],
        exclude: list[str] = [],
        wind_speeds: NDArrayFloat = DEFAULT_WIND_SPEEDS,
        fig_kwargs: dict = {},
        plot_kwargs = {},
        return_fig: bool = False,
        show: bool = False,
    ) -> None | tuple[plt.Figure, plt.Axes]:
        """Plots each power coefficient curve in ``turbine_map`` in a single plot.

        Args:
            fig (plt.figure, optional): A pre-made figure where the plot should exist.
            ax (plt.Axes, optional): A pre-initialized axes object that should be used for the plot.
            which (list[str], optional): A list of which turbine types/names to include. Defaults to
                [].
            exclude (list[str], optional): A list of turbine types/names names to exclude. Defaults
                to [].
            wind_speeds (NDArrayFloat, optional): A 1-D array of wind speeds, in m/s. Defaults to
                0 m/s -> 40 m/s, every 0.5 m/s.
            fig_kwargs (dict, optional): Any keywords arguments to be passed to ``plt.Figure()``.
                Defaults to {}.
            plot_kwargs (dict, optional): Any keyword arguments to be passed to ``plt.plot()``.
                Defaults to {}.
            return_fig (bool, optional): Indicator if the ``Figure`` and ``Axes`` objects should be
                returned. Defaults to False.
            show (bool, optional): Indicator if the figure should be automatically displayed.
                Defaults to False.

        Returns:
            None | tuple[plt.Figure, plt.Axes]: None, if :py:attr:`return_fig` is False, otherwise
                a tuple of the Figure and Axes objects are returned.
        """
        if self.Cp_curves == {} or wind_speeds is None:
            self.compute_Cp_curves(wind_speeds=wind_speeds)

        which = [*self.turbine_map] if which == [] else which

        # Set the figure defaults if none are provided
        if fig is None:
            fig_kwargs.setdefault("dpi", 200)
            fig_kwargs.setdefault("figsize", (4, 3))

            fig = plt.figure(**fig_kwargs)
        if ax is None:
            ax = fig.add_subplot(111)

        min_windspeed = 0
        max_windspeed = 0
        max_power = 0
        for name, (ws, p) in self.Cp_curves.items():
            if name in exclude or name not in which:
                continue
            max_windspeed = max(ws.max(), max_windspeed)
            max_power = max(p.max(), max_power)
            ax.plot(ws, p, label=name, **plot_kwargs)

        ax.grid()
        ax.set_axisbelow(True)
        ax.legend()

        ax.set_xlim(min_windspeed, max_windspeed)
        ax.set_ylim(0, round_nearest(max_power * 100, base=10) / 100)

        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("Power Coefficient")

        if return_fig:
            return fig, ax

        if show:
            fig.tight_layout()

    def plot_Ct_curves(
        self,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        which: list[str] = [],
        exclude: list[str] = [],
        wind_speeds: NDArrayFloat = DEFAULT_WIND_SPEEDS,
        fig_kwargs: dict = {},
        plot_kwargs = {},
        return_fig: bool = False,
        show: bool = False,
    ) -> None | tuple[plt.Figure, plt.Axes]:
        """Plots each thrust coefficient curve in ``turbine_map`` in a single plot.

        Args:
            fig (plt.figure, optional): A pre-made figure where the plot should exist.
            ax (plt.Axes, optional): A pre-initialized axes object that should be used for the plot.
            which (list[str], optional): A list of which turbine types/names to include. Defaults to
                [].
            exclude (list[str], optional): A list of turbine types/names names to exclude. Defaults
                to [].
            wind_speeds (NDArrayFloat, optional): A 1-D array of wind speeds, in m/s. Defaults to
                0 m/s -> 40 m/s, every 0.5 m/s.
            fig_kwargs (dict, optional): Any keywords arguments to be passed to ``plt.Figure()``.
                Defaults to {}.
            plot_kwargs (dict, optional): Any keyword arguments to be passed to ``plt.plot()``.
                Defaults to {}.
            return_fig (bool, optional): Indicator if the ``Figure`` and ``Axes`` objects should be
                returned. Defaults to False.
            show (bool, optional): Indicator if the figure should be automatically displayed.
                Defaults to False.

        Returns:
            None | tuple[plt.Figure, plt.Axes]: None, if :py:attr:`return_fig` is False, otherwise
                a tuple of the Figure and Axes objects are returned.
        """
        if self.Ct_curves == {} or wind_speeds is None:
            self.compute_Ct_curves(wind_speeds=wind_speeds)

        which = [*self.turbine_map] if which == [] else which

        # Set the figure defaults if none are provided
        if fig is None:
            fig_kwargs.setdefault("dpi", 200)
            fig_kwargs.setdefault("figsize", (4, 3))

            fig = plt.figure(**fig_kwargs)
        if ax is None:
            ax = fig.add_subplot(111)

        min_windspeed = 0
        max_windspeed = 0
        max_thrust = 0
        for name, (ws, t) in self.Ct_curves.items():
            if name in exclude or name not in which:
                continue
            max_windspeed = max(ws.max(), max_windspeed)
            max_thrust = max(t.max(), max_thrust)
            ax.plot(ws, t, label=name, **plot_kwargs)

        ax.grid()
        ax.set_axisbelow(True)
        ax.legend()

        ax.set_xlim(min_windspeed, max_windspeed)
        ax.set_ylim(0, round_nearest(max_thrust * 100, base=10) / 100)

        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("Thrust Coefficient")

        if return_fig:
            return fig, ax

        if show:
            fig.tight_layout()

    def plot_rotor_diameters(
        self,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        which: list[str] = [],
        exclude: list[str] = [],
        fig_kwargs: dict = {},
        bar_kwargs = {},
        return_fig: bool = False,
        show: bool = False,
    ) -> None | tuple[plt.Figure, plt.Axes]:
        """Plots a bar chart of rotor diameters for each turbine in ``turbine_map``.

        Args:
            fig (plt.figure, optional): A pre-made figure where the plot should exist.
            ax (plt.Axes, optional): A pre-initialized axes object that should be used for the plot.
            which (list[str], optional): A list of which turbine types/names to include. Defaults to
                [].
            exclude (list[str], optional): A list of turbine types/names names to exclude. Defaults
                to [].
            fig_kwargs (dict, optional): Any keywords arguments to be passed to ``plt.Figure()``.
                Defaults to {}.
            bar_kwargs (dict, optional): Any keyword arguments to be passed to ``plt.plot()``.
                Defaults to {}.
            return_fig (bool, optional): Indicator if the ``Figure`` and ``Axes`` objects should be
                returned. Defaults to False.
            show (bool, optional): Indicator if the figure should be automatically displayed.
                Defaults to False.

        Returns:
            None | tuple[plt.Figure, plt.Axes]: None, if :py:attr:`return_fig` is False, otherwise
                a tuple of the Figure and Axes objects are returned.
        """
        which = [*self.turbine_map] if which == [] else which

        # Set the figure defaults if none are provided
        if fig is None:
            fig_kwargs.setdefault("dpi", 200)
            fig_kwargs.setdefault("figsize", (4, 3))

            fig = plt.figure(**fig_kwargs)
        if ax is None:
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
        ax.set_ylim(0, round_nearest(max(y) / 10, base=5) * 10)

        ax.set_xticks(x)
        ax.set_xticklabels(np.array([*subset_map])[ix_sort])
        ax.set_ylabel("Rotor Diameter (m)")

        if return_fig:
            return fig, ax

        if show:
            fig.tight_layout()

    def plot_hub_heights(
        self,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        which: list[str] = [],
        exclude: list[str] = [],
        fig_kwargs: dict = {},
        bar_kwargs = {},
        return_fig: bool = False,
        show: bool = False,
    ) -> None | tuple[plt.Figure, plt.Axes]:
        """Plots a bar chart of hub heights for each turbine in ``turbine_map``.

        Args:
            fig (plt.figure, optional): A pre-made figure where the plot should exist.
            ax (plt.Axes, optional): A pre-initialized axes object that should be used for the plot.
            which (list[str], optional): A list of which turbine types/names to include. Defaults to
                [].
            exclude (list[str], optional): A list of turbine types/names names to exclude. Defaults
                to [].
            fig_kwargs (dict, optional): Any keywords arguments to be passed to ``plt.Figure()``.
                Defaults to {}.
            bar_kwargs (dict, optional): Any keyword arguments to be passed to ``plt.plot()``.
                Defaults to {}.
            return_fig (bool, optional): Indicator if the ``Figure`` and ``Axes`` objects should be
                returned. Defaults to False.
            show (bool, optional): Indicator if the figure should be automatically displayed.
                Defaults to False.

        Returns:
            None | tuple[plt.Figure, plt.Axes]: None, if :py:attr:`return_fig` is False, otherwise
                a tuple of the Figure and Axes objects are returned.
        """
        which = [*self.turbine_map] if which == [] else which

        # Set the figure defaults if none are provided
        if fig is None:
            fig_kwargs.setdefault("dpi", 200)
            fig_kwargs.setdefault("figsize", (4, 3))

            fig = plt.figure(**fig_kwargs)
        if ax is None:
            ax = fig.add_subplot(111)

        subset_map = {
            name: t for name, t in self.turbine_map.items()
            if name not in exclude or name in which
        }
        x = np.arange(len(subset_map))
        y = [ti.turbine.hub_height for ti in subset_map.values()]
        ix_sort = np.argsort(y)
        y_sorted = np.array(y)[ix_sort]
        ax.bar(x, y_sorted, **bar_kwargs)

        ax.grid(axis="y")
        ax.set_axisbelow(True)

        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_ylim(0, round_nearest(max(y) / 10, base=5) * 10)

        ax.set_xticks(x)
        ax.set_xticklabels(np.array([*subset_map])[ix_sort])
        ax.set_ylabel("Hub Height (m)")

        if return_fig:
            return fig, ax

        if show:
            fig.tight_layout()

    def plot_comparison(
        self,
        which: list[str] = [],
        exclude: list[str] = [],
        wind_speeds: NDArrayFloat = DEFAULT_WIND_SPEEDS,
        fig_kwargs: dict = {},
        plot_kwargs = {},
        bar_kwargs = {},
        return_fig: bool = False
    ) -> None | tuple[plt.Figure, list[plt.Axes]]:
        """Plots each thrust curve in ``turbine_map`` in a single plot.

        Args:
            which (list[str], optional): A list of which turbine types/names to include. Defaults to
                [].
            exclude (list[str], optional): A list of turbine types/names names to exclude. Defaults
                to [].
            wind_speeds (NDArrayFloat, optional): A 1-D array of wind speeds, in m/s. Defaults to
                0 m/s -> 40 m/s, every 0.5 m/s.
            fig_kwargs (dict, optional): Any keywords arguments to be passed to ``plt.Figure()``.
                Defaults to {}.
            plot_kwargs (dict, optional): Any keyword arguments to be passed to ``plt.plot()``.
                Defaults to {}.
            bar_kwargs (dict, optional): Any keyword arguments to be passed to ``plt.bar()``.
                Defaults to {}.
            return_fig (bool, optional): Indicator if the ``Figure`` and ``Axes`` objects should be
                returned. Defaults to False.

        Returns:
            None | tuple[plt.Figure, list[plt.Axes]]: None, if :py:attr:`return_fig` is False,
                otherwise a tuple of the Figure and Axes objects are returned.
        """
        # Set the figure defaults if none are provided
        fig_kwargs.setdefault("dpi", 200)
        fig_kwargs.setdefault("figsize", (6, 5))

        fig = plt.figure(**fig_kwargs)
        ax1 = fig.add_subplot(321)
        ax2 = fig.add_subplot(322)
        ax3 = fig.add_subplot(323)
        ax4 = fig.add_subplot(324)
        ax5 = fig.add_subplot(325)
        ax_list = [ax1, ax2, ax3, ax4, ax5]

        self.plot_power_curves(
            fig,
            ax1,
            which=which,
            exclude=exclude,
            wind_speeds=wind_speeds,
            plot_kwargs=plot_kwargs,
        )
        self.plot_Cp_curves(
            fig,
            ax3,
            which=which,
            exclude=exclude,
            wind_speeds=wind_speeds,
            plot_kwargs=plot_kwargs,
        )
        self.plot_Ct_curves(
            fig,
            ax5,
            which=which,
            exclude=exclude,
            wind_speeds=wind_speeds,
            plot_kwargs=plot_kwargs,
        )
        self.plot_rotor_diameters(fig, ax2, which=which, exclude=exclude, bar_kwargs=bar_kwargs)
        self.plot_hub_heights(fig, ax4, which=which, bar_kwargs=bar_kwargs)

        for ax in ax_list:
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.xaxis.label.set_size(7)
            ax.yaxis.label.set_size(8)

        for ax in (ax1, ax3, ax5):
            ax.legend(fontsize=6)

        if return_fig:
            return fig, ax_list

        fig.tight_layout()
