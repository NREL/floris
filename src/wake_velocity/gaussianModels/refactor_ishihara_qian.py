from typing import List

import attr
import numpy as np

from src.utilities import Vec3, FromDictMixin, tand, float_attrib, model_attrib
from src.base_class import BaseClass
from src.wake_velocity.gaussianModels.refactor_gauss_mixin import GaussMixin


@attr.s(auto_attribs=True)
class NestedParameter(FromDictMixin):  # NEEDS A BETTER NAME
    const: float = float_attrib()
    Ct: float = float_attrib()
    TI: float = float_attrib()


@attr.s(auto_attribs=True)
class IshiharaQian(BaseClass, GaussMixin):
    """Ishihara is a Gaussian wake velocity model that implements a near-wake correction.

    Ishihara is used to compute the wake velocity deficit based on the Gaussian
    wake model with self-similarity and a near wake correction. The Ishihara
    wake model includes a Gaussian wake velocity deficit profile in the y and z
    directions and includes the effects of ambient turbulence, added turbulence
    from upstream wakes, as well as wind shear and wind veer. For more info,
    see :cite:`iqv-qian2018new`.

    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: iqv-

    Args:
        kstar (:py:obj:`Dict[str, float]`): A dict that is a parameter used to
            determine the linear relationship between the
            turbulence intensity and the width of the Gaussian
            wake shape.
        epsilon (:py:obj:`Dict[str, float]`): A dict that is a second parameter
            used to determine the linear relationship between the
            turbulence intensity and the width of the Gaussian
            wake shape.
        a (:py:obj:`Dict[str, float]`): A dict that is a constant coefficient used
            in calculation of wake-added turbulence.
        b (:py:obj:`Dict[str, float]`): A dict that is linear coefficient used in
            calculation of wake-added turbulence.
        c (:py:obj:`Dict[str, float]`): A dict that is near-wake coefficient used
            in calculation of wake-added turbulence.
    """

    kstar: NestedParameter = attr.ib(
        default={"const": 0.11, "Ct": 1.07, "TI": 0.2},
        converter=NestedParameter.from_dict,
        kw_only=True,
    )
    epsilon: NestedParameter = attr.ib(
        default={"const": 0.23, "Ct": -0.25, "TI": 0.17},
        converter=NestedParameter.from_dict,
        kw_only=True,
    )
    a: NestedParameter = attr.ib(
        default={"const": 0.93, "Ct": -0.75, "TI": 0.17},
        converter=NestedParameter.from_dict,
        kw_only=True,
    )
    b: NestedParameter = attr.ib(
        default={"const": 0.42, "Ct": 0.6, "TI": 0.2},
        converter=NestedParameter.from_dict,
        kw_only=True,
    )
    c: NestedParameter = attr.ib(
        default={"const": 0.15, "Ct": -0.25, "TI": -0.7},
        converter=NestedParameter.from_dict,
        kw_only=True,
    )
    calculate_VW_velocities: bool = attr.ib(default=False, converter=bool, kw_only=True)
    use_yaw_aded_recovery: bool = attr.ib(default=False, converter=bool, kw_only=True)
    eps_gain: float = float_attrib(default=0.2)
    model_string: model_attrib(default="ishiharaquian")

    def _calculate_model_parameters(
        self, param: NestedParameter, Ct: float, TI: float
    ) -> float:
        """Calculates the model parameters using current conditions and provided settings.

        Parameters
        ----------
        param : NestedParameter
            The focal parameter. Should be one of `kstar`, `epsilon`, `a`, `b`, or `c`.
        Ct : float
            Thrust coefficient for the current turbine.
        TI : float
            Turbulence intensity.

        Returns
        -------
        float
            Current value of the model parameter.
        """
        return param.const * Ct ** param.Ct * TI ** param.TI

    def function(
        self,
        x_locations: np.ndarray,
        y_locations: np.ndarray,
        z_locations: np.ndarray,
        turbine: np.ndarray,  # we'll see here
        turbine_coord: Vec3,
        # deflection_field: np.ndarray,
        # flow_filed shall be deprecated for the following
        u_initial: np.ndarray,  # flow_field.u_initial
    ) -> None:
        """
        Main function for calculating the IshiharaQian Gaussian wake model.
        This method calculates and returns the wake velocity deficits caused by
        the specified turbine, relative to the freestream velocities at the
        grid of points comprising the wind farm flow field. This function is
        accessible through the :py:class:`~.Wake` class as the
        :py:meth:`~.Wake.velocity_function` method.

        Args:
            x_locations (np.array): An array of floats that contains the
                streamwise direction grid coordinates of the flow field
                domain (m).
            y_locations (np.array): An array of floats that contains the grid
                coordinates of the flow field domain in the direction normal to
                x and parallel to the ground (m).
            z_locations (np.array): An array of floats that contains the grid
                coordinates of the flow field domain in the vertical
                direction (m).
            turbine (:py:obj:`floris.simulation.turbine`): Object that
                represents the turbine creating the wake.
            turbine_coord (:py:obj:`floris.utilities.Vec3`): Object containing
                the coordinate of the turbine creating the wake (m).
            deflection_field (np.array): An array of floats that contains the
                amount of wake deflection in meters in the y direction at each
                grid point of the flow field.
            flow_field (:py:class:`floris.simulation.flow_field`): Object
                containing the flow field information for the wind farm.

        Returns:
            np.array, np.array, np.array:
                Three arrays of floats that contain the wake velocity
                deficit in m/s created by the turbine relative to the freestream
                velocities for the U, V, and W components, aligned with the x,
                y, and z directions, respectively. The three arrays contain the
                velocity deficits at each grid point in the flow field.
        """

        # NOTE: I wonder if we could construct a turbine object that holds the
        # necessary parameters to get around needing to pass a bunch of turbine
        # parameters into any given function.

        # added turbulence model
        TI = turbine._turbulence_intensity

        # turbine parameters
        D = turbine.rotor_diameter

        yaw = -1 * turbine.yaw_angle  # opposite sign convention in this model
        Ct = turbine.Ct
        U_local = u_initial
        local_x = x_locations - turbine_coord.x1
        local_y = y_locations - turbine_coord.x2
        local_z = z_locations - turbine_coord.x3  # adjust for hub height

        # coordinate info
        r = np.sqrt(local_y ** 2 + (local_z) ** 2)

        kstar = self._calculate_model_parameters(self.kstar, Ct, TI)
        epsilon = self._calculate_model_parameters(self.epsilon, Ct, TI)
        a = self._calculate_model_parameters(self.a, Ct, TI)
        b = self._calculate_model_parameters(self.b, Ct, TI)
        c = self._calculate_model_parameters(self.c, Ct, TI)

        k1 = np.cos(np.pi / 2 * (r / D - 0.5)) ** 2
        k1[r / D > 0.5] = 1.0

        k2 = np.cos(np.pi / 2 * (r / D + 0.5)) ** 2
        k2[r / D > 0.5] = 1.0

        # Representative wake width = \sigma / D
        wake_width = kstar * (local_x / D) + epsilon

        # wake velocity deficit = \Delta U (x,y,z) / U_h
        C = 1 / (a + b * (local_x / D) + c * (1 + (local_x / D)) ** (-2)) ** 2
        r_tilde = r
        n = 2
        sigma_tilde = wake_width * D
        velDef = self.gaussian_function(U_local, C, r_tilde, n, sigma_tilde)

        # trim wakes to 1 D upstream to avoid artifacts
        yR = y_locations - turbine_coord.x2
        xR = yR * tand(yaw) + turbine_coord.x1 - D
        velDef[x_locations < xR] = 0

        return velDef, np.zeros(np.shape(velDef)), np.zeros(np.shape(velDef))
