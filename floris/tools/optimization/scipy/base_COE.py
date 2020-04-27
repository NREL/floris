# Copyright 2020 NREL
 
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0
 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
 
# See https://floris.readthedocs.io for documentation

import numpy as np


class BaseCOE():
    """
    BaseCOE is the base cost of energy (COE) class that is used to determine
    the cost of energy associated with a 
    :py:class:`~.optimization.scipy.layout_height.LayoutHeightOptimization`
    object.

    TODO: 1) Add references to NREL 2016 Cost of Wind Energy Review throughout?
    """
    def __init__(self, opt_obj):
        """
        Instantiate a COE model object with a LayoutHeightOptimization object.
        
        Args:
            opt_obj (:py:class:`~.layout_height.LayoutHeightOptimization`):
            The optimization object.
        """
        self.opt_obj = opt_obj

    # Public methods

    def FCR(self):
        """
        This method returns the fixed charge rate used in the COE calculation.

        Returns:
            float: The fixed charge rate.
        """
        return 0.079 # % - Taken from 2016 Cost of Wind Energy Review

    def TCC(self, height):
        """
        This method dertermines the turbine capital costs (TCC), 
        calculating the effect of varying turbine height and rotor 
        diameter on the cost of the tower. The relationship estiamted 
        the mass of steel needed for the tower from the NREL Cost and 
        Scaling Model (CSM), and then adds that to the tower cost 
        portion of the TCC. The proportion is determined from the NREL 
        2016 Cost of Wind Energy Review. A price of 3.08 $/kg is 
        assumed for the needed steel. Tower height is passed directly 
        while the turbine rotor diameter is pulled directly from the 
        turbine object within the 
        :py:class:`~.tools.floris_interface.FlorisInterface`:.

        TODO: Turbine capital cost or tower capital cost?
        
        Args:
            height (float): Turbine hub height in meters.
        
        Returns:
            float: The turbine capital cost of a wind plant in units of $/kWh.
        """
        # From CSM with a fudge factor
        tower_mass = (0.2694*height*(np.pi* \
            (self.opt_obj.fi.floris.farm.turbines[0].rotor_diameter/2)**2) \
            + 1779.3)/(1.341638)

        # Combo of 2016 Cost of Wind Energy Review and CSM
        TCC = 831 + tower_mass * 3.08 \
            * self.opt_obj.nturbs / self.opt_obj.plant_kw

        return TCC

    def BOS(self):
        """
        This method returns the balance of station cost of a wind plant as
        determined by a constant factor. As the rating of a wind plant grows,
        the cost of the wind plant grows as well.
        
        Returns:
            float: The balance of station cost of a wind plant in units of 
            $/kWh.
        """
        return 364. # $/kW - Taken from 2016 Cost of Wind Energy Review

    def FC(self):
        """
        This method returns the finance charge cost of a wind plant as
        determined by a constant factor. As the rating of a wind plant grows,
        the cost of the wind plant grows as well.
        
        Returns:
            float: The finance charge cost of a wind plant in units of $/kWh.
        """
        return 155. # $/kW - Taken from 2016 Cost of Wind Energy Review

    def O_M(self):
        """
        This method returns the operational cost of a wind plant as determined
        by a constant factor. As the rating of a wind plant grows, the cost of
        the wind plant grows as well.
        
        Returns:
            float: The operational cost of a wind plant in units of $/kWh.
        """
        return 52. # $/kW - Taken from 2016 Cost of Wind Energy Review

    def COE(self, height, AEP_sum):
        """
        This method calculates and returns the cost of energy of a wind plant.
        This cost of energy (COE) formulation for a wind plant varies based on
        turbine height, rotor diameter, and total annualized energy production
        (AEP). The components of the COE equation are defined throughout the
        BaseCOE class.
        
        Args:
            height (float): The hub height of the turbines in meters 
                (all turbines are set to the same height).
            AEP_sum (float): The annualized energy production (AEP) 
                for the wind plant as calculated across the wind rose 
                in kWh.
        
        Returns:
            float: The cost of energy for a wind plant in units of 
            $/kWh.
        """
        # Comptue Cost of Energy (COE) as $/kWh for a plant
        return (self.FCR()*(self.TCC(height) + self.BOS() + self.FC()) \
                + self.O_M()) / (AEP_sum/1000/self.opt_obj.plant_kw)