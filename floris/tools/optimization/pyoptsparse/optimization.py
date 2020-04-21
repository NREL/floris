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
from ....utilities import setup_logger


class Optimization():
    """
    Base optimization class.

    Args:
        fi (:py:class:`floris.tools.floris_utilities.FlorisInterface`): 
            Interface from FLORIS to the tools package.

    Returns:
        Optimization: An instantiated Optimization object.
    """

    def __init__(self, model, solver=None):
        """
        Instantiate Optimization object and its parameters.
        """
        self.logger = setup_logger(name=__name__)

        self.model = model
        self.solver_choices = ['SNOPT', 'IPOPT', 'SLSQP', 'NLPQLP',
                             'FSQP', 'NSGA2', 'PSQP', 'ParOpt',
                             'CONMIN', 'ALPSO']

        if solver not in self.solver_choices:
            raise ValueError('Solver must be one supported by pyOptSparse: ' +
                            str(self.solver_choices))

        self.reinitialize(solver=solver)

    # Private methods

    def _reinitialize(self, solver=None, optOptions=None):
        try:
            import pyoptsparse
        except ImportError:
            err_msg = ('It appears you do not have pyOptSparse installed. ' + \
                'Please refer to https://pyoptsparse.readthedocs.io/ for ' + \
                'guidance on how to properly install the module.')
            self.logger.error(err_msg, stack_info=True)
            raise ImportError(err_msg)
        
        self.optProb = pyoptsparse.Optimization(self.model, self.objective_func)

        self.optProb = self.model.add_var_group(self.optProb)
        self.optProb = self.model.add_con_group(self.optProb)
        self.optProb.addObj('obj')

        if solver is not None:
            self.solver = solver
            print('Setting up optimization with user\'s choice of solver: ', 
                  self.solver)
        else:
            self.solver = 'SLSQP'
            print('Setting up optimization with default solver: SLSQP.')
        if optOptions is not None:
            self.optOptions = optOptions
        else:
            self.optOptions = {'Major feasibility tolerance' : 1e-1}

        exec('self.opt = pyoptsparse.' + self.solver + \
             '(optOptions=self.optOptions)')

    def _optimize(self):
        if hasattr(self.model, '_sens'):
            self.sol = self.opt(self.optProb, sens=self.model._sens)
        else:
            self.sol = self.opt(self.optProb, sens='FDR')

    # Public methods

    def reinitialize(self, solver=None):
        self._reinitialize(solver=solver)

    def optimize(self):
        self._optimize()

        return self.sol

    def objective_func(self, varDict):
        return self.model.obj_func(varDict)

    def sensitivity_func(self):
        pass

    # Properties

    