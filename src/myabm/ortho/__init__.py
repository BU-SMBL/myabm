"""
The ortho submodule defines an agent based model for studying formation
and differentiation of skeletal cells/tissues

Ortho Objects
=============
.. autosummary::
    :toctree: generated/

    OrthoModel

Ortho Submodules
================
.. autosummary::
    :toctree: generated/
    
    actions
    geometries
    setup


"""

import copy
import numpy as np
from .. import Model, AgentGrid, Agent
from . import actions
from numba.experimental import jitclass
from numba import int32, int64, float32, float64, literal_unroll
from numba.typed import Dict
from numba.types import string, DictType
from mymesh import *

class OrthoModel(Model):
    """
    OrthoModel class - contains specialized setup for modeling skeletal tissues.
    """
    def __init__(self, Mesh=None, agent_grid=None, 
        model_parameters=None, grid_parameters=None, agent_parameters=None):
        
        self.model_parameters = dict()
        self.grid_parameters = dict()
        self.agent_parameters = dict()

        # Default parameters for OrthoModel:
        self.grid_parameters['Mineral Walk Threshold'] = 0.4  # mg/mm^3
        self.grid_parameters['Tissue Threshold'] = 0.2 # volume fraction
        self.grid_parameters['Max Mineral'] = 0.8 # mg/mm^3
        self.grid_parameters['Mineralization Rate'] = 669.25 # mg/mm^3/day?
        self.grid_parameters['Mineral Solute Concentration'] = 18.08e-5 # mg/mm^3?
        if grid_parameters is not None:
            for key in grid_parameters:
                self.grid_parameters[key] = grid_parameters[key]

        self.agent_parameters['Production'] = 5e-6 # mm^3
        self.agent_parameters['Kcurve'] = 2.5 # mm^-1
        self.agent_parameters['ncurve'] = 3.
        self.agent_parameters['Production Baseline'] = 0. # mm^3
        self.agent_parameters['ProlifRate'] = 0.05
        self.agent_parameters['ApopRate'] = 0.05
        self.agent_parameters['DiffRate'] = 0.05
        self.agent_parameters['MigrRate'] = 0.96 # mm/day
        if agent_parameters is not None:
            for key in agent_parameters:
                self.agent_parameters[key] = agent_parameters[key]

        if model_parameters is not None:
            for key in model_parameters:
                self.model_parameters[key] = model_parameters[key]

        if 'TimeStep' not in self.model_parameters:
            self.model_parameters['TimeStep'] = 1     # days
        if 'SubStep' not in self.model_parameters:
            self.model_parameters['SubStep'] = 0.02   # days

        super().__init__(Mesh, agent_grid, self.model_parameters, 
        self.grid_parameters, self.agent_parameters)

        self.agent_grid.TimeStep = self.model_parameters['SubStep']

        # Initialize element data fields
        for key in ['Volume Fraction', 'ECM Fraction', 'Fibrous Fraction', 'Cartilaginous Fraction', 'Osseous Fraction', 'Scaffold Fraction', 'Mineral Density']:
            if key not in self.agent_grid.ElemData:
                self.agent_grid.ElemData[key] = np.zeros(Mesh.NElem, dtype=np.float64)
        for key in ['Tissue Orientation', 'Tissue Orientation Orthogonal']:
            if key not in self.agent_grid.ElemData:
                self.agent_grid.ElemVectorData[key] = np.zeros((Mesh.NElem,3), dtype=np.float64)
        self.agent_grid.ElemData['Volume'] = quality.Volume(Mesh.NodeCoords, Mesh.NodeConn)


    def default_schedule(self):
        
        self.agent_grid.TimeStep = self.model_parameters['SubStep'] # ensure time steps are synchronized
        nsubstep = int(np.round(self.model_parameters['TimeStep']/self.model_parameters['SubStep']))
        
        self.agent_grid.run_grid(self.grid_actions)
        for f in self.model_actions:
            f(self)

        for substep in range(nsubstep):
            self.agent_grid.run_agents(self.agent_actions)
            self.agent_grid.run_grid((actions.update_mineral,))

            # update history
            self.history['Agent Nodes'].append(self.agent_nodes)
            self.history['Agent States'].append(self.agent_states)
            self.history['ElemData'].append(copy.deepcopy(self.ElemData))
            self.history['NodeData'].append(copy.deepcopy(self.NodeData))

        