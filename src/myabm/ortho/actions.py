#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Mon Apr 28 12:31:40 2025
# @author: toj
"""

Actions for the Ortho Model
===========================

Ortho Agent Actions
-------------------
.. autosummary::
    :toctree: submodules/

    migrate
    migrate_curvotaxis
    proliferate
    apoptose
    produce
    produce_oriented
    differentiate_prendergast
    differentiate_carter

Ortho Grid Actions
------------------
.. autosummary::
    :toctree: submodules/

    update_mineral

Ortho Model Actions
-------------------
.. autosummary::
    :toctree: submodules/

    update_curvature
    update_scaffold_curvature

"""

import sys
import numpy as np
import numba
from numba.experimental import jitclass
from numba import int32, float32, float64
from numba.typed import Dict
from numba.types import string, DictType
import mymesh
from mymesh import *
import scipy
from . import mechanobiology

@numba.njit
def null(agent, grid):
    """
    Do nothing function, used for debugging

    Parameters
    ----------
    agent : myabm.Agent
        Cell agent to perform the action.
    grid : myabm.AgentGrid
        Grid environment that the agent is in.

    """
    
    pass

@numba.njit
def migrate(agent, grid):
    """
    Random walk migration.

    See also: :ref:`Migration - Random Walk`

    Parameters
    ----------
    agent : myabm.Agent
        Cell agent to perform the action.
    grid : myabm.AgentGrid
        Grid environment that the agent is in.

    """

    for required in ['MigrRate']:
        assert required in agent.parameters, 'agent.parameters must contain "'+required+'"'
    for required in ['Mineral Walk Threshold', 'Tissue Threshold', 'h']:
        assert required in grid.parameters, 'agent_grid.parameters must contain "'+required+'"'
    for required in ['Mineral Density', 'Volume Fraction']:
        assert required in grid.ElemData, 'agent_grid.ElemData must contain "'+required+'"'
    for required in ['Cells Allowed']:
        assert required in grid.NodeData, 'agent_grid.NodeData must contain "'+required+'"'

    TimeStep = grid.TimeStep
    StepSize = grid.parameters['h']
    MineralWalkThreshold = grid.parameters['Mineral Walk Threshold']
    ECMWalkThreshold = grid.parameters['Tissue Threshold']
    
    # Determine which edges are traversable
    # if agent.state == 'msc':
    edge_ids = grid.NodeEdgeConn[agent.node]
    edge_ids = edge_ids[edge_ids!=-1]
    edge_set = set(edge_ids)
    for eid in np.copy(edge_ids):
        elems = grid.EdgeElemConn[eid]
        elems = elems[elems!=-1]
        if np.all(grid.ElemData['Mineral Density'][elems] >= MineralWalkThreshold):
            # edge pass through mineralized elements - can't be traversed 
            edge_set.remove(eid)
        elif np.all(grid.ElemData['Volume Fraction'][elems] <= ECMWalkThreshold):
            # edge passes through space with insufficient material
            edge_set.remove(eid)
    
    MigrationProbability = 1 - np.exp(-agent.parameters['MigrRate']/StepSize*TimeStep) 
    if np.random.rand() < MigrationProbability:            
        if len(edge_set) > 0:
            # neighbors that are reachable and not occupied
            open_neighbors = np.array([n for n in np.unique(grid.Edges[np.array(list(edge_set))]) if (n not in grid.NodeAgents) and (grid.NodeData['Cells Allowed'][n] == 1)])
            # TODO: modify this to allow for swapping/
            if len(open_neighbors) > 0:
                step = np.random.choice(open_neighbors) # Choose new node
                grid.move_agent(agent, step)

@numba.njit    
def migrate_curvotaxis(agent, grid):
    """
    Curvotactic walk migration.
    Random walk weighted by surface curvature, based on :cite:t:`Pieuchot2018a`

    See also: :ref:`Curvotaxis`

    Parameters
    ----------
    agent : myabm.Agent
        Cell agent to perform the action.
    grid : myabm.AgentGrid
        Grid environment that the agent is in.

    """

    TimeStep = grid.TimeStep
    StepSize = grid.parameters['h']
    MineralWalkThreshold = grid.parameters['Mineral Walk Threshold']
    ECMWalkThreshold = grid.parameters['Tissue Threshold']
    
    # Determine which edges are traversable
    # if agent.state == 'msc':
    edge_ids = grid.NodeEdgeConn[agent.node]
    edge_ids = edge_ids[edge_ids!=-1]
    edge_set = set(edge_ids)
    for eid in np.copy(edge_ids):
        elems = grid.EdgeElemConn[eid]
        elems = elems[elems!=-1]
        if np.all(grid.ElemData['Mineral Density'][elems] >= MineralWalkThreshold):
            # edge pass through mineralized elements - can't be traversed 
            edge_set.remove(eid)
        elif np.all(grid.ElemData['Volume Fraction'][elems] <= ECMWalkThreshold):
            # edge passes through space with insufficient material
            edge_set.remove(eid)
    
    # step = agent.node
    
    H = grid.NodeData['Mean Curvature'][agent.node]
    if np.isnan(H):
        # Decide what to do if cell moves into an area where curvature hasn't been defined yet
        H = 0
    # a, b, c = (4.2184265 ,  1.08271814, 19.81092902) # from curve fitting
    a = agent.parameters['MigrationWeight0']
    b = agent.parameters['MigrationWeight1']
    c = agent.parameters['MigrationWeight2']
    # MigrationProbability = np.minimum((a * b**H + c) / 1000 * 24 / agent.parameters['MscMaxMigrRate'], 1)
    k = agent.parameters['MigrRate']*(a * b**H + c) # migration rate (mm/day)
    Lambda = k/StepSize # migration event rate (/day)
    MigrationProbability = 1 - np.exp(-Lambda*TimeStep)
    if np.random.rand() < MigrationProbability:             
        if len(edge_set) > 0:
            # neighbors that are reachable and not occupied
            open_neighbors = np.array([n for n in np.unique(grid.Edges[np.array(list(edge_set))]) if (n not in grid.NodeAgents) and (grid.NodeData['Cells Allowed'][n] == 1)])
            # TODO: modify this to allow for swapping?
            
            
            if len(open_neighbors) > 0:
                
                a1 = agent.parameters['MigrationWeight3']
                b1 = agent.parameters['MigrationWeight4']
                # c1 = agent.parameters['MigrationWeight2'] 
                p = np.ones(len(open_neighbors))
                for i,n in enumerate(open_neighbors):
                    
                    H = grid.NodeData['Mean Curvature'][n]
                    # Logistic function 
                    p[i] = 1 - (1/(1 + np.exp(-a1*(H - b1))))
                    # Exponential equation
                    # p[i] = a1 * b1**H + c1
                    if np.isnan(p[i]):
                        p[i] = 0
                
                if np.sum(p) != 0:
                    p /= np.sum(p)
                else:
                    p = np.ones(len(open_neighbors))/len(open_neighbors)
                P = np.cumsum(p)
                step = open_neighbors[np.where(np.random.random() <= P)[0][0]]
                # step = np.random.choice(open_neighbors)#, p = p) # Choose new node
                    
                # Update data structure
                grid.move_agent(agent, step)
      
@numba.njit
def proliferate(agent, grid):
    """
    Cell proliferation. Random chance of creation of a new cell at an
    available adjacent site.

    See also: :ref:`Cell Dynamics - Proliferation & Apoptosis`

    Parameters
    ----------
    agent : myabm.Agent
        Cell agent to perform the action.
    grid : myabm.AgentGrid
        Grid environment that the agent is in.

    """
    for required in ['ProlifRate']:
        assert required in agent.parameters, 'agent.parameters must contain "'+required+'"'
    for required in ['Mineral Walk Threshold', 'Tissue Threshold', 'h']:
        assert required in grid.parameters, 'agent_grid.parameters must contain "'+required+'"'
    for required in ['Mineral Density', 'Volume Fraction']:
        assert required in grid.ElemData, 'agent_grid.ElemData must contain "'+required+'"'
    for required in ['Cells Allowed']:
        assert required in grid.NodeData, 'agent_grid.NodeData must contain "'+required+'"'

    TimeStep = grid.TimeStep
    MineralWalkThreshold = grid.parameters['Mineral Walk Threshold']
    ECMWalkThreshold = grid.parameters['Tissue Threshold']
    ProlifProb = 1 - np.exp(-agent.parameters['ProlifRate'] * TimeStep)
    
    # NOTE: This is doing redundant work with walk
    # Determine which edges are open 
    edge_ids = grid.NodeEdgeConn[agent.node]
    edge_ids = edge_ids[edge_ids!=-1]
    edge_set = set(edge_ids)
    for eid in np.copy(edge_ids):
        elems = grid.EdgeElemConn[eid]
        elems = elems[elems!=-1]
        if np.all(grid.ElemData['Mineral Density'][elems] >= MineralWalkThreshold):
            # edge pass through mineralized elements - can't be traversed 
            edge_set.remove(eid)
        elif np.all(grid.ElemData['Volume Fraction'][elems] <= ECMWalkThreshold):
            # edge passes through space with insufficient material
            edge_set.remove(eid)
        # elif agent.state == 'osteoblast' and np.all(grid.ElemData['Mineral Density'][elems] <= MineralWalkThreshold):
        #     # edge pass through only unmineralized elements - can't be traversed by osteoblasts
        #     edge_set.remove(eid)
            
    if len(edge_set) > 0:
        # neighbors that are reachable and not occupied
        open_neighbors = np.array([n for n in np.unique(grid.Edges[np.array(list(edge_set))]) if (n not in grid.NodeAgents) and (grid.NodeData['Cells Allowed'][n] == 1)])

        open_neighbors = open_neighbors[open_neighbors != agent.node]
        if len(open_neighbors) == 0:
            return
        
        if np.random.rand() < ProlifProb:
            step = np.random.choice(open_neighbors)
            grid.add_agent(step, agent.state, agent.parameters)

@numba.njit             
def proliferate_calcium(agent, grid):
    
    TimeStep = grid.TimeStep
    MineralWalkThreshold = grid.parameters['Mineral Walk Threshold']
    ECMWalkThreshold = grid.parameters['Tissue Threshold']
    ProlifProb = 1 - np.exp(-agent.parameters['ProlifRate'] * TimeStep)
    Calcium = grid.NodeData['Ca2+ Concentration']
    # NOTE: This is doing redundant work with walk
    # Determine which edges are open 
    edge_ids = grid.NodeEdgeConn[agent.node]
    edge_ids = edge_ids[edge_ids!=-1]
    edge_set = set(edge_ids)
    for eid in np.copy(edge_ids):
        elems = grid.EdgeElemConn[eid]
        elems = elems[elems!=-1]
        if np.all(grid.ElemData['Mineral Density'][elems] >= MineralWalkThreshold):
            # edge pass through mineralized elements - can't be traversed 
            edge_set.remove(eid)
        elif np.all(grid.ElemData['Volume Fraction'][elems] <= ECMWalkThreshold):
            # edge passes through space with insufficient material
            edge_set.remove(eid)
        elif agent.state == 'osteoblast' and np.all(grid.ElemData['Mineral Density'][elems] <= MineralWalkThreshold):
            # edge pass through only unmineralized elements - can't be traversed by osteoblasts
            edge_set.remove(eid)
            
    if len(edge_set) > 0:
        # neighbors that are reachable and not occupied
        open_neighbors = np.array([n for n in np.unique(grid.Edges[np.array(list(edge_set))]) if (n not in grid.NodeAgents) and (grid.NodeData['Cells Allowed'][n] == 1)])

        open_neighbors = open_neighbors[open_neighbors != agent.node]
        if len(open_neighbors) == 0:
            return
        
        if np.random.rand() < ProlifProb:
            step = np.random.choice(open_neighbors)
            grid.add_agent(step, agent.state, agent.age, agent.parameters)
                
@numba.njit
def apoptose(agent, grid):
    """
    Cell apoptosis. Random change that the agent will undergo apoptosis and be
    removed from the grid. 

    See also: :ref:`Cell Dynamics - Proliferation & Apoptosis`

    Parameters
    ----------
    agent : myabm.Agent
        Cell agent to perform the action.
    grid : myabm.AgentGrid
        Grid environment that the agent is in.

    """    
    for required in ['ApopRate']:
        assert required in agent.parameters, 'agent.parameters must contain "'+required+'"'
    TimeStep = grid.TimeStep
    ApopProb = 1 - np.exp(-agent.parameters['ApopRate'] * TimeStep)
    if (np.random.rand() < ApopProb):
        grid.remove_agent(agent)
            
@numba.njit
def differentiate_prendergast(agent, grid):
    r"""
    Mechanobiological cell differentiation based on :cite:t:`Prendergast1997a`. 
    :code:`'Stimulus'` must be stored in :code:`grid.NodeData` and should be the 
    stimulus value described by :cite:t:`Huiskes1997`:

    .. math::
        
        S = \frac{\gamma}{0.0375} + \frac{v}{0.003}
    
    where :math:`\gamma` is octahedral shear strain and :math:`v` is 
    interstitial fluid flow velocity from a poroelastic material model.

    See also: :ref:`Differentiation`

    Parameters
    ----------
    agent : myabm.Agent
        Cell agent to perform the action.
    grid : myabm.AgentGrid
        Grid environment that the agent is in.
    """    

    for required in ['DiffRate', 'DiffMaturity']:
        assert required in agent.parameters, 'agent.parameters must contain "'+required+'"'
    for required in ['Stimulus']:
        assert required in grid.NodeData, 'agent_grid.NodeData must contain "'+required+'"'

    TimeStep = grid.TimeStep
    DiffProb = 1 - np.exp(-agent.parameters['DiffRate'] * TimeStep)
    
    S = grid.NodeData['Stimulus'][agent.node]
    if agent.state == 'msc':
        # stimulus should be "S" from the prendergast model
        if (agent.age > agent.parameters['DiffMaturity']) & (np.random.rand() < DiffProb):
            if 0.01 <= S <= 1:
                agent.state = 'osteoblast'
                agent.age = 0
                agent.parameters['ProlifRate'] = 0.3 # Isaksson 2008
                agent.parameters['ApopRate'] = 0.16 # Isaksson 2008
                agent.parameters['MigrRate'] = 0.167 # proportional to Isaksson 2008
                agent.parameters['Production'] = 3e-6 # (mm^3/cell/day) Isaksson 2008
                agent.parameters['DiffRate'] = 0.1
                
            elif 1 < S <= 3:
                agent.state = 'chondrocyte'
                agent.age = 0
                agent.parameters['ProlifRate'] = 0.2 # Isaksson 2008
                agent.parameters['ApopRate'] = 0.1 # Isaksson 2008
                agent.parameters['MigrRate'] = 0.05 # Checa 2011 (See also Morales 2007)
                agent.parameters['Production'] = 5e-6 # (mm^3/cell/day) Isaksson 2008
                
            elif 3 < S:
                agent.state = 'fibroblast'
                agent.age = 0
                agent.parameters['ProlifRate'] = 0.55 # Isaksson 2008
                agent.parameters['ApopRate'] = 0.05 # Isaksson 2008
                agent.parameters['MigrRate'] = 0.667 # proportional to Isaksson 2008
                agent.parameters['Production'] = 5e-6 # (mm^3/cell/day) Isaksson 2008
    elif agent.state == 'osteoblast':
        if np.all(grid.ElemData['Mineral Density'][grid.ElemConn[agent.node]] > .18) and np.random.rand() < DiffProb:
            agent.state = 'osteocyte'
            agent.parameters['ProlifRate'] = 0.
            agent.parameters['ApopRate'] = 0.
            agent.parameters['Production'] = 1e-6
        else:
            if 0.01 <= S <= 1:
                agent.parameters['ApopRate'] = 0.16
                agent.parameters['Production'] = 3e-6
            else:
                agent.parameters['ApopRate'] = 0.3
                agent.parameters['Production'] = 0
    elif agent.state == 'chondrocyte':
        if 1 < S <= 3:
            agent.parameters['ApopRate'] = 0.1
            agent.parameters['Production'] = 5e-6
        else:
            agent.parameters['ApopRate'] = 0.3
            agent.parameters['Production'] = 0
    elif agent.state == 'fibroblast':
        if 3 < S:
            agent.parameters['ApopRate'] = 0.05
            agent.parameters['Production'] = 5e-6
        else:
            agent.parameters['ApopRate'] = 0.3
            agent.parameters['Production'] = 0

@numba.njit
def differentiate_carter(agent, grid):
    r"""
    Mechanobiological cell differentiation based on :cite:t:`Carter1998`. Theshold values are from :cite:t:`Isaksson2006`, :cite:t:`Isaksson2006a`. 
    :code:`'Hydrostatic Stress'` must be stored in :code:`grid.NodeData` and 
    :code:`'Principal Strain'` must be stored in :code:`grid.NodeVectorData`. 
    Hydrostatic Stress is expected in units of MPa and principal strains should not be given as percentages.

    See also: :ref:`Differentiation`

    Parameters
    ----------
    agent : myabm.Agent
        Cell agent to perform the action.
    grid : myabm.AgentGrid
        Grid environment that the agent is in.
    """    

    for required in ['DiffRate', 'DiffMaturity']:
        assert required in agent.parameters, 'agent.parameters must contain "'+required+'"'
    for required in ['Hydrostatic Stress']:
        assert required in grid.NodeData, 'agent_grid.NodeData must contain "'+required+'"'
    for required in ['Principal Strain']:
        assert required in grid.NodeVectorData, 'agent_grid.NodeVectorData must contain "'+required+'"'

    TimeStep = grid.TimeStep
    DiffProb = 1 - np.exp(-agent.parameters['DiffRate'] * TimeStep)
    
    stress = grid.NodeData['Hydrostatic Stress'][agent.node]
    strain = np.max(grid.NodeVectorData['Principal Strain'][agent.node])
    if agent.state == 'msc':
        
        if (agent.age > agent.parameters['DiffMaturity']) & (np.random.rand() < DiffProb):
            if stress <=  0.2 and strain <= 5/100:
                agent.state = 'osteoblast'
                agent.age = 0
                agent.parameters['ProlifRate'] = 0.3 # Isaksson 2008
                agent.parameters['ApopRate'] = 0.16 # Isaksson 2008
                agent.parameters['MigrRate'] = 0.167 # proportional to Isaksson 2008
                agent.parameters['Production'] = 3e-6 # (mm^3/cell/day) Isaksson 2008
                agent.parameters['DiffRate'] = 0.1
                
            elif stress > 0.2:
                agent.state = 'chondrocyte'
                agent.age = 0
                agent.parameters['ProlifRate'] = 0.2 # Isaksson 2008
                agent.parameters['ApopRate'] = 0.1 # Isaksson 2008
                agent.parameters['MigrRate'] = 0.05 # Checa 2011 (See also Morales 2007)
                agent.parameters['Production'] = 5e-6 # (mm^3/cell/day) Isaksson 2008
                
            elif stress <=  0.2 and strain > 5/100:
                agent.state = 'fibroblast'
                agent.age = 0
                agent.parameters['ProlifRate'] = 0.55 # Isaksson 2008
                agent.parameters['ApopRate'] = 0.05 # Isaksson 2008
                agent.parameters['MigrRate'] = 0.667 # proportional to Isaksson 2008
                agent.parameters['Production'] = 5e-6 # (mm^3/cell/day) Isaksson 2008
    elif agent.state == 'osteoblast':
        if np.all(grid.ElemData['Mineral Density'][grid.ElemConn[agent.node]] > .18) and np.random.rand() < DiffProb:
            agent.state = 'osteocyte'
            agent.parameters['ProlifRate'] = 0.
            agent.parameters['ApopRate'] = 0.
            agent.parameters['Production'] = 1e-6
        else:
            if stress <=  0.2 and strain <= 5/100:
                agent.parameters['ApopRate'] = 0.16
                agent.parameters['Production'] = 3e-6
            else:
                agent.parameters['ApopRate'] = 0.3
                agent.parameters['Production'] = 0
    elif agent.state == 'chondrocyte':
        if stress > 0.2:
            agent.parameters['ApopRate'] = 0.1
            agent.parameters['Production'] = 5e-6
        else:
            agent.parameters['ApopRate'] = 0.3
            agent.parameters['Production'] = 0
    elif agent.state == 'fibroblast':
        if stress <=  0.2 and strain > 5/100:
            agent.parameters['ApopRate'] = 0.05
            agent.parameters['Production'] = 5e-6
        else:
            agent.parameters['ApopRate'] = 0.3
            agent.parameters['Production'] = 0

@numba.njit
def produce(agent, grid):
    """
    Constant tissue production. Cells will fill surrounding elements with 
    tissue. Once newly created tissue has reached 
    :code:`grid.parameters['Tissue Threshold']`, the cell will move to the new
    surface, as if the cell is creating tissue beneath it.

    See also: :ref:`Tissue Production`

    Parameters
    ----------
    agent : myabm.Agent
        Cell agent to perform the action.
    grid : myabm.AgentGrid
        Grid environment that the agent is in.
    """

    for required in ['Production']:
        assert required in agent.parameters, 'agent.parameters must contain "'+required+'"'
    for required in ['Mineral Walk Threshold', 'Tissue Threshold', 'h']:
        assert required in grid.parameters, 'agent_grid.parameters must contain "'+required+'"'
    for required in ['Mineral Density', 'Volume Fraction', 'Fibrous Fraction', 'ECM Fraction', 'Cartilaginous Fraction', 'Osseous Fraction']:
        assert required in grid.ElemData, 'agent_grid.ElemData must contain "'+required+'"'

    TimeStep = grid.TimeStep
    Production = TimeStep*agent.parameters['Production']
    
    neighbor_elems = grid.ElemConn[agent.node]
    neighbor_elems = neighbor_elems[neighbor_elems >= 0]
    
    # ECM production
    ProducedFraction = np.minimum(Production/grid.parameters['Volume'], 1) # Production can never exceed the total element volume
    
    if agent.state == 'msc':
        available = 1 - grid.ElemData['Volume Fraction'][neighbor_elems]
    elif agent.state == 'fibroblast':
        available = (1 - grid.ElemData['Volume Fraction'][neighbor_elems]) + (1 - grid.ElemData['ECM Fraction'][neighbor_elems])
    elif agent.state == 'chondrocyte':
        available = (1 - grid.ElemData['Volume Fraction'][neighbor_elems]) + (1 - grid.ElemData['ECM Fraction'][neighbor_elems]) + (1 - grid.ElemData['Fibrous Fraction'][neighbor_elems])
    elif agent.state == 'osteoblast':
        available = (1 - grid.ElemData['Volume Fraction'][neighbor_elems]) + (1 - grid.ElemData['ECM Fraction'][neighbor_elems]) + (1 - grid.ElemData['Fibrous Fraction'][neighbor_elems]) + (1 - grid.ElemData['Cartilaginous Fraction'][neighbor_elems])
    total = np.sum(available)
    
    # Volume fraction update
    if total == 0:
        ElemProducedFraction = np.zeros_like(available)
    else:
        ElemProducedFraction = ProducedFraction*(available/total)
    
    ElemProducedFraction[np.isnan(ElemProducedFraction)] = 0
    NewProduction = np.minimum(1 - grid.ElemData['Volume Fraction'][neighbor_elems], ElemProducedFraction)
    grid.ElemData['Volume Fraction'][neighbor_elems] += NewProduction
    
    # TODO: Track/update specific tissue types
    if agent.state == 'msc':
        grid.ElemData['ECM Fraction'][neighbor_elems] += NewProduction
        # MSCs can't remodel any other tissue types
        
    elif agent.state == 'fibroblast':
        ConvertibleFraction = np.minimum(ElemProducedFraction - NewProduction, grid.ElemData['ECM Fraction'][neighbor_elems]) # Amount of tissue of a different type convertible to Fibrous tissue
        grid.ElemData['Fibrous Fraction'][neighbor_elems] += ConvertibleFraction + NewProduction
        
        Converted_ECM = np.minimum(grid.ElemData['ECM Fraction'][neighbor_elems], ConvertibleFraction)
        grid.ElemData['ECM Fraction'][neighbor_elems] -= Converted_ECM
        
    elif agent.state == 'chondrocyte':
        ConvertibleFraction = np.minimum(ElemProducedFraction - NewProduction, grid.ElemData['ECM Fraction'][neighbor_elems] + grid.ElemData['Fibrous Fraction'][neighbor_elems]) # Amount of tissue of a different type convertible to Fibrous tissue
        grid.ElemData['Cartilaginous Fraction'][neighbor_elems] += ConvertibleFraction + NewProduction
        
        # First ECM is converted, then fibrous if there is still remaining produced cartilaginous tissue
        Converted_ECM = np.minimum(grid.ElemData['ECM Fraction'][neighbor_elems], ConvertibleFraction)
        grid.ElemData['ECM Fraction'][neighbor_elems] -= Converted_ECM 

        Converted_Fib = np.minimum(grid.ElemData['Fibrous Fraction'][neighbor_elems], (ConvertibleFraction-Converted_ECM))
        grid.ElemData['Fibrous Fraction'][neighbor_elems] -= Converted_Fib
        
    elif agent.state == 'osteoblast':
        ConvertibleFraction = np.minimum(ElemProducedFraction - NewProduction, grid.ElemData['ECM Fraction'][neighbor_elems] + grid.ElemData['Fibrous Fraction'][neighbor_elems] + grid.ElemData['Cartilaginous Fraction'][neighbor_elems]) # Amount of tissue of a different type convertible to Fibrous tissue
        grid.ElemData['Osseous Fraction'][neighbor_elems] += ConvertibleFraction + NewProduction
        
        # First ECM is converted, then fibrous, then cartilage if there is still remaining produced osseous tissue
        Converted_ECM = np.minimum(grid.ElemData['ECM Fraction'][neighbor_elems], ConvertibleFraction)
        grid.ElemData['ECM Fraction'][neighbor_elems] -= Converted_ECM 

        Converted_Fib = np.minimum(grid.ElemData['Fibrous Fraction'][neighbor_elems], (ConvertibleFraction-Converted_ECM))
        grid.ElemData['Fibrous Fraction'][neighbor_elems] -= Converted_Fib

        Converted_Cart = np.minimum(grid.ElemData['Cartilaginous Fraction'][neighbor_elems], (ConvertibleFraction-Converted_ECM-Converted_Fib))
        grid.ElemData['Cartilaginous Fraction'][neighbor_elems] -= Converted_Cart
        
    # automatically move cells to new surface
    new_voxels = np.where((grid.ElemData['Volume Fraction'][neighbor_elems]-NewProduction < grid.parameters['Tissue Threshold']) & (grid.ElemData['Volume Fraction'][neighbor_elems] >= grid.parameters['Tissue Threshold']))[0]
    old_voxels = np.where((grid.ElemData['Volume Fraction'][neighbor_elems]-NewProduction > grid.parameters['Tissue Threshold'] ) | (grid.ElemData['Mineral Density'][neighbor_elems] > grid.parameters['Mineral Walk Threshold']))[0]
    if len(new_voxels) > 0:
        if len(old_voxels) > 0:
            new_nodes = np.array([n for e in new_voxels for n in grid.NodeConn[neighbor_elems[e]]]).flatten()
            old_nodes = np.array([n for e in old_voxels for n in grid.NodeConn[neighbor_elems[e]]]).flatten()#grid.NodeConn[neighbor_elems[old_voxels]].flatten()
            new_nodes = np.array([n for n in new_nodes if n not in old_nodes])
        else:
            new_nodes = np.array([n for e in new_voxels for n in grid.NodeConn[neighbor_elems[e]]]).flatten()# grid.NodeConn[neighbor_elems[new_voxels]].flatten()
        new_nodes = np.array([n for n in new_nodes if n not in grid.NodeAgents and grid.NodeData['Cells Allowed'][n]])
        # print(new_nodes)
        if len(new_nodes) > 0:
            grid.move_agent(agent, np.random.choice(new_nodes))
            # step = np.random.choice(new_nodes)
            # del grid.NodeAgents[agent.node]          # Remove cell from old node
            # grid.NodeAgents[step] = agent            # Move cell to new node
            # agent.node = step                        # Update cell's node id
      
@numba.njit
def produce_oriented(agent, grid):
    r"""
    Curvature-dependent production of oriented tissue. 
    Cells will fill surrounding elements with tissue oriented in the direction 
    of minimum principal curvature. Tissue production is dependent on local mean 
    curvature, according to the equation:
    
    .. math::

        V_{prod.}(H_i) = k_{prod} \frac{|\min(0, H_i)|^{n_H}}{K_H^{n_H} + |\min(0, H_i)|^{n_H}}
    
    Where :math:`H_{i}` is the mean curvature of the surface at the location of 
    the cell, :math:`k_{prod}` is a cell type-specific production rate constant 
    (:code:`agent.parameters['Production']`), :math:`K_H`  is the mean curvature 
    at which tissue production is half of the maximum rate 
    (:code:`agent.parameters['Kcurve']`), and :math:`n_H` is the Hill coefficient 
    (:code:`agent.parameters['ncurve']`).

    This curvature-dependent tissue production results in tissue production 
    consistent with the experimental observations of :cite:`Bidan2013`

    Once newly created tissue has reached 
    :code:`grid.parameters['Tissue Threshold']`, the cell will move to the new
    surface, as if the cell is creating tissue beneath it.

    See also: :ref:`Tissue Growth`

    Parameters
    ----------
    agent : myabm.Agent
        Cell agent to perform the action.
    grid : myabm.AgentGrid
        Grid environment that the agent is in.
    """

    for required in ['Production', 'Production Baseline', 'Kcurve', 'ncurve']:
        assert required in agent.parameters, 'agent.parameters must contain "'+required+'"'
    for required in ['Mineral Walk Threshold', 'Tissue Threshold']:
        assert required in grid.parameters, 'agent_grid.parameters must contain "'+required+'"'
    for required in ['Mineral Density', 'Volume Fraction', 'ECM Fraction', 'Fibrous Fraction', 'Cartilaginous Fraction', 'Osseous Fraction']:
        assert required in grid.ElemData, 'agent_grid.ElemData must contain "'+required+'"'
    for required in ['Min Principal Curvature','Max Principal Curvature','Mean Curvature','Cells Allowed']:
        assert required in grid.NodeData, 'agent_grid.NodeData must contain "'+required+'"'
    for required in ['Min Principal Direction','Max Principal Direction']:
        assert required in grid.NodeVectorData, 'agent_grid.NodeData must contain "'+required+'"'

    TimeStep = grid.TimeStep
    Production = TimeStep*agent.parameters['Production']
    Kcurve = agent.parameters['Kcurve']
    ncurve = agent.parameters['ncurve']
    
    neighbor_elems = grid.ElemConn[agent.node]
    neighbor_elems = neighbor_elems[neighbor_elems >= 0]
    
    # Mean Curvature-based tissue production
    a = np.abs(np.minimum(0, grid.NodeData['Mean Curvature'][agent.node]))
    CurvatureProduction = Production*a**ncurve / (Kcurve**ncurve + a**ncurve) + agent.parameters['Production Baseline']
    ProducedFraction = np.minimum(CurvatureProduction/grid.parameters['Volume'], 1)
    # 
    k2 = grid.NodeData['Min Principal Curvature'][agent.node]
    k2v = grid.NodeVectorData['Min Principal Direction'][agent.node]
    k1 = grid.NodeData['Max Principal Curvature'][agent.node]
    k1v = grid.NodeVectorData['Max Principal Direction'][agent.node]
    
    
    if np.isnan(k2):
        k2 = 0
        k2v = np.zeros(3)
    
    if np.isnan(k1):
        k1 = 0
        k1v = np.zeros(3)
        
    ExistingFiberOrientation =  np.array([0.,0.,0.])
    ExistingFiberOrthogonal =  np.array([0.,0.,0.])
    count = 0
    for n in neighbor_elems:
        if np.linalg.norm(grid.ElemVectorData['Tissue Orientation'][n]) != 0:
            count += 1 
            ExistingFiberOrientation += grid.ElemVectorData['Tissue Orientation'][n]
            ExistingFiberOrthogonal += grid.ElemVectorData['Tissue Orientation Orthogonal'][n]
    
    if count > 0:
        ExistingFiberOrientation /= count
        ExistingFiberOrthogonal /= count
    # ExistingFiberOrientation not normalized, so less aligned tissue will have less relative influence
    
    # In this set up, if there is no k1, k2 data (data is np.nan), the new orientation will follow the average of neighboring tissue orientations
    NewOrientation = (k1v + ExistingFiberOrientation)/2
    NewOrthogonal = (k2v + ExistingFiberOrthogonal)/2
    
    if agent.state == 'msc':
        available = 1 - grid.ElemData['Volume Fraction'][neighbor_elems]
    elif agent.state == 'fibroblast':
        available = (1 - grid.ElemData['Volume Fraction'][neighbor_elems]) + (1 - grid.ElemData['ECM Fraction'][neighbor_elems])
    elif agent.state == 'chondrocyte':
        available = (1 - grid.ElemData['Volume Fraction'][neighbor_elems]) + (1 - grid.ElemData['ECM Fraction'][neighbor_elems]) + (1 - grid.ElemData['Fibrous Fraction'][neighbor_elems])
    elif agent.state == 'osteoblast':
        available = (1 - grid.ElemData['Volume Fraction'][neighbor_elems]) + (1 - grid.ElemData['ECM Fraction'][neighbor_elems]) + (1 - grid.ElemData['Fibrous Fraction'][neighbor_elems]) + (1 - grid.ElemData['Cartilaginous Fraction'][neighbor_elems])
    elif agent.state == 'osteocyte':
        return
    total = np.sum(available)
    
    # Volume fraction update
    if total == 0:
        ElemProducedFraction = np.zeros_like(available)
    else:
        ElemProducedFraction = ProducedFraction*(available/total)
    
    ElemProducedFraction[np.isnan(ElemProducedFraction)] = 0
    NewProduction = np.minimum(1 - grid.ElemData['Volume Fraction'][neighbor_elems], ElemProducedFraction)
    
    if ProducedFraction > 0:
        grid.ElemVectorData['Tissue Orientation'][neighbor_elems] = (grid.ElemVectorData['Tissue Orientation'][neighbor_elems,:]*grid.ElemData['Volume Fraction'][neighbor_elems,None] + ElemProducedFraction[:,None] * k2v)/(grid.ElemData['Volume Fraction'][neighbor_elems,None] + ElemProducedFraction[:,None])
        grid.ElemVectorData['Tissue Orientation Orthogonal'][neighbor_elems] = (grid.ElemVectorData['Tissue Orientation Orthogonal'][neighbor_elems,:]*grid.ElemData['Volume Fraction'][neighbor_elems,None] + ElemProducedFraction[:,None] * k1v)/(grid.ElemData['Volume Fraction'][neighbor_elems,None] + ElemProducedFraction[:,None])

    grid.ElemData['Volume Fraction'][neighbor_elems] += NewProduction
    
    if agent.state == 'msc':
        grid.ElemData['ECM Fraction'][neighbor_elems] += NewProduction
        # MSCs can't remodel any other tissue types
        
    elif agent.state == 'fibroblast':
        ConvertibleFraction = np.minimum(ElemProducedFraction - NewProduction, grid.ElemData['ECM Fraction'][neighbor_elems]) # Amount of tissue of a different type convertible to Fibrous tissue
        grid.ElemData['Fibrous Fraction'][neighbor_elems] += ConvertibleFraction + NewProduction
        
        Converted_ECM = np.minimum(grid.ElemData['ECM Fraction'][neighbor_elems], ConvertibleFraction)
        grid.ElemData['ECM Fraction'][neighbor_elems] -= Converted_ECM
        
    elif agent.state == 'chondrocyte':
        ConvertibleFraction = np.minimum(ElemProducedFraction - NewProduction, grid.ElemData['ECM Fraction'][neighbor_elems] + grid.ElemData['Fibrous Fraction'][neighbor_elems]) # Amount of tissue of a different type convertible to Fibrous tissue
        grid.ElemData['Cartilaginous Fraction'][neighbor_elems] += ConvertibleFraction + NewProduction
        
        # First ECM is converted, then fibrous if there is still remaining produced cartilaginous tissue
        Converted_ECM = np.minimum(grid.ElemData['ECM Fraction'][neighbor_elems], ConvertibleFraction)
        grid.ElemData['ECM Fraction'][neighbor_elems] -= Converted_ECM 

        Converted_Fib = np.minimum(grid.ElemData['Fibrous Fraction'][neighbor_elems], (ConvertibleFraction-Converted_ECM))
        grid.ElemData['Fibrous Fraction'][neighbor_elems] -= Converted_Fib
        
    elif agent.state == 'osteoblast':
        ConvertibleFraction = np.minimum(ElemProducedFraction - NewProduction, grid.ElemData['ECM Fraction'][neighbor_elems] + grid.ElemData['Fibrous Fraction'][neighbor_elems] + grid.ElemData['Cartilaginous Fraction'][neighbor_elems]) # Amount of tissue of a different type convertible to Fibrous tissue
        grid.ElemData['Osseous Fraction'][neighbor_elems] += ConvertibleFraction + NewProduction
        
        # First ECM is converted, then fibrous, then cartilage if there is still remaining produced osseous tissue
        Converted_ECM = np.minimum(grid.ElemData['ECM Fraction'][neighbor_elems], ConvertibleFraction)
        grid.ElemData['ECM Fraction'][neighbor_elems] -= Converted_ECM 

        Converted_Fib = np.minimum(grid.ElemData['Fibrous Fraction'][neighbor_elems], (ConvertibleFraction-Converted_ECM))
        grid.ElemData['Fibrous Fraction'][neighbor_elems] -= Converted_Fib

        Converted_Cart = np.minimum(grid.ElemData['Cartilaginous Fraction'][neighbor_elems], (ConvertibleFraction-Converted_ECM-Converted_Fib))
        grid.ElemData['Cartilaginous Fraction'][neighbor_elems] -= Converted_Cart
    
    # automatically move cells to new surface
    new_voxels = np.where((grid.ElemData['Volume Fraction'][neighbor_elems]-NewProduction < grid.parameters['Tissue Threshold']) & (grid.ElemData['Volume Fraction'][neighbor_elems] >= grid.parameters['Tissue Threshold']))[0]
    old_voxels = np.where((grid.ElemData['Volume Fraction'][neighbor_elems]-NewProduction > grid.parameters['Tissue Threshold'] ) | (grid.ElemData['Mineral Density'][neighbor_elems] > grid.parameters['Mineral Walk Threshold']))[0]
    if len(new_voxels) > 0:
        if len(old_voxels) > 0:
            new_nodes = np.array([n for e in new_voxels for n in grid.NodeConn[neighbor_elems[e]]]).flatten()
            old_nodes = np.array([n for e in old_voxels for n in grid.NodeConn[neighbor_elems[e]]]).flatten()#grid.NodeConn[neighbor_elems[old_voxels]].flatten()
            new_nodes = np.array([n for n in new_nodes if n not in old_nodes])
        else:
            new_nodes = np.array([n for e in new_voxels for n in grid.NodeConn[neighbor_elems[e]]]).flatten()# grid.NodeConn[neighbor_elems[new_voxels]].flatten()
        new_nodes = np.array([n for n in new_nodes if n not in grid.NodeAgents and grid.NodeData['Cells Allowed'][n]])
        # print(new_nodes)
        if len(new_nodes) > 0:
            grid.move_agent(agent, np.random.choice(new_nodes))

# Grid Actions
@numba.njit
def update_mineral(grid):
    """
    Update mineralization of osseous tissue

    Parameters
    ----------
    grid : myabm.AgentGrid
        AgentGrid of the model
    
    Returns
    -------
        None.    
    """    
    dmineraldt = grid.parameters['Mineralization Rate'] * (grid.ElemData['Osseous Fraction'] - grid.ElemData['Mineral Density']/grid.parameters['Max Mineral']) * grid.parameters['Mineral Solute Concentration']
    dmineraldt[grid.ElemData['Scaffold Fraction'] == 1] = 0 # Assume no degradation of calcium from scaffold
    grid.ElemData['Mineral Density'] += dmineraldt*grid.TimeStep

@numba.njit
def degrade_tissue(grid):
    """
    Update mineralization of osseous tissue

    Parameters
    ----------
    grid : myabm.AgentGrid
        AgentGrid of the model
    
    Returns
    -------
        None.    
    """    
    degradation_rate = grid.parameters['Tissue Degradation']
    
    elem_stimulus = np.array([np.mean(grid.NodeData['Stimulus'][grid.NodeConn[i]]) for i in range(grid.NElem)])
    grid.ElemData['Osseous Fraction'][(elem_stimulus < .01) | (elem_stimulus > 1)] = np.maximum(grid.ElemData['Osseous Fraction'][(elem_stimulus < .01) | (elem_stimulus > 1)] - degradation_rate / grid.parameters['Volume'] * grid.TimeStep, 0)
    grid.ElemData['Cartilaginous Fraction'][(elem_stimulus < 1) | (elem_stimulus > 3)] = np.maximum(grid.ElemData['Cartilaginous Fraction'][(elem_stimulus < 1) | (elem_stimulus > 3)] - degradation_rate / grid.parameters['Volume'] * grid.TimeStep, 0)
    grid.ElemData['Fibrous Fraction'][(elem_stimulus < 3)] = np.maximum(grid.ElemData['Fibrous Fraction'][(elem_stimulus < 3)] - degradation_rate / grid.parameters['Volume'] * grid.TimeStep, 0)
    
    grid.ElemData['Volume Fraction'] = grid.ElemData['ECM Fraction'] + \
                grid.ElemData['Fibrous Fraction'] + \
                grid.ElemData['Cartilaginous Fraction'] + \
                grid.ElemData['Osseous Fraction'] + \
                grid.ElemData['Scaffold Fraction']    
    

# Model Actions
def update_scaffold_curvature(model):
    """
    Calculate curvatures on a scaffold surface.

    See also: :ref:`Tissue Growth`

    Parameters
    ----------
    model : myabm.ortho.OrthoModel
        Initialized agent-based model

    """

    # This will only calculate the scaffold curvature if scaffold curvature hasn't been calculated yet, assuming the scaffold surface doesn't change

    if np.max(model.agent_grid.ElemData['Scaffold Fraction']) == 0:
        # No scaffold
        return
    if 'Scaffold Mean Curvature' in model.agent_grid.ElemData.keys() and not np.all(np.isnan(model.agent_grid.ElemData['Scaffold Mean Curvature'])):
        # Curvature already computed on scaffold surface
        return
    
    h = model.agent_grid.parameters['h']
    
    Grid, constraints, boundaries, inv = _grid_padding(model)
    Scaf = Grid.Threshold('Scaffold Fraction', 0, '>')
    ScaffoldNodes = Scaf.SurfNodes
    
    if model.model_parameters['Smooth Scaffold']:
        ScafSurf = improvement.LocalLaplacianSmoothing(Scaf.Surface, options=dict(limit=np.sqrt(3)/2*h,  FixedNodes=boundaries, constraint=constraints))
        ScafSurf.verbose=False
        Scaf.NodeCoords = ScafSurf.NodeCoords
        # model.agent_grid.NodeVectorData['Scaffold Coordinates'] = ScafSurf.NodeCoords
    else:
        ScafSurf = Scaf.Surface
    model.agent_grid.NodeVectorData['Scaffold Coordinates'] = ScafSurf.NodeCoords

    # Compute Curvature
    k1, k2, k1v, k2v = curvature.CubicFit(ScafSurf.NodeCoords, ScafSurf.SurfConn, utils.getNodeNeighborhood(*converter.surf2tris(*ScafSurf), 1), ScafSurf.NodeNormals, return_directions=True)

    if model.model_parameters['Periodic'] or model.model_parameters['Symmetric']:
        # remove padding
        k1 = k1[inv[:model.NNode]]
        k2 = k2[inv[:model.NNode]]
        k1v = k1v[inv[:model.NNode],:]
        k2v = k2v[inv[:model.NNode],:]
        # Scaf_copy.NodeCoords = Scaf.NodeCoords[inv[:model.NNode]]
        # Scaf = Scaf_copy
        # ScafSurf = Scaf.Surface
    
    model.agent_grid.NodeData['Scaffold Max Principal Curvature'] = k1
    model.agent_grid.NodeData['Scaffold Min Principal Curvature'] = k2
    model.agent_grid.NodeVectorData['Scaffold Max Principal Direction'] = k1v
    model.agent_grid.NodeVectorData['Scaffold Min Principal Direction'] = k2v
    model.agent_grid.NodeData['Scaffold Mean Curvature'] = curvature.MeanCurvature(k1, k2)
    model.agent_grid.NodeData['Scaffold Gaussian Curvature'] = curvature.GaussianCurvature(k1, k2)


def _grid_padding(model):

    Grid = model.mesh.copy()
    Grid.ElemData['Volume Fraction'] = model.agent_grid.ElemData['Volume Fraction']
    Grid.ElemData['Scaffold Fraction'] = model.agent_grid.ElemData['Scaffold Fraction']
    if 'Tissue Orientation' in model.agent_grid.ElemVectorData:
        Grid.ElemData['Tissue Orientation'] = model.agent_grid.ElemVectorData['Tissue Orientation']
        Grid.ElemData['Tissue Orientation Orthogonal'] = model.agent_grid.ElemVectorData['Tissue Orientation Orthogonal']
    h = model.agent_grid.parameters['h']
    maxxn = np.max(Grid.NodeCoords[:,0])
    minxn = np.min(Grid.NodeCoords[:,0])
    maxyn = np.max(Grid.NodeCoords[:,1])
    minyn = np.min(Grid.NodeCoords[:,1])
    maxzn = np.max(Grid.NodeCoords[:,2])
    minzn = np.min(Grid.NodeCoords[:,2])
    
    if model.model_parameters['Periodic'] and model.model_parameters['Symmetric']:
        raise ValueError('A model that is both Periodic and Symmetric is not supported.\n Change model.model_parameters["Periodic"] and/or model.model_parameters["Symmetric"] to False.')
    
    elif model.model_parameters['Periodic']:
        maxxe = np.max(Grid.Centroids[:,0])
        minxe = np.min(Grid.Centroids[:,0])
        maxye = np.max(Grid.Centroids[:,1])
        minye = np.min(Grid.Centroids[:,1])
        maxze = np.max(Grid.Centroids[:,2])
        minze = np.min(Grid.Centroids[:,2])

        maxx_pad = Grid.Threshold(Grid.Centroids[:,0], minxe, '==')
        maxx_pad.NodeCoords[:,0] += (maxxe-minxe + h)
        minx_pad = Grid.Threshold(Grid.Centroids[:,0], maxxe, '==') 
        minx_pad.NodeCoords[:,0] -= (maxxe-minxe + h)
        
        maxy_pad = Grid.Threshold(Grid.Centroids[:,1], minye, '==')
        maxy_pad.NodeCoords[:,1] += (maxye-minye + h)
        miny_pad = Grid.Threshold(Grid.Centroids[:,1], maxye, '==') 
        miny_pad.NodeCoords[:,1] -= (maxye-minye + h)
        
        maxz_pad = Grid.Threshold(Grid.Centroids[:,2], minze, '==')
        maxz_pad.NodeCoords[:,2] += (maxze-minze + h)
        minz_pad = Grid.Threshold(Grid.Centroids[:,2], maxze, '==') 
        minz_pad.NodeCoords[:,2] -= (maxze-minze + h)
                            
        pad = maxx_pad
        pad.merge([minx_pad, maxy_pad, miny_pad, maxz_pad, minz_pad])
        # Grid_copy = Grid.copy()
        Grid.ElemData['original'] = np.ones(Grid.NElem)
        pad.ElemData['original'] = np.zeros(pad.NElem)
        Grid.merge(pad, cleanup=False)
        Grid.NodeCoords, Grid.NodeConn, idx, inv = utils.DeleteDuplicateNodes(Grid.NodeCoords, Grid.NodeConn, return_inv=True, return_idx=True, tol=1e-9)
        Grid._Surface = None
        Grid.reset('SurfConn')
        Grid._SurfNodes = None

        xnodes = np.where((np.isclose(Grid.NodeCoords[:,0], minxn)) | (np.isclose(Grid.NodeCoords[:,0], maxxn)))[0]
        ynodes = np.where((np.isclose(Grid.NodeCoords[:,1], minyn)) | (np.isclose(Grid.NodeCoords[:,1], maxyn)))[0]
        znodes = np.where((np.isclose(Grid.NodeCoords[:,2], minzn)) | (np.isclose(Grid.NodeCoords[:,2], maxzn)))[0]
        
        constraints = np.vstack([
                np.column_stack((xnodes, np.repeat(0, len(xnodes)), np.zeros(len(xnodes)))),
                np.column_stack((ynodes, np.repeat(1, len(ynodes)), np.zeros(len(ynodes)))),
                np.column_stack((znodes, np.repeat(2, len(znodes)), np.zeros(len(znodes)))),
                ])

        boundaries = set(np.where(
                    (Grid.NodeCoords[:,0] == Grid.NodeCoords[:,0].min()) |
                    (Grid.NodeCoords[:,0] == Grid.NodeCoords[:,0].max()) |
                    (Grid.NodeCoords[:,1] == Grid.NodeCoords[:,1].min()) |
                    (Grid.NodeCoords[:,1] == Grid.NodeCoords[:,1].max()) |
                    (Grid.NodeCoords[:,2] == Grid.NodeCoords[:,2].min()) |
                    (Grid.NodeCoords[:,2] == Grid.NodeCoords[:,2].max())
                )[0].tolist())

    elif model.model_parameters['Symmetric']:
        
        pad_width = 2
        pad_dist = (pad_width)*h
        
        if model.model_parameters['Symmetric'] is True or model.model_parameters['Symmetric'] == 1:
          
            # maxxe = np.max(Grid.Centroids[:,0])
            # maxye = np.max(Grid.Centroids[:,1])
            # maxze = np.max(Grid.Centroids[:,2])
            xn = maxxn
            yn = maxyn
            zn = maxzn

            x_pad = Grid.Threshold(Grid.Centroids[:,0], Grid.Centroids[:,0].max()-pad_dist, '>', cleanup=True).Mirror(x=Grid.NodeCoords[:,0].max(), InPlace=True)
            y_pad = Grid.Threshold(Grid.Centroids[:,1], Grid.Centroids[:,1].max()-pad_dist, '>', cleanup=True).Mirror(y=Grid.NodeCoords[:,1].max(), InPlace=True)
            z_pad = Grid.Threshold(Grid.Centroids[:,2], Grid.Centroids[:,2].max()-pad_dist, '>', cleanup=True).Mirror(z=Grid.NodeCoords[:,2].max(), InPlace=True)
        
        elif model.model_parameters['Symmetric'] == -1:
            xn = minxn
            yn = minyn
            zn = minzn

            x_pad = Grid.Threshold(Grid.Centroids[:,0], Grid.Centroids[:,0].min()+pad_dist, '<', cleanup=True).Mirror(x=Grid.NodeCoords[:,0].min(), InPlace=True)
            y_pad = Grid.Threshold(Grid.Centroids[:,1], Grid.Centroids[:,1].min()+pad_dist, '<', cleanup=True).Mirror(y=Grid.NodeCoords[:,1].min(), InPlace=True)
            z_pad = Grid.Threshold(Grid.Centroids[:,2], Grid.Centroids[:,2].min()+pad_dist, '<', cleanup=True).Mirror(z=Grid.NodeCoords[:,2].min(), InPlace=True)
        
        
        pad = x_pad
        pad.merge([y_pad, z_pad])
        # Grid_copy = Grid.copy()
        Grid.ElemData['original'] = np.ones(Grid.NElem)
        pad.ElemData['original'] = np.zeros(pad.NElem)
        Grid.merge(pad.copy(), cleanup=False)
        Grid.NodeCoords, Grid.NodeConn, idx, inv = utils.DeleteDuplicateNodes(Grid.NodeCoords, Grid.NodeConn, return_inv=True, return_idx=True, tol=1e-9)
        Grid._Surface = None
        Grid.reset('SurfConn')
        Grid._SurfNodes = None

        
        xnodes = np.where((np.isclose(Grid.NodeCoords[:,0], xn)))[0]
        ynodes = np.where((np.isclose(Grid.NodeCoords[:,1], yn)))[0]
        znodes = np.where((np.isclose(Grid.NodeCoords[:,2], zn)))[0]
        
        constraints = np.vstack([
                np.column_stack((xnodes, np.repeat(0, len(xnodes)), np.zeros(len(xnodes)))),
                np.column_stack((ynodes, np.repeat(1, len(ynodes)), np.zeros(len(ynodes)))),
                np.column_stack((znodes, np.repeat(2, len(znodes)), np.zeros(len(znodes)))),
                ])
        if model.model_parameters['Symmetric'] == 1:
            boundaries = set(np.where(
                        (Grid.NodeCoords[:,0] == Grid.NodeCoords[:,0].max()) |
                        (Grid.NodeCoords[:,1] == Grid.NodeCoords[:,1].max()) |
                        (Grid.NodeCoords[:,2] == Grid.NodeCoords[:,2].max())
                    )[0].tolist())
        elif model.model_parameters['Symmetric'] == -1:
            boundaries = set(np.where(
                        (Grid.NodeCoords[:,0] == Grid.NodeCoords[:,0].min()) |
                        (Grid.NodeCoords[:,1] == Grid.NodeCoords[:,1].min()) |
                        (Grid.NodeCoords[:,2] == Grid.NodeCoords[:,2].min())
                    )[0].tolist())
    
    else:
        constraints = np.empty((0,3))
        boundaries = set()
        inv = np.arange(Grid.NNode)
        
    return Grid, constraints, boundaries, inv

def update_curvature(model):
    """
    Calculate curvatures on the tissue boundary and within the tissue.

    See also: :ref:`Tissue Growth`

    Parameters
    ----------
    model : myabm.ortho.OrthoModel
        Initialized agent-based model

    """
    h = model.agent_grid.parameters['h']
    Grid, constraints, boundaries, inv = _grid_padding(model)
    # Surface Curvature
    Tissue = Grid.Threshold('Volume Fraction', model.agent_grid.parameters['Tissue Threshold'])
    
    Surf = Tissue.Surface

    Surf = improvement.LocalLaplacianSmoothing(Surf, options=dict(FixedNodes=boundaries, limit=np.sqrt(3)/2*h, constraint=constraints))
    Surf.verbose = False

    TissueNodes =  np.zeros(Tissue.NNode,dtype=int)
    TissueNodes[np.intersect1d(np.unique(Tissue.NodeConn[Tissue.ElemData['Scaffold Fraction'] == 0]), Tissue.SurfNodes)] = 1
    NonTissueNodes = TissueNodes == 0
    TissueNodesPlus = np.isin(np.arange(Surf.NNode), np.unique(Surf.NodeConn[np.any(TissueNodes[Surf.NodeConn],axis=1)]))
    
    if 'Scaffold Coordinates' not in model.agent_grid.NodeVectorData and np.max(model.agent_grid.ElemData['Scaffold Fraction']) > 0:
        update_scaffold_curvature(model)

    if 'Scaffold Coordinates' in model.agent_grid.NodeVectorData:
        Surf.NodeCoords[NonTissueNodes] = model.agent_grid.NodeVectorData['Scaffold Coordinates'][NonTissueNodes]

    SurfNodeNeighbors =  mymesh.utils.getNodeNeighborhood(*mymesh.converter.surf2tris(*Surf), 1)
    k1, k2, k1v, k2v = mymesh.curvature.CubicFit(Surf.NodeCoords, Surf.NodeConn, SurfNodeNeighbors, Surf.NodeNormals, return_directions=True)
    
    if model.model_parameters['Periodic'] or model.model_parameters['Symmetric']:
        # remove padding
        k1 = k1[inv[:model.NNode]]
        k2 = k2[inv[:model.NNode]]
        k1v = k1v[inv[:model.NNode],:]
        k2v = k2v[inv[:model.NNode],:]
        
        
    if np.max(model.agent_grid.ElemData['Scaffold Fraction']) > 0:

        # Pull scaffold data
        model.agent_grid.NodeData['Max Principal Curvature'] = model.agent_grid.NodeData['Scaffold Max Principal Curvature']
        model.agent_grid.NodeVectorData['Max Principal Direction'] = model.agent_grid.NodeVectorData['Scaffold Max Principal Direction']
        model.agent_grid.NodeData['Min Principal Curvature'] = model.agent_grid.NodeData['Scaffold Min Principal Curvature']
        model.agent_grid.NodeVectorData['Min Principal Direction'] = model.agent_grid.NodeVectorData['Scaffold Min Principal Direction']
        model.agent_grid.NodeData['Mean Curvature'] = model.agent_grid.NodeData['Scaffold Mean Curvature']
        model.agent_grid.NodeData['Gaussian Curvature'] = model.agent_grid.NodeData['Scaffold Gaussian Curvature']
        
    else:
        for curve in ['Max Principal Curvature', 'Min Principal Curvature', 'Mean Curvature', 'Gaussian Curvature']:
            if curve not in model.agent_grid.NodeData:
                model.agent_grid.NodeData[curve] = np.repeat(sys.float_info.max, model.NNode)
        for curve in ['Max Principal Direction', 'Min Principal Direction']:
            if curve not in model.agent_grid.NodeData:
                model.agent_grid.NodeVectorData[curve] = np.zeros((model.NNode,3), dtype=np.float64)
    if 'Smoothed Surface Coordinates' not in model.agent_grid.NodeVectorData:
        model.agent_grid.NodeVectorData['Smoothed Surface Coordinates'] = Surf.NodeCoords[inv[:model.NNode]]
    else:
        model.agent_grid.NodeVectorData['Smoothed Surface Coordinates'] = Surf.NodeCoords[inv[:model.NNode]]# Surf.NodeCoords[TissueNodesPlus[inv[:model.NNode]]]
    
    if model.model_parameters['Periodic'] or model.model_parameters['Symmetric']:
        TissueNodesPlus = TissueNodesPlus[inv[:model.NNode]]
    model.agent_grid.NodeData['Max Principal Curvature'][TissueNodesPlus] = k1[TissueNodesPlus]
    model.agent_grid.NodeData['Min Principal Curvature'][TissueNodesPlus] = k2[TissueNodesPlus]
    model.agent_grid.NodeVectorData['Max Principal Direction'][TissueNodesPlus] = k1v[TissueNodesPlus]
    model.agent_grid.NodeVectorData['Min Principal Direction'][TissueNodesPlus] = k2v[TissueNodesPlus]
    model.agent_grid.NodeData['Mean Curvature'][TissueNodesPlus] = mymesh.curvature.MeanCurvature(k1[TissueNodesPlus], k2[TissueNodesPlus])
    model.agent_grid.NodeData['Gaussian Curvature'][TissueNodesPlus] = mymesh.curvature.GaussianCurvature(k1[TissueNodesPlus], k2[TissueNodesPlus])

    # if normalize:
    #     O1norm = np.linalg.norm(Grid.ElemData['Tissue Orientation'],axis=1)[:,None]
    #     O1 = np.divide(Grid.ElemData['Tissue Orientation'], O1norm, where=O1norm != 0, out=np.zeros_like(Grid.ElemData['Tissue Orientation']))
    #     O2norm = np.linalg.norm(Grid.ElemData['Tissue Orientation Orthogonal'],axis=1)[:,None]
    #     O2 = np.divide(Grid.ElemData['Tissue Orientation Orthogonal'], O2norm, where=O2norm != 0, out=np.zeros_like(Grid.ElemData['Tissue Orientation Orthogonal']))
    # else:
    O1 = model.agent_grid.ElemVectorData['Tissue Orientation']
    O2 = model.agent_grid.ElemVectorData['Tissue Orientation Orthogonal']
        
    normals = np.cross(O2, O1)
    # NOTE: not sure if TissueThreshold should just be 0 here
    normals[model.agent_grid.ElemData['Volume Fraction'] < model.agent_grid.parameters['Tissue Threshold']] = np.nan     # This leads to nan values in mean curvature for in areas of no tissue
    model.agent_grid.ElemVectorData['n'] = normals
    # Grid.write('temp.vtu')
    
    nx = np.pad(converter.voxel2im(*model.mesh, normals[:,0]),1)
    ny = np.pad(converter.voxel2im(*model.mesh, normals[:,1]),1)
    nz = np.pad(converter.voxel2im(*model.mesh, normals[:,2]),1)
    
    
    dnxdx = ((nx[:-1,:-1,1:] - nx[:-1,:-1,:-1]) +  # f[i+1] - f[i]
            (nx[:-1,1:,1:] - nx[:-1,1:,:-1]) + 
            (nx[1:,:-1,1:] - nx[1:,:-1,:-1]) + 
            (nx[1:,1:,1:] - nx[1:,1:,:-1]))
    dnxdx[~np.isnan(dnxdx)] /= 4*h
    
    dnydy = ((ny[:-1,1:,:-1] - ny[:-1,:-1,:-1]) + 
            (ny[:-1,1:,:1] - ny[:-1,:-1,:1]) + 
            (ny[1:,1:,:-1] - ny[1:,:-1,:-1]) + 
            (ny[1:,1:,1:] - ny[1:,:-1,1:]))
    dnydy[~np.isnan(dnydy)] /= 4*h
    
    dnzdz = ((nz[1:,:-1,:-1] - nz[:-1,:-1,:-1]) + 
            (nz[1:,1:,:-1] - nz[:-1,1:,:-1]) + 
            (nz[1:,:-1,1:] - nz[:-1,:-1,1:]) + 
            (nz[1:,1:,1:] - nz[:-1,1:,1:]))
    dnzdz[~np.isnan(dnzdz)] /= 4*h
    
    MeanCurvature = ((dnxdx + dnydy + dnzdz)).flatten(order='F')
    MeanCurvature[~np.isnan(MeanCurvature)] /= 2

    model.agent_grid.NodeData['Mean Curvature'][np.isnan(model.agent_grid.NodeData['Mean Curvature'])] = MeanCurvature[np.isnan(model.agent_grid.NodeData['Mean Curvature'])]
    
    # Fill in NaNs by projecting
    if np.any(np.isnan(model.agent_grid.NodeData['Mean Curvature'])):
        model.agent_grid.NodeData['Mean Curvature'][np.isnan(model.agent_grid.NodeData['Mean Curvature'])]  = sys.float_info.max

        NodeNeighbors = utils.PadRagged(model.mesh.NodeNeighbors)
        NodeNeighbors = np.vstack([NodeNeighbors, np.zeros(len(NodeNeighbors[0]),dtype=int)])
        
        MeanCurvature = np.append(model.agent_grid.NodeData['Mean Curvature'],np.inf)
        mask = MeanCurvature == sys.float_info.max
        i = 0
        while np.any(mask):
            i += 1
            MeanCurvature[mask] = sys.float_info.max
            minfilt = np.take_along_axis(NodeNeighbors, np.argmin(MeanCurvature[NodeNeighbors],axis=1)[:,None], 1).flatten()
            MeanCurvature[mask] = MeanCurvature[minfilt][mask]
            model.agent_grid.NodeData['Max Principal Curvature'][mask[:-1]] = model.agent_grid.NodeData['Max Principal Curvature'][minfilt[:-1]][mask[:-1]]
            model.agent_grid.NodeVectorData['Max Principal Direction'][mask[:-1]] = model.agent_grid.NodeVectorData['Max Principal Direction'][minfilt[:-1]][mask[:-1]]
            model.agent_grid.NodeData['Min Principal Curvature'][mask[:-1]] = model.agent_grid.NodeData['Min Principal Curvature'][minfilt[:-1]][mask[:-1]]
            model.agent_grid.NodeVectorData['Min Principal Direction'][mask[:-1]] = model.agent_grid.NodeVectorData['Min Principal Direction'][minfilt[:-1]][mask[:-1]]
            mask = MeanCurvature == sys.float_info.max
            if i == 100:
                warnings.warn('Iteration limit reached for mean curvature calculation.\nThis could indicate unexpected values or a significantly larger than expected grid size.')
                break
        model.agent_grid.NodeData['Mean Curvature'] = MeanCurvature[:-1]
        for curve in ['Max Principal Curvature', 'Min Principal Curvature', 'Mean Curvature', 'Gaussian Curvature']:
            model.agent_grid.NodeData[curve][model.agent_grid.NodeData[curve] == sys.float_info.max] = np.nan
# def run_compression(model):

#     matprop = (self.agent_grid.ElemData['modulus'],
#                 self.agent_grid.ElemData['poisson'],
#                 self.agent_grid.ElemData['permeability'],
#                 self.agent_grid.ElemData['solid bulk modulus'],
#                 self.agent_grid.ElemData['fluid bulk modulus'],
#                 self.agent_grid.ElemData['fluid specific weight'],
#                 self.agent_grid.ElemData['porosity'])
#     disp = model.model_parameters['displacement']
#     try:
#         apdl = pyAnsysSolvers.Launch(ansys_path, nproc=nproc, smp=True, port=port)
#         M = pyAnsysSolvers.PoroelasticUniaxial(M, *matprop, timestep, disp, axis=2, boundary_eps=0, SurfacePressure=0, nsubsteps=1, constrained=False, mode='disp', mapdl=apdl)
#     except:
#         apdl = pyAnsysSolvers.Launch(ansys_path, nproc=nproc, smp=True, port=port)
#         M = pyAnsysSolvers.PoroelasticUniaxial(M, *matprop, timestep, disp, axis=2, boundary_eps=0, SurfacePressure=0, nsubsteps=1, constrained=False, mode='disp', mapdl=apdl)

def update_properties(model):
    WaterProps = mechanobiology.prendergast_mat(np.repeat(0, model.agent_grid.NElem))
    NeoProps = mechanobiology.prendergast_mat(np.repeat(1, model.agent_grid.NElem))
    FibProps = mechanobiology.prendergast_mat(np.repeat(2, model.agent_grid.NElem))
    CartProps = mechanobiology.prendergast_mat(np.repeat(3, model.agent_grid.NElem))
    MarrowProps = mechanobiology.prendergast_mat(np.repeat(4, model.agent_grid.NElem))
    IboneProps = mechanobiology.prendergast_mat(np.repeat(5, model.agent_grid.NElem))     # Immature bone
    LboneProps = mechanobiology.prendergast_mat(np.repeat(7, model.agent_grid.NElem))     # Lamellar bone
    ScaffProps = mechanobiology.prendergast_mat(np.repeat(8, model.agent_grid.NElem))
    
    
    def modulus_mineral(mineral_density): 
        modulus = np.zeros_like(mineral_density)
        modulus[mineral_density < 0.4] = 32.5*mineral_density[mineral_density < 0.4]**3 +10.5*mineral_density[mineral_density < 0.4]**2 + mineral_density[mineral_density < 0.4] + 0.01
        # modulus[mineral_density < 0.4] = 63.4375*mineral_density[mineral_density < 0.4]**3 - 8.0625*mineral_density[mineral_density < 0.4]**2 + mineral_density[mineral_density < 0.4] + 1
        modulus[mineral_density >= 0.4] = 25 * mineral_density[mineral_density >= 0.4] - 5.83
        
        # Above are in GPa
        modulus *= 1000 # now in MPa
        
        return modulus
    
    # Calculate properties of mineralized tissue
    agent_grid = model.agent_grid
    # mineral_density = np.divide(agent_grid.ElemData['Mineral Density'], agent_grid.ElemData['Osseous Fraction'], where=agent_grid.ElemData['Osseous Fraction']>0, out=np.zeros(agent_grid.NElem))
    mineral_density = np.divide(agent_grid.ElemData['Mineral Density'], 1, where=agent_grid.ElemData['Osseous Fraction']>0, out=np.zeros(agent_grid.NElem))
    max_modulus = modulus_mineral(np.array([0.8])) # max mineralization .8 mg/mm^3
    min_modulus = modulus_mineral(np.array([0.0]))
    modulus = modulus_mineral(mineral_density)
    
    ratio = (modulus - min_modulus)/(max_modulus - min_modulus)
    
    MineralizedProps = []
    for i in range(len(LboneProps)):
        prop = ratio * (LboneProps[i] - IboneProps[i]) + IboneProps[i]
        MineralizedProps.append(prop)
    
    
    # Set volume fraction weighted properties
    E = (NeoProps[0] * agent_grid.ElemData['ECM Fraction'] + \
        FibProps[0] * agent_grid.ElemData['Fibrous Fraction'] + \
        CartProps[0] * agent_grid.ElemData['Cartilaginous Fraction'] + \
        # dividing mineral density by osseous fraction to get modulus of mineral/osteoid volume, then multiplying by volume fraction
        modulus * agent_grid.ElemData['Osseous Fraction'] + \
        ScaffProps[0] * agent_grid.ElemData['Scaffold Fraction'] + \
        WaterProps[0] * (1-agent_grid.ElemData['Volume Fraction']))
    
    nu = (NeoProps[1] * agent_grid.ElemData['ECM Fraction'] + \
        FibProps[1] * agent_grid.ElemData['Fibrous Fraction'] + \
        CartProps[1] * agent_grid.ElemData['Cartilaginous Fraction'] + \
        # IboneProps[1] * agent_grid.ElemData['Osseous Fraction'] + \
        MineralizedProps[1] * agent_grid.ElemData['Osseous Fraction'] + \
        ScaffProps[1] * agent_grid.ElemData['Scaffold Fraction'] + \
        WaterProps[1] * (1-agent_grid.ElemData['Volume Fraction']))
    
    K = (NeoProps[2] * agent_grid.ElemData['ECM Fraction'] + \
        FibProps[2] * agent_grid.ElemData['Fibrous Fraction'] + \
        CartProps[2] * agent_grid.ElemData['Cartilaginous Fraction'] + \
        # IboneProps[2] * agent_grid.ElemData['Osseous Fraction'] + \
        MineralizedProps[2] * agent_grid.ElemData['Osseous Fraction'] + \
        ScaffProps[2] * agent_grid.ElemData['Scaffold Fraction'] + \
        WaterProps[2] * (1-agent_grid.ElemData['Volume Fraction']))
    
    FluidBulk = (NeoProps[3] * agent_grid.ElemData['ECM Fraction'] + \
        FibProps[3] * agent_grid.ElemData['Fibrous Fraction'] + \
        CartProps[3] * agent_grid.ElemData['Cartilaginous Fraction'] + \
        # IboneProps[3] * agent_grid.ElemData['Osseous Fraction'] + \
        MineralizedProps[3] * agent_grid.ElemData['Osseous Fraction'] + \
        ScaffProps[3] * agent_grid.ElemData['Scaffold Fraction'] + \
        WaterProps[3] * (1-agent_grid.ElemData['Volume Fraction']))
    
    FluidSpecWeight = (NeoProps[4] * agent_grid.ElemData['ECM Fraction'] + \
        FibProps[4] * agent_grid.ElemData['Fibrous Fraction'] + \
        CartProps[4] * agent_grid.ElemData['Cartilaginous Fraction'] + \
        # IboneProps[4] * agent_grid.ElemData['Osseous Fraction'] + \
        MineralizedProps[4] * agent_grid.ElemData['Osseous Fraction'] + \
        ScaffProps[4] * agent_grid.ElemData['Scaffold Fraction'] + \
        WaterProps[4] * (1-agent_grid.ElemData['Volume Fraction']))
    
    Porosity = (NeoProps[5] * agent_grid.ElemData['ECM Fraction'] + \
        FibProps[5] * agent_grid.ElemData['Fibrous Fraction'] + \
        CartProps[5] * agent_grid.ElemData['Cartilaginous Fraction'] + \
        # IboneProps[5] * agent_grid.ElemData['Osseous Fraction'] + \
        MineralizedProps[5] * agent_grid.ElemData['Osseous Fraction'] + \
        ScaffProps[5] * agent_grid.ElemData['Scaffold Fraction'] + \
        WaterProps[5] * (1-agent_grid.ElemData['Volume Fraction']))
    
    SolidBulk = (NeoProps[6] * agent_grid.ElemData['ECM Fraction'] + \
        FibProps[6] * agent_grid.ElemData['Fibrous Fraction'] + \
        CartProps[6] * agent_grid.ElemData['Cartilaginous Fraction'] + \
        # IboneProps[6] * agent_grid.ElemData['Osseous Fraction'] + \
        MineralizedProps[6] * agent_grid.ElemData['Osseous Fraction'] + \
        ScaffProps[6] * agent_grid.ElemData['Scaffold Fraction'] + \
        WaterProps[6] * (1-agent_grid.ElemData['Volume Fraction']))


    model.agent_grid.ElemData['modulus'] = E
    model.agent_grid.ElemData['poisson'] = nu
    model.agent_grid.ElemData['permeability'] = K
    model.agent_grid.ElemData['solid bulk modulus'] = SolidBulk
    model.agent_grid.ElemData['fluid bulk modulus'] = FluidBulk
    model.agent_grid.ElemData['fluid specific weight'] = FluidSpecWeight
    model.agent_grid.ElemData['porosity'] = Porosity
