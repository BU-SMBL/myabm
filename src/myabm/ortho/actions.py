#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Mon Apr 28 12:31:40 2025
# @author: toj
"""

Ortho Agent Actions
===================
.. autosummary::
    :toctree: submodules/

    migrate
    migrate_curvotaxis
    proliferate
    apoptose
    produce
    produce_oriented

Ortho Model Actions
===================

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

    Returns
    -------
    None.

    """
    
    pass

@numba.njit
def migrate(agent, grid):
    """
    Random walk migration.

    Parameters
    ----------
    agent : myabm.Agent
        Cell agent to perform the action.
    grid : myabm.AgentGrid
        Grid environment that the agent is in.

    Requires
    --------
    grid.parameters['Mineral Walk Threshold'] : float
        Mineral content threshold above which cells cannot migrate through
        In units of mg/mm^3, ~0.8 corresponds to bone mineral content.  
    grid.parameters['Tissue Threshold'] : float
        Volume fraction of extracellular matrix above which cells can migrate
        across. Below this threshold, the voxel is considered empty and 
        cannot be crossed, in range [0, 1].
    grid.ElemData['Mineral Density'] : np.ndarray(dtype=float)
        Array of mineral content for each voxel in the grid, in units of mg/mm^3
    grid.ElemData['Volume Fraction'] : np.ndarray(dtype=float)
        Array of volume fractions for each voxel in the grid, in range [0, 1]
    grid.NodeData['CellsAllowed'] : np.ndarray(dtype=float)
        Binary array indicating whether a cell is able to migrate to each node

    Returns
    -------
    None.

    """

    # for required in ['MigrRate', 'Tissue Threshold']:
    #     assert required in agent.parameters, 'agent.parameters must contain "'+required+'"'
    # for required in ['Mineral Walk Threshold', 'Tissue Threshold']:
    #     assert required in grid.parameters, 'agent_grid.parameters must contain "'+required+'"'
    # for required in ['Mineral Density', 'Volume Fraction']:
    #     assert required in grid.ElemData, 'agent_grid.ElemData must contain "'+required+'"'
    # for required in ['CellsAllowed']:
    #     assert required in grid.NodeData, 'agent_grid.NodeData must contain "'+required+'"'

    TimeStep = grid.TimeStep
    StepSize = grid.h
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
            open_neighbors = np.array([n for n in np.unique(grid.Edges[np.array(list(edge_set))]) if (n not in grid.NodeAgents) and (grid.NodeData['CellsAllowed'][n] == 1)])
            # TODO: modify this to allow for swapping/
            if len(open_neighbors) > 0:
                step = np.random.choice(open_neighbors) # Choose new node
                
                # Update data structure
                del grid.NodeAgents[agent.node]          # Remove cell from old node
                agent.node = step                        # Update cell's node id
                grid.NodeAgents[step] = agent            # Move cell to new node

@numba.njit    
def migrate_curvotaxis(agent, grid):
    """
    Curvotactic walk migration.
    Random walk weighted by surface curvature, based on Pieuchot et al. 2018

    Parameters
    ----------
    agent : myabm.Agent
        Cell agent to perform the action.
    grid : myabm.AgentGrid
        Grid environment that the agent is in.

    Requires
    --------
    grid.parameters['Mineral Walk Threshold'] : float
        Mineral content threshold above which cells cannot migrate through
        In units of mg/mm^3, ~0.8 corresponds to bone mineral content.  
    grid.parameters['Tissue Threshold'] : float
        Volume fraction of extracellular matrix above which cells can migrate
        across. Below this threshold, the voxel is considered empty and 
        cannot be crossed, in range [0, 1].
    agent.parameters['MigrationWeight0'] : float

        migration_rate = (w0 * w1**H + w2) / 1000 * 24 (mm/day)
    agent.parameters['MigrationWeight1']

        migration_rate = (w0 * w1**H + w2) / 1000 * 24 (mm/day)
    agent.parameters['MigrationWeight2']

        migration_rate = (w0 * w1**H + w2) / 1000 * 24 (mm/day)
    agent.parameters['MigrationWeight3']

    agent.parameters['MigrationWeight4']

    grid.ElemData['Mineral Density'] : np.ndarray(dtype=float)
        Array of mineral content for each voxel in the grid, in units of mg/mm^3
    grid.ElemData['Volume Fraction'] : np.ndarray(dtype=float)
        Array of volume fractions for each voxel in the grid, in range [0, 1]

    grid.NodeData['Mean Curvature']

    Returns
    -------
    None.

    """

    TimeStep = grid.TimeStep
    StepSize = grid.h
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
    k = (a * b**H + c) / 1000 * 24 # migration rate (mm/day)
    Lambda = k/StepSize # migration event rate (/day)
    MigrationProbability = 1 - np.exp(-Lambda*TimeStep)
    if np.random.rand() < MigrationProbability:             
        if len(edge_set) > 0:
            # neighbors that are reachable and not occupied
            open_neighbors = np.array([n for n in np.unique(grid.Edges[np.array(list(edge_set))]) if (n not in grid.NodeAgents) and (grid.NodeData['CellsAllowed'][n] == 1)])
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
                oldnode = agent.node
                agent.node = step                        # Update cell's node id
                grid.NodeAgents[step] = agent            # Move cell to new node
                del grid.NodeAgents[oldnode]          # Remove cell from old node
      
@numba.njit
def proliferate(agent, grid):
    """
    Cell proliferation. Random chance of creation of a new cell at an
    available adjacent site.

    Parameters
    ----------
    agent : myabm.Agent
        Cell agent to perform the action.
    grid : myabm.AgentGrid
        Grid environment that the agent is in.

    Requires
    --------
    agent.parameters['ProlifRate'] : float
        Proliferation rate, in proliferation events per day. 
    grid.parameters['Mineral Walk Threshold'] : float
        Mineral content threshold above which cells cannot migrate through
        In units of mg/mm^3, ~0.8 corresponds to bone mineral content.  
    grid.parameters['Tissue Threshold'] : float
        Volume fraction of extracellular matrix above which cells can migrate
        across. Below this threshold, the voxel is considered empty and 
        cannot be crossed, in range [0, 1].
    grid.ElemData['Mineral Density'] : np.ndarray(dtype=float)
        Array of mineral content for each voxel in the grid, in units of mg/mm^3
    grid.ElemData['Volume Fraction'] : np.ndarray(dtype=float)
        Array of volume fractions for each voxel in the grid, in range [0, 1]

    Returns
    -------
    None.

    """
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
        open_neighbors = np.array([n for n in np.unique(grid.Edges[np.array(list(edge_set))]) if (n not in grid.NodeAgents) and (grid.NodeData['CellsAllowed'][n] == 1)])

        open_neighbors = open_neighbors[open_neighbors != agent.node]
        if len(open_neighbors) == 0:
            return
        
        if np.random.rand() < ProlifProb:
            step = np.random.choice(open_neighbors)
            grid.add_agent(step, agent.state, agent.age, agent.parameters)

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
        open_neighbors = np.array([n for n in np.unique(grid.Edges[np.array(list(edge_set))]) if (n not in grid.NodeAgents) and (grid.NodeData['CellsAllowed'][n] == 1)])

        open_neighbors = open_neighbors[open_neighbors != agent.node]
        if len(open_neighbors) == 0:
            return
        
        if np.random.rand() < ProlifProb:
            step = np.random.choice(open_neighbors)
            grid.add_agent(step, agent.state, agent.age, agent.parameters)
                
@numba.njit
def proliferate_contactinhibited(agent, grid):
    
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
        elif agent.state == 'osteoblast' and np.all(grid.ElemData['Mineral Density'][elems] <= MineralWalkThreshold):
            # edge pass through only unmineralized elements - can't be traversed by osteoblasts
            edge_set.remove(eid)
            
    if len(edge_set) > 0:
        # neighbors that are reachable and not occupied
        open_neighbors = np.array([n for n in np.unique(grid.Edges[np.array(list(edge_set))]) if (n not in grid.NodeAgents) or (grid.NodeData['CellsAllowed'][n] == 0)])

        open_neighbors = open_neighbors[open_neighbors != agent.node]
        if len(open_neighbors) == 0:
            return
        
        contact_inhibited_probability = len(open_neighbors)/6 * ProlifProb
        if np.random.rand() < contact_inhibited_probability:
            step = np.random.choice(open_neighbors)
            grid.add_agent(step, agent.state, agent.age, agent.parameters)
                
@numba.njit
def apoptose(agent, grid):
    """
    Cell apoptosis. Random change that the agent will undergo apoptosis and be
    removed from the grid. 

    Parameters
    ----------
    agent : myabm.Agent
        Cell agent to perform the action.
    grid : myabm.AgentGrid
        Grid environment that the agent is in.

    Returns
    -------
    None.
    """    
    TimeStep = grid.TimeStep
    ApopProb = 1 - np.exp(-agent.parameters['ApopRate'] * TimeStep)
    if (np.random.rand() < ApopProb):
        grid.remove_agent(agent.node)
            
@numba.njit
def differentiate(agent, grid):
    TimeStep = grid.TimeStep
    DiffProb = 1 - np.exp(-agent.parameters['DiffRate'] * TimeStep)
    
    if agent.state == 'msc':
        if (agent.age > 6) & (np.random.rand() < DiffProb):
            agent.state = 'osteoblast'
            
@numba.njit
def differentiate_prendergast(agent, grid):
        
    TimeStep = grid.TimeStep
    DiffProb = 1 - np.exp(-agent.parameters['DiffRate'] * TimeStep)
    
    S = grid.NodeData['Stimulus'][agent.node]
    if agent.state == 'msc':
        # stimulus should be "S" from the prendergast model
        if (agent.age > 6) & (np.random.rand() < DiffProb):
            if 0.01 <= S <= 1:
                agent.state = 'osteoblast'
                agent.age = 0
                agent.parameters['ProlifRate'] = 0.3 # Isaksson 2008
                agent.parameters['ApopRate'] = 0.16 # Isaksson 2008
                agent.parameters['MigrRate'] = 0.96/2 # Half MSCs - CURRENTLY ARBITRARY, NEED SUPPORT
                agent.parameters['Production'] = 3e-6 # (mm^3/cell/day) Isaksson 2008
                
            elif 1 < S <= 3:
                agent.state = 'chondrocyte'
                agent.age = 0
                agent.parameters['ProlifRate'] = 0.2 # Isaksson 2008
                agent.parameters['ApopRate'] = 0.1 # Isaksson 2008
                agent.parameters['MigrRate'] = 0.0 # Checa 2011 (See also Morales 2007)
                agent.parameters['Production'] = 5e-6 # (mm^3/cell/day) Isaksson 2008
                
            elif 3 < S:
                agent.state = 'fibroblast'
                agent.age = 0
                agent.parameters['ProlifRate'] = 0.55 # Isaksson 2008
                agent.parameters['ApopRate'] = 0.05 # Isaksson 2008
                agent.parameters['MigrRate'] = 0.96 # Same as MSCs
                agent.parameters['Production'] = 5e-6 # (mm^3/cell/day) Isaksson 2008
    elif agent.state == 'osteoblast':
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
def produce(agent, grid):
    TimeStep = grid.TimeStep
    Production = TimeStep*agent.parameters['Production']
    
    neighbor_elems = grid.ElemConn[agent.node]
    neighbor_elems = neighbor_elems[neighbor_elems >= 0]
    
    # ECM production
    ProducedFraction = np.minimum(Production/grid.ElementVolume, 1) # Production can never exceed the total element volume
    
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
            new_nodes = grid.NodeConn[neighbor_elems[new_voxels]].flatten()
            old_nodes = grid.NodeConn[neighbor_elems[old_voxels]].flatten()
            new_nodes = np.array([n for n in new_nodes if n not in old_nodes])
        else:
            new_nodes = grid.NodeConn[neighbor_elems[new_voxels]].flatten()
        new_nodes = np.array([n for n in new_nodes if n not in grid.NodeAgents and grid.NodeData['CellsAllowed'][n]])
        # print(new_nodes)
        if len(new_nodes) > 0:
            step = np.random.choice(new_nodes)
            del grid.NodeAgents[agent.node]          # Remove cell from old node
            grid.NodeAgents[step] = agent            # Move cell to new node
            agent.node = step                        # Update cell's node id
      
@numba.njit
def produce_oriented(agent, grid):

    for required in ['Production', 'Production Baseline', 'Kcurve', 'ncurve']:
        assert required in agent.parameters, 'agent.parameters must contain "'+required+'"'
    for required in ['Mineral Walk Threshold', 'Tissue Threshold']:
        assert required in grid.parameters, 'agent_grid.parameters must contain "'+required+'"'
    for required in ['Mineral Density', 'Volume Fraction', 'ECM Fraction', 'Fibrous Fraction', 'Cartilaginous Fraction', 'Osseous Fraction']:
        assert required in grid.ElemData, 'agent_grid.ElemData must contain "'+required+'"'
    for required in ['Min Principal Curvature','Max Principal Curvature','Mean Curvature','CellsAllowed']:
        assert required in grid.NodeData, 'agent_grid.NodeData must contain "'+required+'"'
    for required in ['Min Principal Direction','Max Principal Direction']:
        assert required in grid.NodeVectorData, 'agent_grid.NodeData must contain "'+required+'"'

    TimeStep = grid.TimeStep
    Production = TimeStep*agent.parameters['Production']
    Kcurve = agent.parameters['Kcurve']
    ncurve = agent.parameters['ncurve']
    
    neighbor_elems = grid.ElemConn[agent.node]
    neighbor_elems = neighbor_elems[neighbor_elems >= 0]
    
    # ECM production
    # Constant tissue production
    # ProducedFraction = np.minimum(Production/grid.ElementVolume, 1) # Production can never exceed the total element volume
    # Mean Curvature-based tissue production
    a = np.abs(np.minimum(0, grid.NodeData['Mean Curvature'][agent.node]))
    CurvatureProduction = Production*a**ncurve / (Kcurve**ncurve + a**ncurve) + agent.parameters['Production Baseline']
    ProducedFraction = np.minimum(CurvatureProduction/grid.ElementVolume, 1)
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
            new_nodes = grid.NodeConn[neighbor_elems[new_voxels]].flatten()
            old_nodes = grid.NodeConn[neighbor_elems[old_voxels]].flatten()
            new_nodes = np.array([n for n in new_nodes if n not in old_nodes])
        else:
            new_nodes = grid.NodeConn[neighbor_elems[new_voxels]].flatten()
        new_nodes = np.array([n for n in new_nodes if n not in grid.NodeAgents and grid.NodeData['CellsAllowed'][n]])
        # print(new_nodes)
        if len(new_nodes) > 0:
            step = np.random.choice(new_nodes)
            del grid.NodeAgents[agent.node]          # Remove cell from old node
            grid.NodeAgents[step] = agent            # Move cell to new node
            agent.node = step                        # Update cell's node id

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
    dmineraldt = np.maximum(grid.parameters['Mineralization Rate'] * (grid.ElemData['Osseous Fraction'] - grid.ElemData['Mineral Density']/grid.parameters['Max Mineral']) * grid.parameters['Mineral Solute Concentration'], 0)
    grid.ElemData['Mineral Density'] += dmineraldt*grid.TimeStep


# Model Actions
def update_curvature(model):

    # Surface Curvature
    Tissue = model.mesh.Threshold(model.agent_grid.ElemData['Volume Fraction'], model.agent_grid.parameters['Tissue Threshold'])
    Tissue.ElemData['material'] = np.array([x for i,x in enumerate(model.agent_grid.ElemData['material']) if model.agent_grid.ElemData['Volume Fraction'][i] > model.agent_grid.parameters['Tissue Threshold']])
    Surf = Tissue.Surface

    FixedNodes = set()
    constraints = np.empty((0,3))

    Surf = improvement.LocalLaplacianSmoothing(Surf, options=dict(FixedNodes=FixedNodes, limit=np.sqrt(3)/2*model.agent_grid.h, constraint=constraints))

    TissueNodes =  np.zeros(Tissue.NNode,dtype=int)
    TissueNodes[np.intersect1d(np.unique(Tissue.NodeConn[Tissue.ElemData['material'] !=8]),Tissue.SurfNodes)] = 1
    NonTissueNodes = np.where(TissueNodes == 0)[0]
    TissueNodesPlus = np.unique(Surf.NodeConn[np.any(TissueNodes[Surf.NodeConn],axis=1)])
    # Surf.NodeCoords[NonTissueNodes] = Scaf.NodeCoords[NonTissueNodes]
    SurfNodeNeighbors =  mymesh.utils.getNodeNeighborhood(*mymesh.converter.surf2tris(*Surf), 2)
    k1, k2, k1v, k2v = mymesh.curvature.CubicFit(Surf.NodeCoords, Surf.NodeConn, SurfNodeNeighbors, Surf.NodeNormals, return_directions=True)

    Surf.NodeData['Max Principal Curvature'] = k1
    Surf.NodeData['Min Principal Curvature'] = k2
    Surf.NodeData['Max Principal Direction'] = k1v
    Surf.NodeData['Min Principal Direction'] = k2v
    Surf.NodeData['Mean Curvature'] = mymesh.curvature.MeanCurvature(Surf.NodeData['Max Principal Curvature'], Surf.NodeData['Min Principal Curvature'])
    Surf.NodeData['Gaussian Curvature'] = mymesh.curvature.GaussianCurvature(Surf.NodeData['Max Principal Curvature'], Surf.NodeData['Min Principal Curvature'])
    SurfIdx = ~np.isnan(Surf.NodeData['Mean Curvature'])

    # Tissue Curvature
    
    O1norm = np.linalg.norm(model.agent_grid.ElemVectorData['Tissue Orientation'],axis=1)[:,None]
    O1 = np.divide(model.agent_grid.ElemVectorData['Tissue Orientation'], O1norm, where=O1norm != 0, out=np.zeros_like(model.agent_grid.ElemVectorData['Tissue Orientation']))
    O2norm = np.linalg.norm(model.agent_grid.ElemVectorData['Tissue Orientation Orthogonal'],axis=1)[:,None]
    O2 = np.divide(model.agent_grid.ElemVectorData['Tissue Orientation Orthogonal'], O2norm, where=O2norm != 0, out=np.zeros_like(model.agent_grid.ElemVectorData['Tissue Orientation Orthogonal']))

    
    normals = np.cross(O2, O1)
    normals[model.agent_grid.ElemData['Volume Fraction'] < model.agent_grid.parameters['Tissue Threshold']] = np.nan     # This leads to nan values in mean curvature for in areas of no tissue
    model.agent_grid.ElemVectorData['n'] = normals
    
    nx = np.pad(mymesh.converter.voxel2im(*model.mesh, normals[:,0]),1)
    ny = np.pad(mymesh.converter.voxel2im(*model.mesh, normals[:,1]),1)
    nz = np.pad(mymesh.converter.voxel2im(*model.mesh, normals[:,2]),1)
    
    dnxdx = ((nx[:-1,:-1,1:] - nx[:-1,:-1,:-1]) +  # f[i+1] - f[i]
            (nx[:-1,1:,1:] - nx[:-1,1:,:-1]) + 
            (nx[1:,:-1,1:] - nx[1:,:-1,:-1]) + 
            (nx[1:,1:,1:] - nx[1:,1:,:-1]))
    dnxdx[~np.isnan(dnxdx)] /= 4*model.agent_grid.h
    
    dnydy = ((ny[:-1,1:,:-1] - ny[:-1,:-1,:-1]) + 
            (ny[:-1,1:,:1] - ny[:-1,:-1,:1]) + 
            (ny[1:,1:,:-1] - ny[1:,:-1,:-1]) + 
            (ny[1:,1:,1:] - ny[1:,:-1,1:]))
    dnydy[~np.isnan(dnydy)] /= 4*model.agent_grid.h
    
    dnzdz = ((nz[1:,:-1,:-1] - nz[:-1,:-1,:-1]) + 
            (nz[1:,1:,:-1] - nz[:-1,1:,:-1]) + 
            (nz[1:,:-1,1:] - nz[:-1,:-1,1:]) + 
            (nz[1:,1:,1:] - nz[:-1,1:,1:]))
    dnzdz[~np.isnan(dnzdz)] /= 4*model.agent_grid.h
    
    MeanCurvature = ((dnxdx + dnydy + dnzdz)).flatten(order='F')
    MeanCurvature[~np.isnan(MeanCurvature)] /= 2

    MeanCurvature[SurfIdx] = Surf.NodeData['Mean Curvature'][SurfIdx]

    model.agent_grid.NodeData['Max Principal Curvature'] = np.full(len(MeanCurvature),  np.nan)
    model.agent_grid.NodeVectorData['Max Principal Direction'] = np.full((len(MeanCurvature),3),  np.nan)
    model.agent_grid.NodeData['Min Principal Curvature'] = np.full(len(MeanCurvature),  np.nan)
    model.agent_grid.NodeVectorData['Min Principal Direction'] = np.full((len(MeanCurvature),3),  np.nan)

    model.agent_grid.NodeData['Max Principal Curvature'][SurfIdx] = Surf.NodeData['Max Principal Curvature'][SurfIdx] 
    model.agent_grid.NodeVectorData['Max Principal Direction'][SurfIdx] = Surf.NodeData['Max Principal Direction'][SurfIdx] 
    model.agent_grid.NodeData['Min Principal Curvature'][SurfIdx] = Surf.NodeData['Min Principal Curvature'][SurfIdx] 
    model.agent_grid.NodeVectorData['Min Principal Direction'][SurfIdx] = Surf.NodeData['Min Principal Direction'][SurfIdx] 

    MeanCurvature[np.isnan(MeanCurvature)] = sys.float_info.max

    
    NodeNeighbors = utils.PadRagged(model.mesh.NodeNeighbors)
    NodeNeighbors = np.vstack([NodeNeighbors, np.zeros(len(NodeNeighbors[0]),dtype=int)])
    MeanCurvature = np.append(MeanCurvature,np.inf)
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

def update_curvature_symmetric(model):
    pass

def update_curvature_periodic(model):
    pass

def run_compression(model):

    matprop = (self.agent_grid.ElemData['modulus'],
                self.agent_grid.ElemData['poisson'],
                self.agent_grid.ElemData['permeability'],
                self.agent_grid.ElemData['solid bulk modulus'],
                self.agent_grid.ElemData['fluid bulk modulus'],
                self.agent_grid.ElemData['fluid specific weight'],
                self.agent_grid.ElemData['porosity'])
    disp = model.parameters['displacement']
    try:
        apdl = pyAnsysSolvers.Launch(ansys_path, nproc=nproc, smp=True, port=port)
        M = pyAnsysSolvers.PoroelasticUniaxial(M, *matprop, timestep, disp, axis=2, boundary_eps=0, SurfacePressure=0, nsubsteps=1, constrained=False, mode='disp', mapdl=apdl)
    except:
        apdl = pyAnsysSolvers.Launch(ansys_path, nproc=nproc, smp=True, port=port)
        M = pyAnsysSolvers.PoroelasticUniaxial(M, *matprop, timestep, disp, axis=2, boundary_eps=0, SurfacePressure=0, nsubsteps=1, constrained=False, mode='disp', mapdl=apdl)


def update_properties(model):
    
    WaterProps = mechanobiology.prendergast_mat(np.repeat(0, agent_grid.NElem))
    NeoProps = mechanobiology.prendergast_mat(np.repeat(1, agent_grid.NElem))
    FibProps = mechanobiology.prendergast_mat(np.repeat(2, agent_grid.NElem))
    CartProps = mechanobiology.prendergast_mat(np.repeat(3, agent_grid.NElem))
    MarrowProps = mechanobiology.prendergast_mat(np.repeat(4, agent_grid.NElem))
    IboneProps = mechanobiology.prendergast_mat(np.repeat(5, agent_grid.NElem))     # Immature bone
    LboneProps = mechanobiology.prendergast_mat(np.repeat(7, agent_grid.NElem))     # Lamellar bone
    ScaffProps = mechanobiology.prendergast_mat(np.repeat(8, agent_grid.NElem))
    
    
    def modulus_mineral(mineral_density): 
        modulus = np.zeros_like(mineral_density)
        modulus[mineral_density < 0.4] = 32.5*mineral_density[mineral_density < 0.4]**3 +10.5*mineral_density[mineral_density < 0.4]**2 + mineral_density[mineral_density < 0.4] + 0.01
        # modulus[mineral_density < 0.4] = 63.4375*mineral_density[mineral_density < 0.4]**3 - 8.0625*mineral_density[mineral_density < 0.4]**2 + mineral_density[mineral_density < 0.4] + 1
        modulus[mineral_density >= 0.4] = 25 * mineral_density[mineral_density >= 0.4] - 5.83
        
        # Above are in GPa
        modulus *= 1000 # now in MPa
        
        return modulus
    
    # Calculate properties of mineralized tissue
    mineral_density = np.divide(agent_grid.ElemData['Mineral Density'], agent_grid.ElemData['Osseous Fraction'], where=agent_grid.ElemData['Osseous Fraction']>0, out=np.zeros(agent_grid.NElem))
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


    self.agent_grid.ElemData['modulus'] = E
    self.agent_grid.ElemData['poisson'] = nu
    self.agent_grid.ElemData['permeability'] = L
    self.agent_grid.ElemData['solid bulk modulus'] = SolidBulk
    self.agent_grid.ElemData['fluid bulk modulus'] = FluidBulk
    self.agent_grid.ElemData['fluid specific weight'] = FluidSpecWeight
    self.agent_grid.ElemData['porosity'] = Porosity
