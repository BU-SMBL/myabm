#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Thu Jun  5 11:08:02 2025
# @author: toj
"""

Setup functions for initializing ortho models
=============================================
.. autosummary::
    :toctree: submodules/

    implicit_scaffold
    demo_block
    wellplate
"""
import sys, os, warnings,  copy
import tqdm
import numpy as np
from mymesh import converter, curvature, image, implicit, improvement, mesh, primitives, utils, quality
from .. import Model, AgentGrid, Agent
from . import actions, geometries, mechanobiology
from . import OrthoModel

def implicit_scaffold(func, bounds, h, seeding_density=1e3, agent_parameters=None, filled=False, ncells=None):
    """
    Setup an Ortho model with a scaffold defined by an implicit function

    Parameters
    ----------
    func : callable
        Implicit function of the form :math:`\phi = f(x,y,z)`
    bounds : array_like
        6 element list of bounds, :code:`[minx, maxx, miny, maxy, minz, maxz]`
    h : float
        Grid spacing
    seeding_density : float, optional
        Density of cells to seed on the scaffold, in cells/mm^3, by default 1e3
    agent_parameters : dict, optional
        Dictionary of parameters to use for the seeded cells, by default None.
        If None are provided, default parameters for MSCs are used
    filled : bool, optional
        If True, generate a model where the scaffold is pre-filled with tissue, by default False
    ncells : int, optional
        Number of cells to seed on the surface. If provided, will override 
        :code:`seeding_density`, by default None

    Returns
    -------
    modek_description_ : myabm.ortho.OrthoModel
        Initialized model seeded with cells
    """

    # seeding_density is default, but if ncells is provided it takes precedence
    
    # Create mesh grid
    Grid = primitives.Grid(bounds, h)
    Grid.verbose=False
    model = OrthoModel(Grid)
    model.agent_grid.parameters['h'] = h

    # initialize properties
    if filled is not False:
        
        model.agent_grid.ElemData['material'] = np.ones(Grid.NElem)
        if isinstance(filled, (int, float, np.number)):
            fill = filled
        else:
            fill = 1
    else:
        model.agent_grid.ElemData['material'] = np.zeros(Grid.NElem)
    model.agent_grid.ElemData['material'][func(*Grid.Centroids.T) < 0] = 8 # Scaffold
    # Grid.ElemData['age'] = np.zeros(Grid.NElem)
    Pore = Grid.Threshold(model.agent_grid.ElemData['material'], 8, '!=')
    Pore.verbose=False
    exclude = np.setdiff1d(Grid.SurfNodes, Pore.SurfNodes)
    # Create agent grid

    model.agent_grid.NodeData['Cells Allowed'] = np.ones(model.agent_grid.NNode)
    model.agent_grid.NodeData['Cells Allowed'][exclude] = 0 
    model.agent_grid.ElemData['Scaffold Fraction'][model.agent_grid.ElemData['material'] == 8] = 1
    model.agent_grid.ElemData['Volume Fraction'][model.agent_grid.ElemData['material'] == 8] = 1
    if filled:
        model.agent_grid.ElemData['Volume Fraction'][model.agent_grid.ElemData['material'] == 1] = fill
        model.agent_grid.ElemData['ECM Fraction'][model.agent_grid.ElemData['material'] == 1] = fill
    model.agent_grid.ElemData['Mineral Density'][model.agent_grid.ElemData['material'] == 8] = 3 # mg/mm^3
    
    
    if filled:
        # Random tissue orientation
        idx = model.agent_grid.ElemData['material'] == 1
        model.agent_grid.ElemVectorData['Tissue Orientation'] = np.zeros((Grid.NElem ,3))
        model.agent_grid.ElemVectorData['Tissue Orientation Orthogonal'] = np.zeros((Grid.NElem ,3))
        model.agent_grid.ElemVectorData['Tissue Orientation'][idx] = np.random.rand(np.sum(idx), 3)
        model.agent_grid.ElemVectorData['Tissue Orientation'][idx] /= np.linalg.norm(model.agent_grid.ElemVectorData['Tissue Orientation'][idx],axis=1)[:,None]
        rand_vec = np.random.rand(np.sum(idx), 3)
        rand_vec /= np.linalg.norm(rand_vec,axis=1)[:,None]
        model.agent_grid.ElemVectorData['Tissue Orientation Orthogonal'][idx] = np.cross(model.agent_grid.ElemVectorData['Tissue Orientation'][idx], rand_vec)
        model.agent_grid.ElemVectorData['Tissue Orientation Orthogonal'][idx] /= np.linalg.norm(model.agent_grid.ElemVectorData['Tissue Orientation Orthogonal'][idx], axis=1)[:,None]
        
    # Seed scaffold
    Scaffold = Grid.Threshold(model.agent_grid.ElemData['material'], 8, '==')
    Scaffold.verbose=False
    ScafSurfNodes = np.setdiff1d(Scaffold.SurfNodes, np.arange(Grid.NNode)[model.agent_grid.NodeData['Cells Allowed']==0])
    
    if filled:
        tissue = Grid.Threshold(model.agent_grid.ElemData['Volume Fraction'], 0.2, '>=')
        ScafInteriorNodes = np.setdiff1d(Scaffold.MeshNodes, Scaffold.SurfNodes)
        
        SeedNodes = np.setdiff1d(
                np.union1d(np.setdiff1d(tissue.MeshNodes, np.arange(Grid.NNode)[model.agent_grid.NodeData['Cells Allowed']==0]), ScafSurfNodes),
                ScafInteriorNodes)
    else:
        SeedNodes = ScafSurfNodes

    nSurfElems = np.sum(np.all(np.isin(Scaffold.SurfConn, SeedNodes),axis=1))
    SurfArea = nSurfElems * h**2
    
    if ncells is None:
        ncells = int(np.round(seeding_density*SurfArea))
    
    model.seed(ncells, 'msc', SeedNodes, 'random', agent_parameters)
    

    return model

def demo_block(h, ncells=1, agent_parameters=None):
    """
    Generate a 2x2 voxel block, used to demonstrate agent actions

    Parameters
    ----------
    h : float
        grid spacing
    ncells : int, optional
        Number of cells to place on the block, by default 1
    agent_parameters : dict, optional
        Dictionary of parameters to use for the seeded cells, by default None.
        If None are provided, default parameters for MSCs are used

    Returns
    -------
    model : myabm.ortho.OrthoModel
        Initialized model of the demo block
    """
    dt = 0.02
    h = .025
    func, bounds = geometries.demo_block(h)
    model = implicit_scaffold(func, bounds, h, ncells=ncells, agent_parameters=agent_parameters)
    # agent_grid.NodeData['Cells Allowed'] = np.repeat(np.int32(1), agent_grid.NNode) 
    model.agent_grid.TimeStep = dt
    node = list(model.agent_grid.NodeAgents.keys())[0]
    if ncells == 1:
        if node != 29:
            model.agent_grid.move_agent(model.agents[0], 29)
    return model

def wellplate(size, h, media_volume=False, zstep=None, ncells=None, parameters=None):
    """
    Generate a model of the well of a tissue culture well plate

    
    Well plate specifications based on: 
    `Useful Numbers for Cell Culture <https://www.thermofisher.com/us/en/home/references/gibco-cell-culture-basics/cell-culture-protocols/cell-culture-useful-numbers.html>`_

    +------------+-------------------------------------+---------------------------+-----------------------------+
    | Well Plate | Surface Area (mm\ :superscript:`2`) | Growth Medium Volume (mL) | Recommended Seeding Density |
    +============+=====================================+===========================+=============================+
    | 6-well     | 960                                 | 2                         | 0.3e6 cells/well            |
    +------------+-------------------------------------+---------------------------+-----------------------------+
    | 12-well    | 350                                 | 1                         | 0.1e6 cells/well            |
    +------------+-------------------------------------+---------------------------+-----------------------------+
    | 24-well    | 190                                 | 0.75                      | 0.05e6 cells/well,          |
    +------------+-------------------------------------+---------------------------+-----------------------------+
    | 48-well    | 110                                 | 0.3                       | 0.03e6 cells/well           |
    +------------+-------------------------------------+---------------------------+-----------------------------+
    | 96-well    | 32                                  | 0.15                      | 0.01e6 cells/well           | 
    +------------+-------------------------------------+---------------------------+-----------------------------+
    | 384-well   | 8.4                                 | 0.15                      | 0.0018e6 cells/well         | 
    +------------+-------------------------------------+---------------------------+-----------------------------+


    Parameters
    ----------
    size : int
        Size of the well plate, in terms of the number of wells on the plate:
        6, 12, 24, 48, 96
    h : float
        grid spacing
    media_volume : bool
        Volume of media in the well, used to set a height of empty space in 
        the mesh grid above the plate, by default False.
    zstep : float
        Grid spacing in the z direction, used if media_volume is True. By default, equal to h.
    ncells : float
        Number of cells to seed in the well
    parameters : dict, optional
        Dictionary of parameters to use for the seeded cells, by default None.
        If None are provided, default parameters for MSCs are used

    Returns
    -------
    model : myabm.ortho.OrthoModel
        Initialized model of the well plate, seeded with cells

    """
    func, bounds = geometries.wellplate(size, media_volume=media_volume)
    # Create mesh grid
    if zstep is None:
        zstep = h
    Grid = primitives.Grid(bounds, (h, h, zstep))
    Grid2 = primitives.Grid([bounds[0], bounds[1], bounds[2], bounds[3], -h, bounds[4]], h)
    Grid.merge(Grid2)
    if size == 6:
        r = np.sqrt(960/np.pi)
        if ncells is None:
            ncells = 0.3e6
    elif size == 12:
        r = np.sqrt(350/np.pi)
        if ncells is None:
            ncells = 0.1e6
    elif size == 24:
        r = np.sqrt(190/np.pi)
        if ncells is None:
            ncells = 0.05e6
    elif size == 48:
        r = np.sqrt(110/np.pi)
        if ncells is None:
            ncells = 0.03e6
    elif size == 96:
        r = np.sqrt(32/np.pi)
        if ncells is None:
            ncells = 0.01e6
    elif size == 384:
        r = np.sqrt(.084/np.pi)
        if ncells is None:
            ncells = 0.0018e6
    else:
        raise ValueError('Invalid well plate size, must be one of: 384, 96, 48, 24, 12, 6.')
    Grid.verbose = False
    Grid = Grid.Threshold(implicit.cylinder([0,0,0], r)(*Grid.Centroids.T), 0, '<', InPlace=True, cleanup=True)

    model = OrthoModel(Grid)
    model.agent_grid.parameters['h'] = h

    # initialize properties
    model.agent_grid.ElemData['material'] = np.zeros(Grid.NElem)
    model.agent_grid.ElemData['material'][func(*Grid.Centroids.T) < 0] = 8 # Scaffold
    
    # Create agent grid
    # agent_grid = abm.initialize(Grid)
    model.agent_grid.parameters['Volume'] = h*h*zstep
    model.agent_grid.NodeData['Cells Allowed'][model.mesh.SurfNodes] = 0     # TODO: Should make this an option
    model.agent_grid.ElemData['Scaffold Fraction'][model.agent_grid.ElemData['material'] == 8] = 1
    model.agent_grid.ElemData['Volume Fraction'][model.agent_grid.ElemData['material'] == 8] = 1
    model.agent_grid.ElemData['Mineral Density'][model.agent_grid.ElemData['material'] == 8] = 3 # mg/mm^3
    
    model.agent_grid.ElemData['Volume Fraction'] = model.agent_grid.ElemData['Volume Fraction']
    # Seed scaffold
    Scaffold = Grid.Threshold(model.agent_grid.ElemData['material'], 8, '==')
    Scaffold.verbose=False
    
    SeedNodes = np.setdiff1d(Scaffold.SurfNodes, Grid.SurfNodes)
    nSurfElems = np.sum(np.all(np.isin(Scaffold.SurfConn, SeedNodes),axis=1))
    SurfArea = nSurfElems * h**2
    # ncells = int(np.round(seeding_density*SurfArea))
    
    model.seed(ncells, 'msc', SeedNodes, parameters=parameters)

    return model
