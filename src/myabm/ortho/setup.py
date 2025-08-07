#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 11:08:02 2025

@author: toj
"""
import sys, os, warnings,  copy
import tqdm
import numpy as np
from mymesh import converter, curvature, image, implicit, improvement, mesh, primitives, utils, quality
from .. import Model, AgentGrid, Agent
from . import actions, geometries, mechanobiology
from . import OrthoModel

def implicit_scaffold(func, bounds, h, seeding_density=1e3, agent_parameters=None, filled=False, ncells=None):
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

# def setup_wellplate(size, h, zstep=None, media_volume=True, seeding_density=None, parameters=None):
#     """
#     Well plate specifications based on: 
#     `Useful Numbers for Cell Culture<https://www.thermofisher.com/us/en/home/references/gibco-cell-culture-basics/cell-culture-protocols/cell-culture-useful-numbers.html>_`

#     +============+====================+===========================+==========================================+
#     | Well Plate | Surface Area (mm2) | Growth Medium Volume (mL) | Recommended Seeding Density              |
#     +============+====================+===========================+==========================================+
#     | 6-well     | 960                | 2                         | 0.3e6 cells/well, 313 cells/mm2          |      |
#     +------------+--------------------+---------------------------+------------------------------------------+
#     | 12-well    | 350                | 1                         | 0.1e6 cells/well, 285 cells/mm2          |
#     +------------+--------------------+---------------------------+------------------------------------------+
#     | 24-well    | 190                | 0.75                      | 0.05e6 cells/well, 263 cells/mm2         |
#     +------------+--------------------+---------------------------+------------------------------------------+
#     | 48-well    | 110                | 0.3                       | 0.03e6 cells/well, 273 cells/mm2         |
#     +------------+--------------------+---------------------------+------------------------------------------+
#     | 96-well    | 32                 | 0.15                      | 0.01e6 cells/well, 313 cells/mm2         | 
#     +------------+--------------------+---------------------------+------------------------------------------+
#     """
#     func, bounds = geometries.wellplate(size, media_volume=media_volume)
#     # Create mesh grid
#     if zstep is None:
#         zstep = h
#     Grid = primitives.Grid(bounds, (h, h, zstep))
#     Grid2 = primitives.Grid([bounds[0], bounds[1], bounds[2], bounds[3], -h, bounds[4]], h)
#     Grid.merge(Grid2)
#     if size == 6:
#         r = np.sqrt(960/np.pi)
#         if seeding_density is None:
#             seeding_density = 313
#     elif size == 12:
#         r = np.sqrt(350/np.pi)
#         if seeding_density is None:
#             seeding_density = 285
#     elif size == 24:
#         r = np.sqrt(190/np.pi)
#         if seeding_density is None:
#             seeding_density = 263
#     elif size == 48:
#         r = np.sqrt(110/np.pi)
#         if seeding_density is None:
#             seeding_density = 273
#     elif size == 96:
#         r = np.sqrt(32/np.pi)
#         if seeding_density is None:
#             seeding_density = 313
#     else:
#         raise ValueError('Invalid well plate size, must be one of: 96, 48, 24, 12, 6.')
#     Grid.verbose = False
#     Grid = Grid.Threshold(implicit.cylinder([0,0,0], r)(*Grid.Centroids.T), 0, '<', InPlace=True, cleanup=True)
    
#     # initialize properties
#     agent_grid.ElemData['material'] = np.zeros(Grid.NElem)
#     agent_grid.ElemData['material'][func(*Grid.Centroids.T) < 0] = 8 # Scaffold
#     # Grid.ElemData['age'] = np.zeros(Grid.NElem)
    
#     # Create agent grid
#     agent_grid = abm.initialize(Grid)
#     agent_grid.ElementVolume = h*h*zstep
#     agent_grid.NodeData['Cells Allowed'][agent_grid.SurfNodes] = 0     # TODO: Should make this an option
#     agent_grid.ElemData['Scaffold Fraction'][agent_grid.ElemData['material'] == 8] = 1
#     agent_grid.ElemData['Volume Fraction'][agent_grid.ElemData['material'] == 8] = 1
#     agent_grid.ElemData['Mineral Density'][agent_grid.ElemData['material'] == 8] = 3 # mg/mm^3
    
#     agent_grid.ElemData['Volume Fraction'] = agent_grid.ElemData['Volume Fraction']
#     # Seed scaffold
#     Scaffold = Grid.Threshold(agent_grid.ElemData['material'], 8, '==')
#     Scaffold.verbose=False
    
#     SeedNodes = np.setdiff1d(Scaffold.SurfNodes, Grid.SurfNodes)
#     nSurfElems = np.sum(np.all(np.isin(Scaffold.SurfConn, SeedNodes),axis=1))
#     SurfArea = nSurfElems * h**2
#     ncells = int(np.round(seeding_density*SurfArea))
    
#     agent_grid.seed(ncells, 'msc', SeedNodes, 'random', parameters)
#     return Grid, agent_grid

# def CalculateSurfaceCurvature(Grid, h, TissueThreshold=0.2, smooth_scaffold=False, func=None, symmetric=False, periodic=False, nrings=2, Scaf=None):
    
    
#     if symmetric:
#         # Remove closed faces of the symmetric planes
#         # Surf = Surf.Threshold(((Surf.Centroids[:,0] == np.max(Scaf.Surface.Centroids[:,0])) | (Surf.Centroids[:,1] == np.max(Scaf.Surface.Centroids[:,1])) | (Surf.Centroids[:,2] == np.max(Scaf.Surface.Centroids[:,2]))).astype(int), 0, '==')
#         # mirror
#         # Grid.NodeData['original'] = np.ones(Grid.NNode)
#         pad_width = 4
#         pad_dist = (pad_width-1)*h
#         Grid.NodeData['id'] = np.arange(Grid.NNode)
#         maxxn = Grid.NodeCoords[:,0].max()
#         maxyn = Grid.NodeCoords[:,1].max()
#         maxzn = Grid.NodeCoords[:,2].max()
#         pad = Grid.Threshold(Grid.Centroids[:,0], Grid.Centroids[:,0].max()-pad_dist, '>=', cleanup=True).Mirror(x=Grid.NodeCoords[:,0].max(), InPlace=True)
#         pad.merge(Grid.Threshold(Grid.Centroids[:,1], Grid.Centroids[:,1].max()-pad_dist, '>=', cleanup=True).Mirror(y=Grid.NodeCoords[:,1].max(), InPlace=True))
#         pad.merge(Grid.Threshold(Grid.Centroids[:,2], Grid.Centroids[:,2].max()-pad_dist, '>=', cleanup=True).Mirror(z=Grid.NodeCoords[:,2].max(), InPlace=True))
#         v = quality.Volume(*pad)
#         pad.NodeConn[v < 0] = np.hstack((pad.NodeConn[v < 0, 4:], pad.NodeConn[v < 0, :4]))
#         pad.NodeData['id'][pad.MeshNodes] = -1
#         Grid.merge(pad)
#         FixedNodes = np.where((Grid.NodeCoords[:,0] == Grid.NodeCoords[:,0].max()) | (Grid.NodeCoords[:,1] == Grid.NodeCoords[:,1].max()) | (Grid.NodeCoords[:,2] == Grid.NodeCoords[:,2].max()))[0]
#         xnodes = np.where(Grid.NodeCoords[:,0] == maxxn)[0]
#         ynodes = np.where(Grid.NodeCoords[:,1] == maxyn)[0]
#         znodes = np.where(Grid.NodeCoords[:,2] == maxzn)[0]
#         constraints = np.vstack((np.column_stack((xnodes, np.repeat(0, len(xnodes)), np.zeros(len(xnodes)))), 
#                                  np.column_stack((ynodes, np.repeat(1, len(ynodes)), np.zeros(len(ynodes)))),
#                                  np.column_stack((znodes, np.repeat(2, len(znodes)), np.zeros(len(znodes))))))
#     else:
#         FixedNodes = set()
#         constraints = np.empty((0,3))
            
#     # Scaffold Surface
#     if Scaf is None:
#         runscaf = True
#         Scaf = Grid.Threshold('material', 8)
#         Scaf.verbose = False
#         if smooth_scaffold:
            
#             maxxe = np.max(Scaf.Centroids[:,0])
#             minxe = np.min(Scaf.Centroids[:,0])
#             maxye = np.max(Scaf.Centroids[:,1])
#             minye = np.min(Scaf.Centroids[:,1])
#             maxze = np.max(Scaf.Centroids[:,2])
#             minze = np.min(Scaf.Centroids[:,2])
            
#             maxxn = np.max(Scaf.NodeCoords[:,0])
#             minxn = np.min(Scaf.NodeCoords[:,0])
#             maxyn = np.max(Scaf.NodeCoords[:,1])
#             minyn = np.min(Scaf.NodeCoords[:,1])
#             maxzn = np.max(Scaf.NodeCoords[:,2])
#             minzn = np.min(Scaf.NodeCoords[:,2])
            
#             if periodic:
                
#                 maxx_pad = Scaf.Threshold(Scaf.Centroids[:,0], minxe, '==')
#                 maxx_pad.NodeCoords[:,0] += (maxxe-minxe + h)
#                 minx_pad = Scaf.Threshold(Scaf.Centroids[:,0], maxxe, '==') 
#                 minx_pad.NodeCoords[:,0] -= (maxxe-minxe + h)
                
#                 maxy_pad = Scaf.Threshold(Scaf.Centroids[:,1], minye, '==')
#                 maxy_pad.NodeCoords[:,1] += (maxye-minye + h)
#                 miny_pad = Scaf.Threshold(Scaf.Centroids[:,1], maxye, '==') 
#                 miny_pad.NodeCoords[:,1] -= (maxye-minye + h)
                
#                 maxz_pad = Scaf.Threshold(Scaf.Centroids[:,2], minze, '==')
#                 maxz_pad.NodeCoords[:,2] += (maxze-minze + h)
#                 minz_pad = Scaf.Threshold(Scaf.Centroids[:,2], maxze, '==') 
#                 minz_pad.NodeCoords[:,2] -= (maxze-minze + h)
                                    
#                 pad = mesh()
#                 pad.merge([maxx_pad, minx_pad, maxy_pad, miny_pad, maxz_pad, minz_pad])
#                 Scaf_copy = Scaf.copy()
#                 Scaf.ElemData['original'] = np.ones(Scaf.NElem)
#                 pad.ElemData['original'] = np.zeros(pad.NElem)
#                 Scaf.merge(pad, cleanup=False)
#                 # Scaf.cleanup(tol=1e-9)
#                 Scaf.NodeCoords, Scaf.NodeConn, idx, inv = utils.DeleteDuplicateNodes(Scaf.NodeCoords, Scaf.NodeConn, return_inv=True, return_idx=True, tol=1e-9)
#                 Scaf._Surface = None
#                 Scaf.reset('SurfConn')
#                 Scaf._SurfNodes = None
            
#             xnodes = np.where((np.isclose(Scaf.NodeCoords[:,0], minxn)) | (np.isclose(Scaf.NodeCoords[:,0], maxxn)))[0]
#             ynodes = np.where((np.isclose(Scaf.NodeCoords[:,1], minyn)) | (np.isclose(Scaf.NodeCoords[:,1], maxyn)))[0]
#             znodes = np.where((np.isclose(Scaf.NodeCoords[:,2], minzn)) | (np.isclose(Scaf.NodeCoords[:,2], maxzn)))[0]
#             constraints = np.vstack([
#                 np.column_stack((xnodes, np.repeat(0, len(xnodes)), np.zeros(len(xnodes)))),
#                 np.column_stack((ynodes, np.repeat(1, len(ynodes)), np.zeros(len(ynodes)))),
#                 np.column_stack((znodes, np.repeat(2, len(znodes)), np.zeros(len(znodes)))),
#                 ])
                
#             boundaries = set(np.where(
#                     (Scaf.NodeCoords[:,0] == Scaf.NodeCoords[:,0].min()) |
#                     (Scaf.NodeCoords[:,0] == Scaf.NodeCoords[:,0].max()) |
#                     (Scaf.NodeCoords[:,1] == Scaf.NodeCoords[:,1].min()) |
#                     (Scaf.NodeCoords[:,1] == Scaf.NodeCoords[:,1].max()) |
#                     (Scaf.NodeCoords[:,2] == Scaf.NodeCoords[:,2].min()) |
#                     (Scaf.NodeCoords[:,2] == Scaf.NodeCoords[:,2].max())
#                 )[0].tolist())
                
#             # ScafSurf = improvement.LocalLaplacianSmoothing(Scaf.Surface, options=dict(limit=np.sqrt(3)/2*h, FixedNodes=boundaries))
#             ScafSurf = improvement.LocalLaplacianSmoothing(Scaf.Surface, options=dict(limit=np.sqrt(3)/2*h,  FixedNodes=boundaries, constraint=constraints))
#             Scaf.NodeCoords = ScafSurf.NodeCoords
#             if periodic:
#                 Scaf_copy.NodeCoords = Scaf.NodeCoords[inv[:Scaf_copy.NNode]]
#                 Scaf = Scaf_copy
#                 ScafSurf = Scaf.Surface
#         else:
#             ScafSurf = Scaf.Surface
#         ScafSurf.verbose = False
#     else:
#         runscaf = False
    
#     # Tissue surface
#     Tissue = Grid.Threshold('Volume Fraction', TissueThreshold)
#     Tissue.verbose=False
    
#     Surf = Tissue.Surface
#     Surf.verbose=False
    
#     NonScaff = Tissue.Threshold('material', 8, '!=')
#     NonScaff.verbose=False
#     # TissueNodes = np.intersect1d(Surf.SurfNodes, NonScaff.SurfNodes)
    
#     # FixedNodes = set() #Scaf.MeshNodes
#     Surf.verbose = False
#     Surf = improvement.LocalLaplacianSmoothing(Surf, options=dict(FixedNodes=FixedNodes, limit=np.sqrt(3)/2*h, constraint=constraints))
#     Surf.Type='surf'
#     Surf.verbose = False
    
#     # fix non tissue nodes
#     TissueNodes =  np.zeros(Tissue.NNode,dtype=int)
#     TissueNodes[np.intersect1d(np.unique(Tissue.NodeConn[Tissue.ElemData['material'] !=8]),Tissue.SurfNodes)] = 1
#     NonTissueNodes = np.where(TissueNodes == 0)[0]
#     TissueNodesPlus = np.unique(Surf.NodeConn[np.any(TissueNodes[Surf.NodeConn],axis=1)])
#     TissueNodes = np.where(TissueNodes == 1)[0]
#     Surf.NodeCoords[NonTissueNodes] = Scaf.NodeCoords[NonTissueNodes]
    
#     if runscaf:
#         k1, k2, k1v, k2v = curvature.CubicFit(ScafSurf.NodeCoords, ScafSurf.SurfConn, utils.getNodeNeighborhood(*converter.surf2tris(*ScafSurf), 1), ScafSurf.NodeNormals, return_directions=True)
#         Scaf.NodeData['Max Principal Curvature'] = k1#[scafsurfinv[:Scaf.NNode]]
#         Scaf.NodeData['Min Principal Curvature'] = k2#[scafsurfinv[:Scaf.NNode]]
#         Scaf.NodeData['Max Principal Direction'] = k1v#[scafsurfinv[:Scaf.NNode]]
#         Scaf.NodeData['Min Principal Direction'] = k2v#[scafsurfinv[:Scaf.NNode]]
#         Scaf.NodeData['Mean Curvature'] = curvature.MeanCurvature(k1, k2)#[scafsurfinv[:Scaf.NNode]]
#         Scaf.NodeData['Gaussian Curvature'] = curvature.GaussianCurvature(k1, k2)#[scafsurfinv[:Scaf.NNode]]
#     Surf.NodeData = copy.copy(Scaf.NodeData)
    
    
#     # if symmetric:
#     #     neighbors = utils.getNodeNeighborhood(*converter.surf2tris(*Surf), nrings)
#     #     idx = Surf.NNode
#     #     PaddedCoords = np.copy(Surf.NodeCoords)
#     #     NewCoords = []
#     #     NewNormals = []
#     #     # neighbors = copy.deepcopy(Surf.NodeNeighbors)
#     #     for node in Surf.BoundaryNodes:
#     #         n = Surf.NodeNormals[node]
#     #         for neighbor in Surf.NodeNeighbors[node]:
#     #             if neighbor not in Surf.BoundaryNodes:
#     #                 newpt = Surf.NodeCoords[neighbor].copy()
#     #                 # Rotate 180 about the central node/local z axis
#     #                 q = [np.cos(np.pi/2),               # Quaternion Rotation
#     #                      n[0]*np.sin(np.pi/2),
#     #                      n[1]*np.sin(np.pi/2),
#     #                      n[2]*np.sin(np.pi/2)]
                
#     #                 R = np.array([[2*(q[0]**2+q[1]**2)-1,   2*(q[1]*q[2]-q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2]), 0],
#     #                      [2*(q[1]*q[2]+q[0]*q[3]), 2*(q[0]**2+q[2]**2)-1,   2*(q[2]*q[3]-q[0]*q[1]), 0],
#     #                      [2*(q[1]*q[3]-q[0]*q[2]), 2*(q[2]*q[3]+q[0]*q[1]), 2*(q[0]**2+q[3]**2)-1,   0],
#     #                      [0,                       0,                       0,                       1]
#     #                      ])
                    
#     #                 newpt = (R @ np.append((newpt-Surf.NodeCoords[node]),0)[:,None])[:3,0] + Surf.NodeCoords[node]
#     #                 newn = (R @  np.append((Surf.NodeNormals[node]),0)[:,None])[:3,0]
#     #                 NewCoords.append(newpt)
#     #                 NewNormals.append(newn)
#     #                 neighbors[node].append(idx)
#     #                 idx += 1
#     #     SurfNodeCoords = np.vstack((Surf.NodeCoords, NewCoords))
#     #     SurfNodeNeighbors = neighbors
#     #     SurfNodeNormals = np.vstack((Surf.NodeNormals, NewNormals))
            
#     # else:
#     SurfNodeCoords = Surf.NodeCoords
#     SurfNodeNeighbors =  utils.getNodeNeighborhood(*converter.surf2tris(*Surf), nrings)
#     SurfNodeNormals = Surf.NodeNormals
        
#     # Tissue boundary curvature
#     k1, k2, k1v, k2v = curvature.CubicFit(SurfNodeCoords, Surf.NodeConn, SurfNodeNeighbors, SurfNodeNormals, return_directions=True)
#     Surf.NodeData['Max Principal Curvature'][TissueNodesPlus] = k1[TissueNodesPlus]
#     Surf.NodeData['Min Principal Curvature'][TissueNodesPlus] = k2[TissueNodesPlus]
#     Surf.NodeData['Max Principal Direction'][TissueNodesPlus] = k1v[TissueNodesPlus]
#     Surf.NodeData['Min Principal Direction'][TissueNodesPlus] = k2v[TissueNodesPlus]
#     Surf.NodeData['Mean Curvature'] = curvature.MeanCurvature(Surf.NodeData['Max Principal Curvature'], Surf.NodeData['Min Principal Curvature'])
#     Surf.NodeData['Gaussian Curvature'] = curvature.GaussianCurvature(Surf.NodeData['Max Principal Curvature'], Surf.NodeData['Min Principal Curvature'])
    
#     if symmetric:
#         ids = Grid.NodeData['id'].copy()
#         Grid = Grid.Threshold('id', 0, '>=', cleanup=False, InPlace=True)
#         Grid.NodeCoords, Grid.NodeConn, origids = utils.RemoveNodes(*Grid)
#         for key in Grid.NodeData.keys():
#             Grid.NodeData[key] = Grid.NodeData[key][origids]
        
#         # Don't unpad scaf because it'll get reused with the padded meshes
#         # if runscaf:
#         #     Scaf = Scaf.Threshold('id', 0, '>=', cleanup=False, InPlace=True)
#         #     replace = np.zeros(Scaf.NNode,dtype=int)
#         #     replace[origids] = np.arange(len(origids))
#         #     Scaf.NodeConn = replace[Scaf.NodeConn]
#         #     Scaf.NodeCoords = Scaf.NodeCoords[origids]
            
#         #     for key in Scaf.NodeData.keys():
#         #         Scaf.NodeData[key] = Scaf.NodeData[key][origids]
                
#         Surf = Surf.Threshold('id', 0, '>=', cleanup=False, InPlace=True)
        
#         replace = np.zeros(Surf.NNode,dtype=int)
#         replace[origids] = np.arange(len(origids))
#         Surf.NodeConn = replace[Surf.NodeConn]
#         Surf.NodeCoords = Surf.NodeCoords[origids]
        
#         for key in Surf.NodeData.keys():
#             Surf.NodeData[key] = Surf.NodeData[key][origids]
        
#         # NonTissueNodes = replace[NonTissueNodes]
    
#     # Surf.NodeCoords[NonTissueNodes] = Scaf.NodeCoords[NonTissueNodes]
    
    
#     if runscaf:
#         return Grid, Surf, Scaf
#     return Grid, Surf

# def CalculateTissueCurvature(Grid, h, normalize=True, TissueThreshold=0.2):
#     if normalize:
#         O1norm = np.linalg.norm(Grid.ElemData['Tissue Orientation'],axis=1)[:,None]
#         O1 = np.divide(Grid.ElemData['Tissue Orientation'], O1norm, where=O1norm != 0, out=np.zeros_like(Grid.ElemData['Tissue Orientation']))
#         O2norm = np.linalg.norm(Grid.ElemData['Tissue Orientation Orthogonal'],axis=1)[:,None]
#         O2 = np.divide(Grid.ElemData['Tissue Orientation Orthogonal'], O2norm, where=O2norm != 0, out=np.zeros_like(Grid.ElemData['Tissue Orientation Orthogonal']))
#     else:
#         O1 = Grid.ElemData['Tissue Orientation']
#         O2 = Grid.ElemData['Tissue Orientation Orthogonal']
        
#     normals = np.cross(O2, O1)
#     # NOTE: not sure if TissueThreshold should just be 0 here
#     normals[Grid.ElemData['Volume Fraction'] < TissueThreshold] = np.nan     # This leads to nan values in mean curvature for in areas of no tissue
#     Grid.ElemData['n'] = normals
#     # Grid.write('temp.vtu')
    
#     nx = np.pad(converter.voxel2im(*Grid, normals[:,0]),1)
#     ny = np.pad(converter.voxel2im(*Grid, normals[:,1]),1)
#     nz = np.pad(converter.voxel2im(*Grid, normals[:,2]),1)
    
    
#     dnxdx = ((nx[:-1,:-1,1:] - nx[:-1,:-1,:-1]) +  # f[i+1] - f[i]
#             (nx[:-1,1:,1:] - nx[:-1,1:,:-1]) + 
#             (nx[1:,:-1,1:] - nx[1:,:-1,:-1]) + 
#             (nx[1:,1:,1:] - nx[1:,1:,:-1]))
#     dnxdx[~np.isnan(dnxdx)] /= 4*h
    
#     dnydy = ((ny[:-1,1:,:-1] - ny[:-1,:-1,:-1]) + 
#             (ny[:-1,1:,:1] - ny[:-1,:-1,:1]) + 
#             (ny[1:,1:,:-1] - ny[1:,:-1,:-1]) + 
#             (ny[1:,1:,1:] - ny[1:,:-1,1:]))
#     dnydy[~np.isnan(dnydy)] /= 4*h
    
#     dnzdz = ((nz[1:,:-1,:-1] - nz[:-1,:-1,:-1]) + 
#             (nz[1:,1:,:-1] - nz[:-1,1:,:-1]) + 
#             (nz[1:,:-1,1:] - nz[:-1,:-1,1:]) + 
#             (nz[1:,1:,1:] - nz[:-1,1:,1:]))
#     dnzdz[~np.isnan(dnzdz)] /= 4*h
    
#     MeanCurvature = ((dnxdx + dnydy + dnzdz)).flatten(order='F')
#     MeanCurvature[~np.isnan(MeanCurvature)] /= 2
#     return MeanCurvature

# def CalculateCurvature(Grid, h, normalize=True, smooth_scaffold=False, func=None, symmetric=False, periodic=False, TissueThreshold=0.2, clip=(-75,75), nrings=2, Scaf=None):
    
#     # For curved surface scaffolds, set smooth_scaffold to True
    
#     # Surface curvature
#     if Scaf is None:
#         runscaf = True
#     else:
#         runscaf = False
#     if runscaf:
#         Grid, Surf, Scaf = CalculateSurfaceCurvature(Grid, h, symmetric=symmetric, periodic=periodic, smooth_scaffold=smooth_scaffold, TissueThreshold=TissueThreshold, nrings=nrings, Scaf=Scaf, func=func)
#         Scaf.NodeData['Mean Curvature'][Scaf.NodeData['Min Principal Curvature'] < clip[0]] = 0
#     else:
#         Grid, Surf = CalculateSurfaceCurvature(Grid, h, symmetric=symmetric, periodic=periodic, smooth_scaffold=smooth_scaffold, TissueThreshold=TissueThreshold, nrings=nrings, Scaf=Scaf)
#     SurfIdx = ~np.isnan(Surf.NodeData['Mean Curvature'])
#     Surf.NodeData['Mean Curvature'] = np.clip(Surf.NodeData['Mean Curvature'], *clip)
#     # Suppress extreme concave curvatures on the scaffold surface - this prevents edge effects on the scaffold surface leading to extreme growth spurts
#     Surf.NodeData['Mean Curvature'][np.intersect1d(np.where(Surf.NodeData['Min Principal Curvature'] < clip[0])[0], Scaf.SurfNodes)] = 0
#     # Scaffold curvature
#     # Grid, Scaf = CalculateScaffoldCurvature(Grid, phi0func, smooth=True) # References phi0func defined outside function
#     # Tisse curvature
#     MeanCurvature = CalculateTissueCurvature(Grid, h, normalize=normalize, TissueThreshold=TissueThreshold)
#     # Merge Surface, Scaffold, and tissue curvatures
#     MeanCurvature[SurfIdx] = Surf.NodeData['Mean Curvature'][SurfIdx]
#     MeanCurvature = np.clip(MeanCurvature, *clip)
#     # MeanCurvature[Scaf.SurfNodes] = Scaf.NodeData['Mean Curvature'][Scaf.SurfNodes] # Assign surf first so it can be overwritten by scaf if necessary
    
    
#     Grid.NodeData['Max Principal Curvature'] = np.full(len(MeanCurvature),  np.nan)
#     Grid.NodeData['Max Principal Direction'] = np.full((len(MeanCurvature),3),  np.nan)
#     Grid.NodeData['Min Principal Curvature'] = np.full(len(MeanCurvature),  np.nan)
#     Grid.NodeData['Min Principal Direction'] = np.full((len(MeanCurvature),3),  np.nan)
    
#     # Assign surf first so it can be overwritten by scaf if necessary
    
#     Grid.NodeData['Max Principal Curvature'][SurfIdx] = Surf.NodeData['Max Principal Curvature'][SurfIdx] 
#     Grid.NodeData['Max Principal Direction'][SurfIdx] = Surf.NodeData['Max Principal Direction'][SurfIdx] 
#     Grid.NodeData['Min Principal Curvature'][SurfIdx] = Surf.NodeData['Min Principal Curvature'][SurfIdx] 
#     Grid.NodeData['Min Principal Direction'][SurfIdx] = Surf.NodeData['Min Principal Direction'][SurfIdx] 
    
#     # Extend curvature to nan values (should correspond to unfilled tissue)
#     MeanCurvature[np.isnan(MeanCurvature)] = sys.float_info.max
#     # Grid.NodeData['Mean Curvature'] = MeanCurvature  
#     ###
#     NodeNeighbors = utils.PadRagged(Grid.NodeNeighbors)
#     NodeNeighbors = np.vstack([NodeNeighbors, np.zeros(len(NodeNeighbors[0]),dtype=int)])
#     MeanCurvature = np.append(MeanCurvature,np.inf)
#     mask = MeanCurvature == sys.float_info.max
#     i = 0
#     while np.any(mask):
#         i += 1
#         MeanCurvature[mask] = sys.float_info.max
#         minfilt = np.take_along_axis(NodeNeighbors, np.argmin(MeanCurvature[NodeNeighbors],axis=1)[:,None], 1).flatten()
#         MeanCurvature[mask] = MeanCurvature[minfilt][mask]
#         Grid.NodeData['Max Principal Curvature'][mask[:-1]] = Grid.NodeData['Max Principal Curvature'][minfilt[:-1]][mask[:-1]]
#         Grid.NodeData['Max Principal Direction'][mask[:-1]] = Grid.NodeData['Max Principal Direction'][minfilt[:-1]][mask[:-1]]
#         Grid.NodeData['Min Principal Curvature'][mask[:-1]] = Grid.NodeData['Min Principal Curvature'][minfilt[:-1]][mask[:-1]]
#         Grid.NodeData['Min Principal Direction'][mask[:-1]] = Grid.NodeData['Min Principal Direction'][minfilt[:-1]][mask[:-1]]
#         mask = MeanCurvature == sys.float_info.max
#         if i == 100:
#             warnings.warn('Iteration limit reached for mean curvature calculation.\nThis could indicate unexpected values or a significantly larger than expected grid size.')
#             break
    
#     Grid.NodeData['Mean Curvature'] = MeanCurvature[:-1]
#     if runscaf:
#         return Grid, Surf, Scaf
#     return Grid, Surf


# def properties(agent_grid):
#     WaterProps = mechanobiology.prendergast_mat(np.repeat(0, agent_grid.NElem))
#     NeoProps = mechanobiology.prendergast_mat(np.repeat(1, agent_grid.NElem))
#     FibProps = mechanobiology.prendergast_mat(np.repeat(2, agent_grid.NElem))
#     CartProps = mechanobiology.prendergast_mat(np.repeat(3, agent_grid.NElem))
#     IboneProps = mechanobiology.prendergast_mat(np.repeat(5, agent_grid.NElem))     # Immature bone
#     LboneProps = mechanobiology.prendergast_mat(np.repeat(7, agent_grid.NElem))     # Lamellar bone
#     ScaffProps = mechanobiology.prendergast_mat(np.repeat(8, agent_grid.NElem))
    
#     def modulus_mineral(mineral_density): 
#         modulus = np.zeros_like(mineral_density)
#         modulus[mineral_density < 0.4] = 32.5*mineral_density[mineral_density < 0.4]**3 +10.5*mineral_density[mineral_density < 0.4]**2 + mineral_density[mineral_density < 0.4] + 0.01
#         # modulus[mineral_density < 0.4] = 63.4375*mineral_density[mineral_density < 0.4]**3 - 8.0625*mineral_density[mineral_density < 0.4]**2 + mineral_density[mineral_density < 0.4] + 1
#         modulus[mineral_density >= 0.4] = 25 * mineral_density[mineral_density >= 0.4] - 5.83
        
#         # Above are in GPa
#         modulus *= 1000 # now in MPa
        
#         return modulus
    
#     # Calculate properties of mineralized tissue
#     mineral_density = np.divide(agent_grid.ElemData['Mineral Density'], agent_grid.ElemData['Osseous Fraction'], where=agent_grid.ElemData['Osseous Fraction']>0, out=np.zeros(agent_grid.NElem))
#     max_modulus = modulus_mineral(np.array([0.8])) # max mineralization .8 mg/mm^3
#     min_modulus = modulus_mineral(np.array([0.0]))
#     modulus = modulus_mineral(mineral_density)
    
#     ratio = (modulus - min_modulus)/(max_modulus - min_modulus)
    
#     MineralizedProps = []
#     for i in range(len(LboneProps)):
#         prop = ratio * (LboneProps[i] - IboneProps[i]) + IboneProps[i]
#         MineralizedProps.append(prop)
    
    
#     # Set volume fraction weighted properties
#     E = (NeoProps[0] * agent_grid.ElemData['ECM Fraction'] + \
#         FibProps[0] * agent_grid.ElemData['Fibrous Fraction'] + \
#         CartProps[0] * agent_grid.ElemData['Cartilaginous Fraction'] + \
#         # dividing mineral density by osseous fraction to get modulus of mineral/osteoid volume, then multiplying by volume fraction
#         modulus * agent_grid.ElemData['Osseous Fraction'] + \
#         ScaffProps[0] * agent_grid.ElemData['Scaffold Fraction'] + \
#         WaterProps[0] * (1-agent_grid.ElemData['Volume Fraction']))
    
#     nu = (NeoProps[1] * agent_grid.ElemData['ECM Fraction'] + \
#         FibProps[1] * agent_grid.ElemData['Fibrous Fraction'] + \
#         CartProps[1] * agent_grid.ElemData['Cartilaginous Fraction'] + \
#         # IboneProps[1] * agent_grid.ElemData['Osseous Fraction'] + \
#         MineralizedProps[1] * agent_grid.ElemData['Osseous Fraction'] + \
#         ScaffProps[1] * agent_grid.ElemData['Scaffold Fraction'] + \
#         WaterProps[1] * (1-agent_grid.ElemData['Volume Fraction']))
    
#     K = (NeoProps[2] * agent_grid.ElemData['ECM Fraction'] + \
#         FibProps[2] * agent_grid.ElemData['Fibrous Fraction'] + \
#         CartProps[2] * agent_grid.ElemData['Cartilaginous Fraction'] + \
#         # IboneProps[2] * agent_grid.ElemData['Osseous Fraction'] + \
#         MineralizedProps[2] * agent_grid.ElemData['Osseous Fraction'] + \
#         ScaffProps[2] * agent_grid.ElemData['Scaffold Fraction'] + \
#         WaterProps[2] * (1-agent_grid.ElemData['Volume Fraction']))
    
#     FluidBulk = (NeoProps[3] * agent_grid.ElemData['ECM Fraction'] + \
#         FibProps[3] * agent_grid.ElemData['Fibrous Fraction'] + \
#         CartProps[3] * agent_grid.ElemData['Cartilaginous Fraction'] + \
#         # IboneProps[3] * agent_grid.ElemData['Osseous Fraction'] + \
#         MineralizedProps[3] * agent_grid.ElemData['Osseous Fraction'] + \
#         ScaffProps[3] * agent_grid.ElemData['Scaffold Fraction'] + \
#         WaterProps[3] * (1-agent_grid.ElemData['Volume Fraction']))
    
#     FluidSpecWeight = (NeoProps[4] * agent_grid.ElemData['ECM Fraction'] + \
#         FibProps[4] * agent_grid.ElemData['Fibrous Fraction'] + \
#         CartProps[4] * agent_grid.ElemData['Cartilaginous Fraction'] + \
#         # IboneProps[4] * agent_grid.ElemData['Osseous Fraction'] + \
#         MineralizedProps[4] * agent_grid.ElemData['Osseous Fraction'] + \
#         ScaffProps[4] * agent_grid.ElemData['Scaffold Fraction'] + \
#         WaterProps[4] * (1-agent_grid.ElemData['Volume Fraction']))
    
#     Porosity = (NeoProps[5] * agent_grid.ElemData['ECM Fraction'] + \
#         FibProps[5] * agent_grid.ElemData['Fibrous Fraction'] + \
#         CartProps[5] * agent_grid.ElemData['Cartilaginous Fraction'] + \
#         # IboneProps[5] * agent_grid.ElemData['Osseous Fraction'] + \
#         MineralizedProps[5] * agent_grid.ElemData['Osseous Fraction'] + \
#         ScaffProps[5] * agent_grid.ElemData['Scaffold Fraction'] + \
#         WaterProps[5] * (1-agent_grid.ElemData['Volume Fraction']))
    
#     SolidBulk = (NeoProps[6] * agent_grid.ElemData['ECM Fraction'] + \
#         FibProps[6] * agent_grid.ElemData['Fibrous Fraction'] + \
#         CartProps[6] * agent_grid.ElemData['Cartilaginous Fraction'] + \
#         # IboneProps[6] * agent_grid.ElemData['Osseous Fraction'] + \
#         MineralizedProps[6] * agent_grid.ElemData['Osseous Fraction'] + \
#         ScaffProps[6] * agent_grid.ElemData['Scaffold Fraction'] + \
#         WaterProps[6] * (1-agent_grid.ElemData['Volume Fraction']))

#     return E, nu, K, FluidBulk, FluidSpecWeight, Porosity, SolidBulk

