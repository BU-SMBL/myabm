#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 15:41:08 2025

@author: toj
"""
import sys, os, warnings
import tqdm
import numpy as np
from mymesh import converter, curvature, image, implicit, improvement, mesh, primitives, utils

sys.path.append('..')
import ABM
from . import actions, setup, geometries, mechanobiology
try:
    import pyAnsysSolvers
    ansys_path='/ad/eng/opt/ansys/v212/ansys/bin/ansys212'
except:
    pass

def growth(Grid, agent_grid, days, timestep=1, substep=0.02,
        actions=(actions.proliferate, actions.migrate_curvotaxis, actions.produce_oriented, actions.apoptose),
        smooth_scaffold=False, func=None, symmetric=False, periodic=False, TissueThreshold=0.2, nrings=2):
    
    AgentMeshes = []
    nodes = list(agent_grid.NodeAgents.keys())
    a = mesh(Grid.NodeCoords[nodes], [[0,0,0]])
    a.NodeData['NodeID'] = nodes
    a.NodeData['state'] = [1 if agent.state=='msc' else 2 if agent.state=='fibroblast' else 3 if agent.state=='chondrocyte' else 5 if agent.state=='osteoblast' else 0 for i,agent in agent_grid.NodeAgents.items()]
    AgentMeshes.append(a)
    
    if 'Tissue Orientation' not in Grid.ElemData:
        Grid.ElemData['Tissue Orientation'] = np.zeros((Grid.NElem, 3), dtype=np.float64)
        agent_grid.ElemVectorData['Tissue Orientation'] = Grid.ElemData['Tissue Orientation'] 
    
        Grid.ElemData['Tissue Orientation Orthogonal'] = np.zeros((Grid.NElem, 3), dtype=np.float64)
        agent_grid.ElemVectorData['Tissue Orientation Orthogonal'] = Grid.ElemData['Tissue Orientation Orthogonal'] 

    Grid, Surf, Scaf = ABM_setup.CalculateCurvature(Grid, agent_grid.h, False, smooth_scaffold, symmetric=symmetric, periodic=periodic, TissueThreshold=TissueThreshold, nrings=nrings, func=func)

    agent_grid.NodeData['Min Principal Curvature'] = Grid.NodeData['Min Principal Curvature']
    agent_grid.NodeVectorData['Min Principal Direction'] = Grid.NodeData['Min Principal Direction']
    agent_grid.NodeData['Max Principal Curvature'] = Grid.NodeData['Max Principal Curvature']
    agent_grid.NodeVectorData['Max Principal Direction'] = Grid.NodeData['Max Principal Direction']
    agent_grid.NodeData['Mean Curvature'] = Grid.NodeData['Mean Curvature']
    matprop = ABM_setup.properties(agent_grid)
    Grid.ElemData['modulus'] = matprop[0]
    GridMeshes = [Grid.copy()]
    SurfMeshes = [Surf.copy()]
    
    agent_grid.TimeStep = substep
    
    agenttimesteps = int(np.round(timestep/substep))
    progress = tqdm.tqdm(total=days, unit='day')
    for i in np.arange(0,days,timestep):
        
        # iterate ABM
        for j in range(agenttimesteps):
            agent_grid.run_agents(actions)
            progress.n = np.round(progress.n + substep, 4)
            progress.refresh()
            
        nodes = list(agent_grid.NodeAgents.keys())
        a = mesh(Grid.NodeCoords[nodes])
        a.NodeData['NodeID'] = nodes
        a.NodeData['state'] = [1 if agent.state=='msc' else 2 if agent.state=='fibroblast' else 3 if agent.state=='chondrocyte' else 5 if agent.state=='osteoblast' else 0 for i,agent in agent_grid.NodeAgents.items()]
        AgentMeshes.append(a)
        
        Grid.ElemData['Tissue Orientation'] = agent_grid.ElemVectorData['Tissue Orientation']
        Grid.ElemData['Tissue Orientation Orthogonal'] = agent_grid.ElemVectorData['Tissue Orientation Orthogonal']
        Grid.ElemData['Volume Fraction'] = agent_grid.ElemData['Volume Fraction']
        Grid.ElemData['Scaffold Fraction'] = agent_grid.ElemData['Scaffold Fraction']
        Grid.ElemData['ECM Fraction'] = agent_grid.ElemData['ECM Fraction']
        Grid.ElemData['Fibrous Fraction'] = agent_grid.ElemData['Fibrous Fraction']
        Grid.ElemData['Cartilaginous Fraction'] = agent_grid.ElemData['Cartilaginous Fraction']
        Grid.ElemData['Osseous Fraction'] = agent_grid.ElemData['Osseous Fraction']
        matprop = ABM_setup.properties(agent_grid)
        Grid.ElemData['modulus'] = matprop[0]
        #
        
        agent_grid.NodeData['Min Principal Curvature'] = Grid.NodeData['Min Principal Curvature']
        agent_grid.NodeVectorData['Min Principal Direction'] = Grid.NodeData['Min Principal Direction']
        agent_grid.NodeData['Max Principal Curvature'] = Grid.NodeData['Max Principal Curvature']
        agent_grid.NodeVectorData['Max Principal Direction'] = Grid.NodeData['Max Principal Direction']
        
        Grid, Surf = ABM_setup.CalculateCurvature(Grid, agent_grid.h, normalize=True, smooth_scaffold=smooth_scaffold, symmetric=symmetric, periodic=periodic, nrings=nrings, Scaf=Scaf)
        
        agent_grid.NodeData['Mean Curvature'] = Grid.NodeData['Mean Curvature']
        SurfMeshes.append(Surf)
    
        # Update material properties, rerun mechanics
        # matprop = properties(agent_grid)
        # Grid.ElemData['modulus'] = matprop[0]
        
        # apdl = pyAnsysSolvers.Launch(ansys_path, nproc=nproc, smp=True, port=50052)
        # Grid = pyAnsysSolvers.PoroelasticUniaxial(Grid, *matprop, time_final, disp, axis=2, boundary_eps=0, SurfacePressure=0, nsubsteps=1, constrained=False, mode='disp', mapdl=apdl, nproc=16)
        
        GridMeshes.append(Grid.copy())
    return GridMeshes, SurfMeshes, AgentMeshes


def growthdiff(Grid, agent_grid, days, disp, timestep=1, substep=0.02,
        actions=(actions.proliferate, actions.migrate_curvotaxis, actions.differentiate_prendergast, actions.produce_oriented, actions.apoptose),
        smooth_scaffold=False,  nproc=2, symmetric=False, periodic=False, port=50052, TissueThreshold=0.2, nrings=2):
    
    # Initialize agent meshes
    AgentMeshes = []
    nodes = list(agent_grid.NodeAgents.keys())
    a = mesh(Grid.NodeCoords[nodes], [[0,0,0]])
    a.NodeData['NodeID'] = nodes
    a.NodeData['state'] = [1 if agent.state=='msc' else 2 if agent.state=='fibroblast' else 3 if agent.state=='chondrocyte' else 5 if agent.state=='osteoblast' else 0 for i,agent in agent_grid.NodeAgents.items()]
    AgentMeshes.append(a)
    
    # Initialize tissue orientation
    if 'Tissue Orientation' not in Grid.ElemData:
        Grid.ElemData['Tissue Orientation'] = np.zeros((Grid.NElem, 3), dtype=np.float64)
        agent_grid.ElemVectorData['Tissue Orientation'] = Grid.ElemData['Tissue Orientation'] 
    
        Grid.ElemData['Tissue Orientation Orthogonal'] = np.zeros((Grid.NElem, 3), dtype=np.float64)
        agent_grid.ElemVectorData['Tissue Orientation Orthogonal'] = Grid.ElemData['Tissue Orientation Orthogonal'] 
    
    # Initialize curvature
    Grid, Surf, Scaf = ABM_setup.CalculateCurvature(Grid, agent_grid.h, False, smooth_scaffold, symmetric=symmetric, periodic=periodic, nrings=nrings,)

    agent_grid.NodeData['Min Principal Curvature'] = Grid.NodeData['Min Principal Curvature']
    agent_grid.NodeVectorData['Min Principal Direction'] = Grid.NodeData['Min Principal Direction']
    agent_grid.NodeData['Max Principal Curvature'] = Grid.NodeData['Max Principal Curvature']
    agent_grid.NodeVectorData['Max Principal Direction'] = Grid.NodeData['Max Principal Direction']
    agent_grid.NodeData['Mean Curvature'] = Grid.NodeData['Mean Curvature']
    agent_grid.NodeData['Stimulus'] = np.zeros(agent_grid.NNode)
    # Initialize Grid/Surf Meshes
    matprop = ABM_setup.properties(agent_grid)
    Grid.ElemData['modulus'] = matprop[0]
    GridMeshes = [Grid.copy()]
    SurfMeshes = [Surf.copy()]
    
    agent_grid.TimeStep = substep
    
    agenttimesteps = int(np.round(timestep/substep))
    progress = tqdm.tqdm(total=days, unit='day')
    
    # Loop
    for i in range(0,days,timestep):
        
        # iterate ABM
        for j in range(agenttimesteps):
            agent_grid.run_agents(actions)
            # Update mineral
            # Using maximum() to prevent artificial eroding of mineral from scaffold, which has mineral but no osteoid
            dmineraldt = np.maximum(669.25 * (agent_grid.ElemData['Osseous Fraction'] - agent_grid.ElemData['Mineral Density']/0.8) * 18.08e-5,0) # need to abstract these parameters
            agent_grid.ElemData['Mineral Density'] += dmineraldt*substep
            # Update progress bar
            progress.n = np.round(progress.n + substep, 4)
            progress.refresh()
            
        nodes = list(agent_grid.NodeAgents.keys())
        a = mesh(Grid.NodeCoords[nodes])
        a.NodeData['NodeID'] = nodes
        a.NodeData['state'] = [1 if agent.state=='msc' else 2 if agent.state=='fibroblast' else 3 if agent.state=='chondrocyte' else 5 if agent.state=='osteoblast' else 0 for i,agent in agent_grid.NodeAgents.items()]
        AgentMeshes.append(a)
        
        # Transfer data to mesh
        Grid.ElemData['Tissue Orientation'] = agent_grid.ElemVectorData['Tissue Orientation']
        Grid.ElemData['Tissue Orientation Orthogonal'] = agent_grid.ElemVectorData['Tissue Orientation Orthogonal']
        Grid.ElemData['Volume Fraction'] = agent_grid.ElemData['Volume Fraction']
        Grid.ElemData['Scaffold Fraction'] = agent_grid.ElemData['Scaffold Fraction']
        Grid.ElemData['ECM Fraction'] = agent_grid.ElemData['ECM Fraction']
        Grid.ElemData['Fibrous Fraction'] = agent_grid.ElemData['Fibrous Fraction']
        Grid.ElemData['Cartilaginous Fraction'] = agent_grid.ElemData['Cartilaginous Fraction']
        Grid.ElemData['Osseous Fraction'] = agent_grid.ElemData['Osseous Fraction']
        Grid.ElemData['Mineral Density'] = agent_grid.ElemData['Mineral Density']
        
        # Update curvature        
        Grid, Surf = ABM_setup.CalculateCurvature(Grid, agent_grid.h, normalize=True, smooth_scaffold=smooth_scaffold, symmetric=symmetric,  periodic=periodic, TissueThreshold=TissueThreshold, nrings=nrings, Scaf=Scaf)
        agent_grid.NodeData['Min Principal Curvature'] = Grid.NodeData['Min Principal Curvature']
        agent_grid.NodeVectorData['Min Principal Direction'] = Grid.NodeData['Min Principal Direction']
        agent_grid.NodeData['Max Principal Curvature'] = Grid.NodeData['Max Principal Curvature']
        agent_grid.NodeVectorData['Max Principal Direction'] = Grid.NodeData['Max Principal Direction']
        agent_grid.NodeData['Mean Curvature'] = Grid.NodeData['Mean Curvature']
        SurfMeshes.append(Surf)
    
        # Update material properties
        matprop = ABM_setup.properties(agent_grid)
        
        Grid.ElemData['modulus'] = matprop[0]
        matprop = [prop[agent_grid.ElemData['Volume Fraction'] > 0.] for prop in matprop] # Filter out properties associated with empty elements (modulus=0)
        
        # Run Mechanics
        # M = Grid.Threshold('Volume Fraction', 0.2, '>=') # Filter out properties associated with empty elements (modulus=0)
        M = Grid.Threshold('Volume Fraction', 0., '>')
        
        if symmetric:
            
            axis = 2; boundary_eps = 0
            surfnodes = M.SurfNodes
             
            minx = np.min(M.NodeCoords[surfnodes,0])
            maxx = np.max(M.NodeCoords[surfnodes,0])
            miny = np.min(M.NodeCoords[surfnodes,1])
            maxy = np.max(M.NodeCoords[surfnodes,1])
            minz = np.min(M.NodeCoords[surfnodes,2])
            maxz = np.max(M.NodeCoords[surfnodes,2])
            
            mins = np.min(M.NodeCoords[surfnodes,axis],axis=0)
            maxs = np.max(M.NodeCoords[surfnodes,axis],axis=0)
            BottomNodes = np.array(list(set(np.where(M.NodeCoords[:,axis] <= mins+boundary_eps)[0]).intersection(surfnodes)))
            TopNodes = np.array(list(set(np.where(M.NodeCoords[:,axis] >= maxs-boundary_eps)[0]).intersection(surfnodes)))
        
            TopBCs = np.column_stack([TopNodes,np.repeat(axis,len(TopNodes)),np.repeat(disp,len(TopNodes))])
            BottomBCs = np.column_stack([BottomNodes,np.repeat(axis,len(BottomNodes)),np.repeat(0,len(BottomNodes))])
            otheraxis1, otheraxis2 = {0,1,2}.difference([axis])
            
            TopConstraints = np.empty((0,3))
            pin1 = list(BottomNodes)[0]
            pin2 = np.intersect1d(M.NodeNeighbors[pin1],list(BottomNodes))[0]
            BottomConstraints = np.array([[pin1, otheraxis1, 0],
                                          [pin1, otheraxis2, 0],
                                          [pin2, otheraxis1, 0],
                                          [pin2, otheraxis2, 0]])
            
            yMaxNodes = list(set(np.where(M.NodeCoords[:,1] >= maxy-boundary_eps)[0]).intersection(surfnodes))
            xMaxNodes = list(set(np.where(M.NodeCoords[:,0] >= maxx-boundary_eps)[0]).intersection(surfnodes))
            SymmetryConsraints = np.vstack([
                np.column_stack([xMaxNodes, np.zeros(len(xMaxNodes)), np.zeros(len(xMaxNodes))]),
                np.column_stack([yMaxNodes, np.ones(len(yMaxNodes)), np.zeros(len(yMaxNodes))])    
                ])
            
            Forces = np.empty((0,3))
            Disps = np.vstack([TopBCs, BottomBCs, TopConstraints, BottomConstraints])
                
            Pressures = np.column_stack([surfnodes, np.repeat(0,len(surfnodes))])
            
            try:
                apdl = pyAnsysSolvers.Launch(ansys_path, nproc=nproc, smp=True, port=port)
                M = pyAnsysSolvers.Poroelastic(M, *matprop, timestep, Disps=Disps, Forces=Forces, Pressures=Pressures, mapdl=apdl)
            except:
                apdl = pyAnsysSolvers.Launch(ansys_path, nproc=nproc, smp=True, port=port)
                M = pyAnsysSolvers.Poroelastic(M, *matprop, timestep, Disps=Disps, Forces=Forces, Pressures=Pressures, mapdl=apdl)
        else:
            
            try:
                apdl = pyAnsysSolvers.Launch(ansys_path, nproc=nproc, smp=True, port=port)
                M = pyAnsysSolvers.PoroelasticUniaxial(M, *matprop, timestep, disp, axis=2, boundary_eps=0, SurfacePressure=0, nsubsteps=1, constrained=False, mode='disp', mapdl=apdl)
            except:
                apdl = pyAnsysSolvers.Launch(ansys_path, nproc=nproc, smp=True, port=port)
                M = pyAnsysSolvers.PoroelasticUniaxial(M, *matprop, timestep, disp, axis=2, boundary_eps=0, SurfacePressure=0, nsubsteps=1, constrained=False, mode='disp', mapdl=apdl)
        # Process results
        Grid.NodeData['OctahedralShearStrain'] = M.NodeData['OctahedralShearStrain']
        Grid.NodeData['V'] = M.NodeData['V']
        Grid.NodeData['RF'] = M.NodeData['RF']
        Grid.NodeData['U'] = M.NodeData['U']
        shear = Grid.NodeData['OctahedralShearStrain']
        flow = np.linalg.norm(Grid.NodeData['V'],axis=1)
        agent_grid.NodeData['Stimulus'] = (shear/.0375 + flow*1000/3).astype(np.float64)
        Grid.NodeData['Stimulus'] = (shear/.0375 + flow*1000/3).astype(np.float64)
        
        GridMeshes.append(Grid.copy())
    return GridMeshes, SurfMeshes, AgentMeshes


def export(name, GridMeshes=None, AgentMeshes=None,SurfMeshes=None, path=None):
    
    if path is None:
        path = os.getcwd()
    if not os.path.exists(os.path.join(path,f'{name:s}')):
        os.mkdir(os.path.join(path,f'{name:s}'))
    
    if GridMeshes is not None:
        for i,M in enumerate(GridMeshes):
            M.write(os.path.join(path,f'{name:s}/grid_{i:03d}.vtu'))    
        
        with open(os.path.join(path,f'{name:s}/grid.pvd'),'w') as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian" compressor="vtkZLibDataCompressor">\n')
            f.write('<Collection>\n')
            for i in range(len(GridMeshes)):
                f.write(f'<DataSet timestep="{i:d}" group="" part="0" file="{path:s}/{name:s}/grid_{i:03d}.vtu"/>\n')
            f.write('</Collection>\n')
            f.write('</VTKFile>')
            
    if SurfMeshes is not None:
        for i,M in enumerate(SurfMeshes):
            M.write(os.path.join(path,f'{name:s}/surf_{i:03d}.vtu'))
        
        if SurfMeshes is not None:
            with open(os.path.join(path,f'{name:s}/surf.pvd'),'w') as f:
                f.write('<?xml version="1.0"?>\n')
                f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian" compressor="vtkZLibDataCompressor">\n')
                f.write('<Collection>\n')
                for i in range(len(SurfMeshes)):
                    f.write(f'<DataSet timestep="{i:d}" group="" part="0" file="{path:s}/{name:s}/surf_{i:03d}.vtu"/>\n')
                f.write('</Collection>\n')
                f.write('</VTKFile>')
    
    if AgentMeshes is not None:
        if isinstance(AgentMeshes[0], (tuple, list, np.ndarray)):
            # substep
            with open(os.path.join(path,f'{name:s}/agent.pvd'),'w') as f:
                f.write('<?xml version="1.0"?>\n')
                f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian" compressor="vtkZLibDataCompressor">\n')
                f.write('<Collection>\n')
                
                timesteps = []
                for i in range(len(AgentMeshes)):
                    SubMeshes = AgentMeshes[i]
                    nsub = len(SubMeshes)
                    dt = 1/nsub
                    for j,M in enumerate(SubMeshes):
                        if len(timesteps) == 0:
                            timesteps.append(0)
                        else:
                            timesteps.append(np.round((timesteps[-1]+dt)*1000)/1000)
                        M.write(os.path.join(path,f'{name:s}/agent_{timesteps[-1]:03.4f}.vtu'))
                        f.write(f'<DataSet timestep="{timesteps[-1]:f}" group="" part="0" file="{path:s}/{name:s}/agent_{timesteps[-1]:03.4f}.vtu"/>\n')
            
                    
                f.write('</Collection>\n')
                f.write('</VTKFile>')
        else:
            for i,M in enumerate(AgentMeshes):
                M.write(os.path.join(path,f'{name:s}/agent_{i:03d}.vtu'))
    
            with open(os.path.join(path,f'{name:s}/agent.pvd'),'w') as f:
                f.write('<?xml version="1.0"?>\n')
                f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian" compressor="vtkZLibDataCompressor">\n')
                f.write('<Collection>\n')
                for i in range(len(AgentMeshes)):
                    f.write(f'<DataSet timestep="{i:d}" group="" part="0" file="{path:s}/{name:s}/agent_{i:03d}.vtu"/>\n')
                f.write('</Collection>\n')
                f.write('</VTKFile>')   