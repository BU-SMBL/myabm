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

import sys, os, copy, itertools, warnings
import numpy as np
from .. import Model, AgentGrid, Agent
from . import actions
import numba
from numba.experimental import jitclass
from numba import int32, int64, float32, float64, literal_unroll
from numba.typed import Dict
from numba.types import string, DictType
from mymesh import *
import matplotlib
import pyvista as pv
warnings.simplefilter('ignore', numba.core.errors.NumbaExperimentalFeatureWarning)

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
        self.agent_parameters['ProlifRate'] = 0.6
        self.agent_parameters['ApopRate'] = 0.05
        self.agent_parameters['DiffRate'] = 0.05
        self.agent_parameters['DiffMaturity'] = 6 # days
        self.agent_parameters['MigrRate'] = 1 # mm/day
        self.agent_parameters['MigrationWeight0'] = 0.101
        self.agent_parameters['MigrationWeight1'] = 1.083
        self.agent_parameters['MigrationWeight2'] = 0.475
        self.agent_parameters['MigrationWeight3'] = 0.23239629193097144
        self.agent_parameters['MigrationWeight4'] = -19.97456247818576
        if agent_parameters is not None:
            for key in agent_parameters:
                self.agent_parameters[key] = agent_parameters[key]

        if Mesh is not None:
            element_volumes = quality.Volume(Mesh.NodeCoords, Mesh.NodeConn)
            if not np.all(np.isclose(element_volumes, element_volumes[0])):
                warnings.warn('Mesh has elements of unequal volumes. OrthoModel assumes a uniform grid.')
                self.grid_parameters['Volume'] = np.mean(element_volumes)
            else:
                self.grid_parameters['Volume'] = element_volumes[0]

            edge_lengths = np.linalg.norm(np.diff(Mesh.NodeCoords[Mesh.Edges],axis=1), axis=2)[:,0]
            if not np.all(np.isclose(edge_lengths, edge_lengths[0])):
                warnings.warn('Mesh has edges of unequal lengths. OrthoModel assumes a uniform grid.')
                self.grid_parameters['h'] = np.mean(edge_lengths)
            else:
                self.grid_parameters['h'] = edge_lengths[0]

        if model_parameters is not None:
            for key in model_parameters:
                self.model_parameters[key] = model_parameters[key]

        if 'TimeStep' not in self.model_parameters:
            self.model_parameters['TimeStep'] = 1     # days
        if 'SubStep' not in self.model_parameters:
            self.model_parameters['SubStep'] = 0.02   # days
        if 'Smooth Scaffold' not in self.model_parameters:
            self.model_parameters['Smooth Scaffold'] = True
        if 'Periodic' not in self.model_parameters:
            self.model_parameters['Periodic'] = False
        if 'Symmetric' not in self.model_parameters:
            self.model_parameters['Symmetric'] = False
        
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

        if 'Cells Allowed' not in self.agent_grid.NodeData:
            self.agent_grid.NodeData['Cells Allowed'] = np.ones(self.agent_grid.NNode)

    def default_schedule(self):
        
        self.agent_grid.TimeStep = self.model_parameters['SubStep'] # ensure time steps are synchronized
        nsubstep = int(np.round(self.model_parameters['TimeStep']/self.model_parameters['SubStep']))
        
        for substep in range(nsubstep):
            self.agent_grid.run_agents(self.agent_actions)
            self.agent_grid.run_grid((actions.update_mineral,))
            
        # Run grid and model actions after the last substep
        self.agent_grid.run_grid(self.grid_actions)
        for f in self.model_actions:
            f(self)
        # update history
        self.history['Agent Nodes'].append(self.agent_nodes)
        self.history['Agent States'].append(self.agent_states)
        self.history['ElemData'].append(copy.deepcopy(self.ElemData))
        self.history['NodeData'].append(copy.deepcopy(self.NodeData))
        self.history['Time'].append(self.history['Time'][-1] + self.model_parameters['TimeStep'])
    
    def substep_saver_schedule(self):
        
        self.agent_grid.TimeStep = self.model_parameters['SubStep'] # ensure time steps are synchronized
        nsubstep = int(np.round(self.model_parameters['TimeStep']/self.model_parameters['SubStep']))
        
        for substep in range(nsubstep):
            self.agent_grid.run_agents(self.agent_actions)
            self.agent_grid.run_grid((actions.update_mineral,))
            
            if substep == nsubstep-1:
                # Run grid and model actions after the last substep
                self.agent_grid.run_grid(self.grid_actions)
                for f in self.model_actions:
                    f(self)
            # update history
            self.history['Agent Nodes'].append(self.agent_nodes)
            self.history['Agent States'].append(self.agent_states)
            self.history['ElemData'].append(copy.deepcopy(self.ElemData))
            self.history['NodeData'].append(copy.deepcopy(self.NodeData))
            self.history['Time'].append(self.history['Time'][-1] + self.model_parameters['SubStep'])
        
    def animate(self, filename, fps=10, timestep=None, view='isometric', show_agents=True, show_mesh=True, show_mesh_edges=True, mesh_kwargs={}, agent_kwargs={}, state_color=None, show_timer=True, tissue_threshold=True, tissue_opacity=False, scale_colors=False, scalars=None, clim=None, cmap=None):
        """
        Create an animated gif of the model.
        Model should have already been run with agent history stored in 
        model.history['Agent Nodes'] and model.history['Agent States'].

        .. note::

            While this method provides a reasonable amount of flexibility,
            some models may require custom visualization. This method uses
            pyvista to visualize the model; the source of this method can
            be used as a starting point for custom visualization methods.

        Parameters
        ----------
        filename : str
            Name of the file to be written. If the name does not contain a 
            .gif extension, one will be added.
        fps : int
            Frames per second for the animation
        timestep : float, NoneType
            Time increment at which frames will be rendered in the animation.
            This must be a value that lines up with the time data in 
            :code:`model.history['Time']`. Frames will be rendered if 
            :code:`np.isclose(model.history['Time'] % timestep, 0)`.
        view : str, NoneType, array_like, optional
            Specify how to view the model, by default isometric.
            If given as a string, options are:

                - 'isometric'
                - 'xy'
                - 'xz'
                - 'yx'
                - 'yz'
                - 'zx'
                - 'zy'

            If given as an three element list, tuple, or np.ndarray, this
            will be passed to :func:`pyvista.Plotter.view_vector`

        show_mesh : bool, optional
            Option to display the model's mesh, by default True
        show_mesh_edges : bool, optional
            Option to display the model mesh's edges, by default True
        mesh_kwargs : dict, optional
            Optional keyword arguments passed to :func:`pyvista.Plotter.add_mesh`
            when plotting the mesh, by default {}.
        agent_kwargs : dict, optional
            Optional keyword arguments passed to :func:`pyvista.Plotter.add_points` 
            when plotting the agents, by default {}.
        state_color : dict, NoneType, optional
            Dictionary to map agent states to colors, by default None.
            If None, default colors will be assigned - only 6 colors are in
            the default color scheme, so if there are more than 6 agent states,
            colors will be repeated. To hide a particular state, assign it 
            `None`.

            Example:
                :code:`state_color = {'state 1' : 'blue', 'state 2' : 'red'}`
        show_timer : bool
            If True, attach a timer to display the current model time in each 
            frame, by default True.
        tissue_threshold : bool, float
            If True, only elements with volume 
            fraction > self.agent_grid.parameters['Tissue Threshold']. If a 
            float, the float will be used as the threshold for visualization, by
            default, True.
        tissue_opacity : bool
            If True, element opacity will be scaled by volume fraction, by 
            default, False.

        """        

        neotissue = np.array([141/255, 211/255, 199/255, 1])# np.array([255/255, 176/255, 0/255, 1])  # 1
        fibrous = np.array([254/255, 97/255, 0/255, 1])     # 2 
        cartilage = np.array([220/255, 38/255, 127/255, 1]) # 3
        bone = np.array([100/255, 143/255, 255/255, 1])     # 5
        scaffold = np.array([238/255, 238/255, 238/255, 1]) # 7
        understim = np.array([102/255, 102/255, 102/255, 1])# 8

        if type(tissue_threshold) is bool:
            if tissue_threshold:
                tissue_threshold = self.agent_grid.parameters['Tissue Threshold']

        # Set default color mapping
        if state_color is None:
            state_color = dict(
                msc = neotissue,
                fibroblast = fibrous,
                chondrocyte = cartilage,
                osteoblast = bone,
            )
        
        neotissue_soft = (neotissue + 2*np.array([1, 1, 1, 1]))/3
        neotissue_stiff = (neotissue + np.array([0, 0, 0, 1]))/2
        neotissue_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('neotissue', [neotissue_soft, neotissue_stiff], N=5)
        
        fibrous_soft = (fibrous + 2*np.array([1, 1, 1, 1]))/3
        fibrous_stiff = (fibrous + np.array([0, 0, 0, 1]))/2
        fibrous_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('fibrous', [fibrous_soft, fibrous_stiff], N=5)
        
        cartilage_soft = (cartilage + 2*np.array([1, 1, 1, 1]))/3
        cartilage_stiff = (cartilage + np.array([0, 0, 0, 1]))/2
        cartilage_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('cartilage', [cartilage_soft, cartilage_stiff], N=5)
        
        bone_soft = (bone + 2*np.array([1, 1, 1, 1]))/3
        bone_stiff = (bone + np.array([0, 0, 0, 1]))/2
        bone_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('bone', [bone_soft, bone_stiff], N=5)
        

        # Create plotter
        plotter = pv.Plotter(notebook=False, off_screen=True)
        mesh_actors = []
        if show_mesh:
            if scalars is None:
                ids = np.argmax(np.column_stack((np.zeros(self.NElem),self.history['ElemData'][0]['Fibrous Fraction'], self.history['ElemData'][0]['Cartilaginous Fraction'], self.history['ElemData'][0]['Osseous Fraction'], self.history['ElemData'][0]['Scaffold Fraction'])), axis=1)
                if tissue_threshold is not False:
                    ids[self.history['ElemData'][0]['Volume Fraction'] < tissue_threshold] = -1
                m = self.mesh.copy()
                if 'Smoothed Surface Coordinates' in self.history['NodeData'][0]:
                    m.NodeCoords = self.history['NodeData'][0]['Smoothed Surface Coordinates']
                Neotissue = m.Threshold(ids, 0, '==')
                Fibrous = m.Threshold(ids, 1, '==')
                Cartilage = m.Threshold(ids, 2, '==')
                Bone = m.Threshold(ids, 3, '==')
                Scaffold = m.Threshold(ids, 4, '==')

                if scale_colors:
                    Neotissue.ElemData['ECM Fraction'] = self.history['ElemData'][0]['ECM Fraction'][ids == 0]
                    Fibrous.ElemData['Fibrous Fraction'] = self.history['ElemData'][0]['Fibrous Fraction'][ids == 1]
                    Cartilage.ElemData['Cartilaginous Fraction'] = self.history['ElemData'][0]['Cartilaginous Fraction'][ids == 2]
                    Bone.ElemData['Osseous Fraction'] = self.history['ElemData'][0]['Osseous Fraction'][ids == 3]
                else:
                    Neotissue.ElemData['ECM Fraction'] = np.repeat(0.4, Neotissue.NElem)
                    Fibrous.ElemData['Fibrous Fraction'] = np.repeat(0.4, Fibrous.NElem)
                    Cartilage.ElemData['Cartilaginous Fraction'] = np.repeat(0.4, Cartilage.NElem)
                    Bone.ElemData['Osseous Fraction'] = np.repeat(0.4, Bone.NElem)

                # Understim = M.Threshold(ids, 8, '==')
                if tissue_opacity:
                    neo_opacity = 1 - Neotissue.ElemData['ECM Fraction']
                    fib_opacity = 1 - Fibrous.ElemData['Fibrous Fraction']
                    cart_opacity = 1 - Cartilage.ElemData['Cartilaginous Fraction']
                    bone_opacity = 1 - Bone.ElemData['Osseous Fraction']
                else:
                    neo_opacity = None
                    fib_opacity = None
                    cart_opacity = None
                    bone_opacity = None

                if Neotissue.NElem > 0 : 
                    mesh_actors.append(plotter.add_mesh(pv.wrap(Neotissue.mymesh2meshio()), show_edges=show_mesh_edges, scalars='ECM Fraction', cmap=neotissue_cmap, clim=(0,1), scalar_bar_args=dict(title='Neotissue'), show_scalar_bar=False, log_scale=False, opacity=neo_opacity))
                if Fibrous.NElem > 0 : 
                    mesh_actors.append(plotter.add_mesh(pv.wrap(Fibrous.mymesh2meshio()), show_edges=show_mesh_edges, scalars='Fibrous Fraction', cmap=fibrous_cmap, clim=(0,1), scalar_bar_args=dict(title='Fibrous'), show_scalar_bar=False, log_scale=False, opacity=fib_opacity))
                if Cartilage.NElem > 0 : 
                    mesh_actors.append(plotter.add_mesh(pv.wrap(Cartilage.mymesh2meshio()), show_edges=show_mesh_edges, scalars='Cartilaginous Fraction', cmap=cartilage_cmap, clim=(0,1), scalar_bar_args=dict(title='Cartilage'), show_scalar_bar=False, log_scale=False, opacity=cart_opacity))
                if Bone.NElem > 0 :
                    mesh_actors.append(plotter.add_mesh(pv.wrap(Bone.mymesh2meshio()),show_edges=show_mesh_edges,  cmap=bone_cmap, scalars='Osseous Fraction', scalar_bar_args=dict(title='Bone'), show_scalar_bar=False, clim=(0,1), log_scale=False, opacity=bone_opacity))
                if Scaffold.NElem > 0 : 
                    mesh_actors.append(plotter.add_mesh(pv.wrap(Scaffold.mymesh2meshio()), show_edges=show_mesh_edges, color=scaffold, opacity=1))
            else:
                m = self.mesh.copy()
                m.NodeData = self.history['NodeData'][0]
                thresh = m.Threshold(self.ElemData['Volume Fraction'], tissue_threshold, '>')
                
                mesh_actors.append(plotter.add_mesh(pv.wrap(thresh.mymesh2meshio()),      show_edges=show_mesh_edges, cmap=cmap, scalars=scalars, show_scalar_bar=True, clim=clim))
        # Add agents
        point_actors = []
        if show_agents:
            for state in np.unique(self.history['Agent States'][0]):
                if state_color[state] is not None:
                    nodes = self.history['Agent Nodes'][0][self.history['Agent States'][0] == state]
                    point_actors.append(plotter.add_points(self.mesh.NodeCoords[nodes], color=state_color[state], **agent_kwargs))
        if show_timer and len(self.history['Time'])==len(self.history['Agent States']):
            text_actor = plotter.add_text(f"Time = {self.history['Time'][0]:.2f}", position='upper_edge')

        # Set view
        if view is not None:
            if type(view) is str:
                if view == 'xy':
                    plotter.view_xy()
                elif view == 'xz':
                    plotter.view_xz()
                elif view == 'yx':
                    plotter.view_yx()
                elif view == 'yz':
                    plotter.view_yz()
                elif view == 'zx':
                    plotter.view_zx()
                elif view == 'zy':
                    plotter.view_zy()
                elif view == 'isometric' or view == 'iso':
                    plotter.view_isometric()
            elif isinstance(view, (list, tuple, np.ndarray)):
                plotter.view_vector(view)
            else:
                warnings.warn('Invalid view option')

        if os.path.splitext(filename)[-1].lower() == '.gif' or '':
            # Open gif
            plotter.open_gif(filename, fps=fps)
        else:
            plotter.open_movie(filename, framerate=fps)

        # Iterate through history
        for i,t in enumerate(self.history['Time']):
            if timestep is not None:
                if not np.isclose(t % timestep, 0):
                    continue
            for actor in point_actors:
                plotter.remove_actor(actor)
            for actor in mesh_actors:
                plotter.remove_actor(actor)
            
            point_actors = []
            mesh_actors = []
            
            ###
            if show_mesh:
                if scalars is None:
                    ids = np.argmax(np.column_stack((np.zeros(self.NElem),self.history['ElemData'][i]['Fibrous Fraction'], self.history['ElemData'][i]['Cartilaginous Fraction'], self.history['ElemData'][i]['Osseous Fraction'], self.history['ElemData'][i]['Scaffold Fraction'])), axis=1)
                    if tissue_threshold is not False:
                        ids[self.history['ElemData'][i]['Volume Fraction'] < tissue_threshold] = -1
                    # ids[self.ElemData['modulus'] > 1000] = 3
                    # ids[self.ElemData['material'] == 8] = 4
                    m = self.mesh.copy()
                    if 'Smoothed Surface Coordinates' in self.history['NodeData'][i]:
                        m.NodeCoords = self.history['NodeData'][i]['Smoothed Surface Coordinates']
                    Neotissue = m.Threshold(ids, 0, '==')
                    Fibrous = m.Threshold(ids, 1, '==')
                    Cartilage = m.Threshold(ids, 2, '==')
                    Bone = m.Threshold(ids, 3, '==')
                    Scaffold = m.Threshold(ids, 4, '==')
                    # Understim = M.Threshold(ids, 8, '==')

                    if scale_colors:
                        Neotissue.ElemData['ECM Fraction'] = self.history['ElemData'][i]['ECM Fraction'][ids == 0]
                        Fibrous.ElemData['Fibrous Fraction'] = self.history['ElemData'][i]['Fibrous Fraction'][ids == 1]
                        Cartilage.ElemData['Cartilaginous Fraction'] = self.history['ElemData'][i]['Cartilaginous Fraction'][ids == 2]
                        Bone.ElemData['Osseous Fraction'] = self.history['ElemData'][i]['Osseous Fraction'][ids == 3]
                    else:
                        Neotissue.ElemData['ECM Fraction'] = np.repeat(0.4, Neotissue.NElem)
                        Fibrous.ElemData['Fibrous Fraction'] = np.repeat(0.4, Fibrous.NElem)
                        Cartilage.ElemData['Cartilaginous Fraction'] = np.repeat(0.4, Cartilage.NElem)
                        Bone.ElemData['Osseous Fraction'] = np.repeat(0.4, Bone.NElem)

                    if tissue_opacity:
                        neo_opacity = Neotissue.ElemData['ECM Fraction']
                        fib_opacity = Fibrous.ElemData['Fibrous Fraction']
                        cart_opacity = Cartilage.ElemData['Cartilaginous Fraction']
                        bone_opacity = Bone.ElemData['Osseous Fraction']
                    else:
                        neo_opacity = None
                        fib_opacity = None
                        cart_opacity = None
                        bone_opacity = None

                    if Neotissue.NElem > 0 : 
                        mesh_actors.append(plotter.add_mesh(pv.wrap(Neotissue.mymesh2meshio()), show_edges=show_mesh_edges, scalars='ECM Fraction', cmap=neotissue_cmap, clim=(0,1), scalar_bar_args=dict(title='Neotissue'), show_scalar_bar=False, log_scale=False, opacity=neo_opacity))
                    if Fibrous.NElem > 0 : 
                        mesh_actors.append(plotter.add_mesh(pv.wrap(Fibrous.mymesh2meshio()), show_edges=show_mesh_edges, scalars='Fibrous Fraction', cmap=fibrous_cmap, clim=(0,1), scalar_bar_args=dict(title='Fibrous'), show_scalar_bar=False, log_scale=False, opacity=fib_opacity))
                    if Cartilage.NElem > 0 : 
                        mesh_actors.append(plotter.add_mesh(pv.wrap(Cartilage.mymesh2meshio()), show_edges=show_mesh_edges, scalars='Cartilaginous Fraction', cmap=cartilage_cmap, clim=(0,1), scalar_bar_args=dict(title='Cartilage'), show_scalar_bar=False, log_scale=False, opacity=cart_opacity))
                    if Bone.NElem > 0 :
                        mesh_actors.append(plotter.add_mesh(pv.wrap(Bone.mymesh2meshio()), show_edges=show_mesh_edges, cmap=bone_cmap, scalars='Osseous Fraction', scalar_bar_args=dict(title='Bone'), show_scalar_bar=False, clim=(0,1), log_scale=False, opacity=bone_opacity))
                    if Scaffold.NElem > 0 : 
                        mesh_actors.append(plotter.add_mesh(pv.wrap(Scaffold.mymesh2meshio()), show_edges=show_mesh_edges, color=scaffold, opacity=1))
                else:
                    m = self.mesh.copy()
                    m.NodeData = self.history['NodeData'][i]
                    thresh = m.Threshold(self.ElemData['Volume Fraction'], tissue_threshold, '>')
                    
                    mesh_actors.append(plotter.add_mesh(pv.wrap(thresh.mymesh2meshio()),      show_edges=show_mesh_edges, cmap=cmap, scalars=scalars, show_scalar_bar=True, clim=clim, lighting=False))
            if show_agents:
                for state in np.unique(self.history['Agent States'][i]):
                    if state_color[state] is not None:
                        nodes = self.history['Agent Nodes'][i][self.history['Agent States'][i] == state]
                        point_actors.append(plotter.add_points(self.mesh.NodeCoords[nodes], color=state_color[state], **agent_kwargs))

            if show_timer:
                plotter.remove_actor(text_actor)
                text_actor = plotter.add_text(f"Time = {self.history['Time'][i]:.2f}", position='upper_edge')

            plotter.write_frame()
        plotter.close()