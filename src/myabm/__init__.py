#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: toj
"""
.. currentmodule:: myabm

Classes
=======
.. autosummary::
    :toctree: generated/

    Model
    AgentGrid
    Agent

Submodules
==========
.. autosummary::
    :toctree: generated/
    
    ortho
    actions
    utils

"""

#%%
import os, copy, itertools, warnings
import numpy as np
import numba
from numba.experimental import jitclass
from numba import int32, int64, float32, float64, literal_unroll
from numba.typed import Dict, List
from numba.types import string, DictType, ListType
import meshio
import pyvista as pv
import mymesh
import scipy
from . import actions, utils
warnings.simplefilter('ignore', numba.core.errors.NumbaExperimentalFeatureWarning)

@jitclass({
    'id' : int64,
    'node' : int64,
    'state' : string,
    'age' : float64,
    'parameters': DictType(string, float64),
    'vector_parameters': DictType(string, float64[:]),
})
class Agent():
    """
    Generic agent class
    """    
    # NOTE: Agent() methods cannot create new Agent() instances directly
    def __init__(self, node=0, state='', parameters=None, id=0):
        self.node = node
        self.state = state
        self.age = 0
        self.id = id
        self.parameters = Dict.empty(key_type=string,value_type=float64)
    
        if parameters is not None:
            for key in parameters:
                self.parameters[key] = parameters[key]

        self.vector_parameters = Dict.empty(key_type=numba.types.string, value_type=float64[:])
    
    def act(self, agent_grid, actions):
        """
        Perform a set of actions. Each agent action should take as input only
        the agent and the AgentGrid

        Parameters
        ----------
        agent_grid : myabm.AgentGrid
            AgentGrid  object that contains the agent
        actions : tuple
            Tuple of agent action functions. For a single action, it should
            still be contained in a tuple (e.g. `(action_function,)`).
        """        
        # NOTE: Currently if migration happens, it will prevent apoptosis in the same step due to the way the agents are managed within the grid
        for action in literal_unroll(actions):
            action(self, agent_grid)
    

@jitclass({
    'action_id' : int64,
    'agent_id' : int64,
    'node' : int64,
    'state' : string,
    'parameters' : DictType(string, float64),
})
class DelayedAgentAction():

    def __init__(self, action_id=-1, agent_id=-1, node=-1, state='', parameters=None):
        self.action_id = action_id
        # 0 : remove
        # 1 : add
        # 2 : move
        # 3: change state
        self.agent_id = agent_id
        self.node = node
        self.state = state
        if parameters is None:
            self.parameters = Dict.empty(key_type=string,value_type=float64)
        else:
            self.parameters = parameters

AgentValue = Agent()._numba_type_
DelayedType = DelayedAgentAction()._numba_type_
@jitclass({
    'NNode' : int64,
    'NElem' : int64,
    'TimeStep' : float64,
    'NodeConn' : DictType(int64, int64[:]),
    'Edges' : int64[:,:],
    'EdgeElemConn' : DictType(int64, int64[:]),
    'NodeEdgeConn' : DictType(int64, int64[:]),
    'ElemConn' : DictType(int64, int64[:]),
    'NodeNeighbors' : DictType(int64, int64[:]),
    'NodeMooreNeighbors' : DictType(int64, int64[:]),
    'parameters' : DictType(string, float64),
    'vector_parameters' : DictType(string, float64[:]),
    'Agents' : DictType(int64, AgentValue),
    'NodeAgents' : DictType(int64, int64),
    '_next_id' : int64,
    'ElemData' : DictType(string, float64[:]),
    'NodeData' : DictType(string, float64[:]),
    'ElemVectorData' : DictType(string, float64[:,:]),
    'NodeVectorData' : DictType(string, float64[:,:]),
    'queue' : ListType(DelayedType),
})
class AgentGrid():
    """
    
    Parameters
    ----------
    NNode : int
        Number of nodes in the mesh
    NElem : int
        Number of elements in the mesh
    TimeStep : float
        Time step size for the model
    NodeConn : dict
        Dictionary containing nodes that define each element
    Edges : np.ndarray(dtype=np.int64)
        Node connectivity of edges
    EdgeElemConn : numba.typed.Dict
        Dictionary containing element IDs connected to an edge
    NodeEdgeConn : numba.typed.Dict
        Dictionary containing edge IDs connected to a node
    ElemConn : numba.typed.Dict
        Dictionary containing element IDs connected to a node
    NodeNeighbors : numba.typed.Dict
        Dictionary containing node ids connected to a node
    NodeMooreNeighbors : numba.typed.Dict
        Dictionary containing node ids in the Moore neighborhood of a node
    parameters : numba.typed.Dict
        Dictionary containing scalar model parameters for the agent grid, keyed 
        by strings.
    parameters : numba.typed.Dict
        Dictionary containing vector model parameters for the agent grid, keyed 
        by strings.
    
    Attributes
    ----------
    Agents : numba.typed.Dict
        Dictionary of agents keyed by agent ID
    NodeAgents : numba.typed.Dict
        Dictionary of agent ids keyed by node
    ElemData : numba.typed.Dict
        Dictionary containing scalar data associated with elements
    NodeData : numba.typed.Dict
        Dictionary containing scalar data associated with nodes
    ElemVectorData : numba.typed.Dict
        Dictionary containing vector data associated with elements
    NodeVectorData : numba.typed.Dict
        Dictionary containing vector data associated with nodes

    

    

    """

    def __init__(self, NNode, NElem, TimeStep, NodeConn, Edges, EdgeElemConn, NodeEdgeConn, ElemConn, NodeNeighbors, NodeMooreNeighbors, parameters=None, vector_parameters=None):

        self.NNode = NNode
        self.NElem = NElem
        self.TimeStep = TimeStep 
        
        # Mesh Connectivity
        self.NodeConn = NodeConn
        self.Edges = Edges
        self.EdgeElemConn = EdgeElemConn
        self.NodeEdgeConn = NodeEdgeConn
        self.ElemConn = ElemConn
        self.NodeNeighbors = NodeNeighbors
        self.NodeMooreNeighbors = NodeMooreNeighbors
        
        # Agents
        self.Agents = Dict.empty(key_type=int64, value_type=AgentValue)
        self.NodeAgents = Dict.empty(key_type=int64, value_type=int64)
        self._next_id = -1
        self.queue = List.empty_list(DelayedType)

        # Data
        self.ElemData = Dict.empty(key_type=numba.types.string, value_type=float64[:])
        self.NodeData = Dict.empty(key_type=numba.types.string, value_type=float64[:])
        self.ElemVectorData = Dict.empty(key_type=numba.types.string, value_type=float64[:,:])
        self.NodeVectorData = Dict.empty(key_type=numba.types.string, value_type=float64[:,:])

        # Parameters
        self.parameters = Dict.empty(key_type=string,value_type=float64)
        if parameters is not None:
            for key in parameters:
                self.parameters[key] = parameters[key]

        self.vector_parameters = Dict.empty(key_type=string,value_type=float64[:])
        if vector_parameters is not None:
            for key in vector_parameters:
                self.vector_parameters[key] = vector_parameters[key]
        
    def seed(self, N, state='', nodes=np.empty(0,dtype=np.int64), method='random', parameters=None):
        """
        Seed the grid with agents

        Parameters
        ----------
        N : int
            Number of agents to be added to the mesh
        state : str, optional
            Agent state applied to the created agents, by default 'none'
        nodes : np.ndarray(dtype=np.int64), optional
            Array of node indices that agents can be seeded onto. If none are
            provided, all nodes are considered seedable, by default 
            np.empty(0,dtype=np.int64). If N >= len(nodes), all seedable nodes 
            will be seeded. If agents have already been seeded onto the mesh,
            only unoccupied agents will be seeded.
        method : str, optional
            Seeding method, by default 'random'.

            Options:

            - 'random' : Agents are randomly seeded on the available nodes.

        parameters : DictType[unicode_type,float64] or NoneType, optional
            Numba Dict of agent parameters, by default None

        """   
        if len(nodes) == 0:
            nodes = np.arange(self.NNode, dtype=np.int64)

        if len(self.NodeAgents) > 0:
            # Mesh already has agents, don't overwrite
            occupied = set(list(self.NodeAgents.keys()))
            nodes = np.array([i for i in nodes if i not in occupied])
        else:
            nodes = np.copy(nodes).astype(np.int64)
        
        if N > len(nodes):
            N = len(nodes)
        if method == 'random':
            np.random.shuffle(nodes)
            for i in range(N):
                node = nodes[i]
                self.add_agent(node, state, parameters)
        else:
            raise Exception('Invalid seeding method.')

    def remove_agent(self, agent, delay=False):
        """
        Remove an agent from the mesh at the specified node

        Parameters
        ----------
        agent : myabm.Agent
            Agent to be removed
        delay : bool
            If True, this action won't occur immediately but will be added to 
            :attr:`myabm.AgentGrid.queue` to be processed later.
        """        
        if agent.node in self.NodeAgents:
            if delay:
                self.queue.append(DelayedAgentAction(0, agent.id, -1, '', None))
            else:
                del self.Agents[self.NodeAgents[agent.node]]
                del self.NodeAgents[agent.node]

    def add_agent(self, node, state='', parameters=None, delay=False):
        """
        Add an agent to the mesh

        Parameters
        ----------
        node : int
            Node ID to place the agent. 
        state : str, optional
            Agent state descriptor, by default ''
        parameters : DictType[unicode_type,float64] or NoneType, optional
            Numba Dict of agent parameters, by default None
        delay : bool
            If True, this action won't occur immediately but will be added to 
            :attr:`myabm.AgentGrid.queue` to be processed later.
        """        
        if delay:
            self.queue.append(DelayedAgentAction(1, -1, node, state, parameters))
        else:
            agent = Agent(node, state, parameters, self.next_id)
            self.Agents[agent.id] = agent
            self.NodeAgents[node] = agent.id

    def move_agent(self, agent, node, delay=False):
        """
        Move an agent from it's current location to a new node.

        Parameters
        ----------
        agent : myabm.Agent
            Agent to be moved
        node : int
            Node ID that the agent will be moved to
        delay : bool
            If True, this action won't occur immediately but will be added to 
            :attr:`myabm.AgentGrid.queue` to be processed later.
        """        
        if delay:
            self.queue.append(DelayedAgentAction(2, agent.id, agent.node, '', None))
        else:
            old_node = agent.node
            del self.NodeAgents[old_node]           # Remove cell from old node
            agent.node = node                       # Update cell's node id
            self.NodeAgents[node] = agent.id        # Move cell to new node

    def change_agent(self, agent, state, delay=False):
        """
        Change the agent's state

        Parameters
        ----------
        agent : myabm.Agent
            Agent that will have its state (:attr:`myabm.Agent.state`) changed
        state : str
            New agent state
        delay : bool
            If True, this action won't occur immediately but will be added to 
            :attr:`myabm.AgentGrid.queue` to be processed later.
        
        """        
        if delay:
            self.queue.append(DelayedAgentAction(3, agent.id, -1, state, None))
        else:
            agent.state = state

    def run_agents(self, actions=()):
        """
        Run a set of agent actions. Each agent action function should take
        as its input only an AgentGrid object. Action functions must also be
        numba.njit compiled functions.

        Parameters
        ----------
        actions : tuple, optional
            Tuple of agent actions to run, by default ()
        """ 
        if len(actions) > 0: 
            keys = list(self.Agents.keys())
            for key in keys:
                if key in self.Agents:
                    # TODO: Need to figure out why sometimes this isn't true - seems to be related to apoptosis, maybe specifically migration followed by apoptosis
                    agent = self.Agents[key]
                    agent.age += self.TimeStep
                    agent.act(self, actions)
            self.run_delayed()
    def run_grid(self, actions=()):
        """
        Run a set of grid actions. Each grid action function should take
        as its input only an AgentGrid object. Action functions must also be
        numba.njit compiled functions.

        Parameters
        ----------
        actions : tuple, optional
            Tuple of grid actions to run, by default ()
        """    
        if len(actions) > 0:    
            for action in literal_unroll(actions):
                action(self)

    def run_delayed(self):
        """
        Run all delayed actions stored in :attr:`myabm.AgentGrid.queue`. 
        Used when :code:`delay=True` is given to 
        :meth:`myabm.AgentGrid.remove_agent`, :meth:`myabm.AgentGrid.add_agent`,
        :meth:`myabm.AgentGrid.move_agent`, or :meth:`myabm.AgentGrid.change_agent`.
        """        
        for delayed in self.queue:
            if delayed.action_id == 0:
                self.remove_agent(self.Agents[delayed.agent_id])
            elif delayed.action_id == 1:
                self.add_agent(delayed.node, delayed.state, delayed.parameters)
            elif delayed.action_id == 2:
                self.move_agent(self.Agents[delayed.agent_id], delayed.node)
            elif delayed.action_id == 3:
                self.change_agent(self.Agents[delayed.agent_id], delayed.state)

        self.queue.clear()

    @property
    def next_id(self):
        self._next_id += 1
        return self._next_id

class Model():
    """
    Model class, containing :class:`~mymesh.mesh`, :class:`AgentGrid`, and :class:`Agent`

    Parameters
    ----------
    Mesh : mymesh.mesh
        Mesh object defining the geometry of the model
    agent_grid : myabm.AgentGrid
        Agent grid of the model
    model_parameters : dict
        Dictionary containing model parameters
    grid_parameters : dict
        Parameters passed to the :class:`myabm.AgentGrid`. 
    agent_parameters : dict
        Parameters passed to :class:`myabm.Agent` instances seeded in the model by
        :meth:`myabm.Model.seed`. Note that this doesn't change the parameters 
        of agents already in the model.

    """
    def __init__(self, Mesh=None, agent_grid=None, 
        model_parameters=None, grid_parameters=None, agent_parameters=None):
        
        self.mesh = Mesh
        self.mesh.verbose=False
        if agent_grid is None and Mesh is not None:
            self.initialize_grid()
        else:
            self.agent_grid = agent_grid
        
        self.agent_actions = ()
        self.grid_actions = ()
        self.model_actions = ()

        self.model_parameters = dict()
        self.grid_parameters = dict()
        self.agent_parameters = dict()

        if model_parameters is not None:
            for key in model_parameters:
                self.model_parameters[key] = model_parameters[key]
        
        if grid_parameters is not None:
            for key in grid_parameters:
                self.grid_parameters[key] = grid_parameters[key]

        if agent_parameters is not None:
            for key in agent_parameters:
                self.agent_parameters[key] = agent_parameters[key]
        
        self.agent_grid.parameters = utils.dict_to_Dict(self.grid_parameters, numba.types.string, numba.types.float64)
        self.history = {}
        self.history['Agent Nodes'] = []
        self.history['Agent States'] = []
        self.history['ElemData'] = []
        self.history['NodeData'] = []
        self.history['Time'] = []

    def initialize_grid(self):
        """
        Initialize the :class:`myabm.AgentGrid` from the mesh
        """        
        self.agent_grid = initialize(self.mesh)

    def seed(self, N=None, state='none', nodes=None, method='random', parameters=None):
        """
        Seed the grid with agents. 
        This passes through arguments to :func:`AgentGrid.seed`, but is more 
        flexible with typing.

        Parameters
        ----------
        N : int, NoneType
            Number of agents to be added to the mesh. If not provided, all
            nodes specified by the `nodes` input will be seeded. If neither
            `N` nor `nodes` are defined, all nodes will be seeded.
        state : str, optional
            Agent state applied to the created agents, by default 'none'
        nodes : array_like, NoneType, optional
            Array of node indices that agents can be seeded onto. If none are
            provided, all nodes are considered seedable, by default None. 
            If N >= len(nodes), all seedable nodes  will be seeded. 
            If agents have already been seeded onto the mesh, only unoccupied 
            agents will be seeded. If N is not provided, all `nodes` will be 
            seeded.
        method : str, optional
            Seeding method, by default 'random'.

            Options:

            - 'random' : Agents are randomly seeded on the available nodes.

        parameters : DictType[unicode_type,float64] or NoneType, optional
            Numba Dict of agent parameters, by default None
            self.agent_parameters will be applied unless overridden by 
            a value in the `parameters` input
        """        
        if nodes is None:
            nodes = np.arange(self.agent_grid.NNode, dtype=np.int64)
        elif isinstance(nodes, (list, tuple, np.ndarray)):
            nodes = np.asarray(nodes, dtype=np.int64)
            
        if N is None:
            N = len(nodes)

        
        # TODO: convert dict to numba Dict
        if parameters is None:
            parameters = dict()
        for key in self.agent_parameters:
            if key not in parameters:
                parameters[key] = self.agent_parameters[key]

        parameters = utils.dict_to_Dict(parameters, numba.types.string, numba.types.float64)

        self.agent_grid.seed(N, state, nodes, method, parameters)
    
    def default_schedule(self):
        """
        Default scheduler that performs agent actions, grid actions, and model
        actions. Actions must be defined in :attr:`myabm.Model.agent_actions`, 
        :attr:`myabm.Model.grid_actions`, and :attr:`myabm.Model.model_actions`

        Parameters
        ----------
        self : myabm.Model
            Model to perform the schedule on
        """        

        self.agent_grid.run_agents(self.agent_actions)
        self.agent_grid.run_grid(self.grid_actions)
        for f in self.model_actions:
            f(self)

        # update history
        self.history['Agent Nodes'].append(self.agent_nodes)
        self.history['Agent States'].append(self.agent_states)
        self.history['ElemData'].append(copy.deepcopy(self.ElemData))
        self.history['NodeData'].append(copy.deepcopy(self.NodeData))
        self.history['Time'].append(self.history['Time'][-1]+self.agent_grid.TimeStep)

    def act(self, schedule=None):
        """
        Execute a :ref:`model action <Model Action Template>` action or a 
        :ref:`schedule <Schedule Templates>`

        Parameters
        ----------
        schedule : callable, optional
            A function directing execution of model actions, by default 
            :meth:`myabm.Model.default_schedule`. A single model action 
            can also be provided.
        """        
        if schedule is None:
            self.default_schedule()
        else:
            schedule(self)

    def iterate(self, n, schedule=None):
        """
        Repeatedly run a schedule for a specified number of steps

        Parameters
        ----------
        n : int
            Number of iterations
        schedule : callable, optional
            A function directing execution of model actions, by default 
            :meth:`myabm.Model.default_schedule`. A single model action 
            can also be provided.
        """        
        # Set initial history
        self.history['Agent Nodes'].append(self.agent_nodes)
        self.history['Agent States'].append(self.agent_states)
        self.history['ElemData'].append(copy.deepcopy(self.ElemData))
        self.history['NodeData'].append(copy.deepcopy(self.NodeData))
        if len(self.history['Time']) == 0:
            self.history['Time'].append(0)
        else:
            self.history['Time'].append(self.history['Time'][-1]+self.agent_grid.TimeStep)
        # Iterate
        for i in range(n):
            self.act(schedule)

    def export_history(self, filename, time=None, agent_lookup=None, noagent_value=-1):
        """
        Export model history to an XDMF file.
        XDMF files can be visualized in ParaView.
        The exported data will contain all node and element data, as well
        as node data field called 'Agent States' that numerically labels agent 
        states. 
        By default, numeric labels will be arbitrarily assigned to each
        unique state that occurs over the course of the history. Alternatively,
        an `agent_lookup` dict can be provided with keys indicating agent
        states and values assigning numbers to each state, for example:
        `agent_lookup = {'state 1': 1, 'state 2': 2}`. By default, a value of
        -1 will be assigned to nodes with no agents. This can be changed by 
        setting `noagent_value`.


        Parameters
        ----------
        filename : str
            Filename for the exported file
        time : array_like, NoneType, optional
            Array of time steps that correspond to the model history, by default 
            None. If not provided,
            time = np.arange(len(self.history['ElemData']))*self.agent_grid.TimeStep
        agent_lookup : dict, optional
            Dictionary to map agent states to numeric values, by default None.
            If none is provided, one will be created, but it may not be possible
            to determine which state is assigned which value.
            Example: `agent_lookup = {'state 1': 1, 'state 2': 2}`
        noagent_value : int, float, optional
            Value to assign to nodes that don't contain an agent, by default -1.
        """
        m = self.mesh.copy()
        M = m.mymesh2meshio()

        if agent_lookup is None:
            # Create a lookup table to map agent states to numeric
            agent_lookup = dict()
            states = set()
            for agent_states in self.history['Agent States']:
                states.update(agent_states)

            for i,state in enumerate(states):
                agent_lookup[state] = i

        if time is None:
            dt = self.agent_grid.TimeStep
            time = np.arange(len(self.history['ElemData']))*dt

        with meshio.xdmf.TimeSeriesWriter(filename) as writer:
            writer.write_points_cells(M.points, M.cells)
            for i,t in enumerate(time):
                m.ElemData = dict(self.history['ElemData'][i])
                m.NodeData = dict(self.history['NodeData'][i])

                # Add agent state info into node data
                m.NodeData['Agent State'] = np.repeat(-1, m.NNode)
                if len(self.history['Agent Nodes'][i]) > 0:
                    m.NodeData['Agent State'][self.history['Agent Nodes'][i]] = [agent_lookup[state] for state in self.history['Agent States'][i]]
                meshio_mesh = m.mymesh2meshio()
                writer.write_data(t, point_data=meshio_mesh.point_data, cell_data=meshio_mesh.cell_data)
            
    def export_mesh(self, filename):
        """
        Export the mesh with NodeData and ElemData at the current state.

        Parameters
        ----------
        filename : str
            File name/path to write the mesh file to. Any mesh file types 
            supported by `meshio <https://github.com/nschloe/meshio>`_ are 
            allowed.
        """        
        m = self.mesh.copy()
        for key in self.ElemData:
            if len(self.ElemData[key]) == m.NElem:
                m.ElemData[key] = self.ElemData[key]
        for key in self.NodeData:
            if len(self.NodeData[key]) == m.NNode:
                m.NodeData[key] = self.NodeData[key]    

        m.write(filename)
    
    def animate(self, filename, fps=10, timestep=None, view='isometric', show_mesh=True, show_mesh_edges=True, mesh_kwargs={}, agent_kwargs={}, state_color=None, show_timer=True):
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
        """        

        # Set default color mapping
        if state_color is None:
            default_colors = [
                '#5e81ac',
                '#bf616a',
                '#ebcb8b',
                '#a3be8c',
                '#b48ead',
                '#d08770',
            ]
            states = set()
            for agent_states in self.history['Agent States']:
                states.update(agent_states)
            state_color = {}
            for state, color in zip(states, itertools.cycle(default_colors)):
                    state_color[state] = color

        # Create plotter
        plotter = pv.Plotter(notebook=False, off_screen=True)
        if show_mesh:
            plotter.add_mesh(pv.wrap(self.mesh.mymesh2meshio()), show_edges=show_mesh_edges, color='white', **mesh_kwargs)

        # Add agents
        point_actors = []
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

            point_actors = []
            for state in np.unique(self.history['Agent States'][i]):
                if state_color[state] is not None:
                    nodes = self.history['Agent Nodes'][i][self.history['Agent States'][i] == state]
                    point_actors.append(plotter.add_points(self.mesh.NodeCoords[nodes], color=state_color[state], **agent_kwargs))

            if show_timer and len(self.history['Time'])==len(self.history['Agent States']):
                plotter.remove_actor(text_actor)
                text_actor = plotter.add_text(f"Time = {self.history['Time'][i]:.2f}", position='upper_edge')

            plotter.write_frame()
        plotter.close()

    @property
    def NNode(self):
        """
        Number of nodes in the mesh.
        """        
        return self.mesh.NNode
    @property
    def NElem(self):
        """
        Number of elements in the mesh.
        """  
        return self.mesh.NElem
    @property
    def agents(self):
        """
        Get a list of all agents currently in the :class:`myabm.AgentGrid`.

        Returns
        -------
        agent_list : list
            A list of each agent in the agent_grid.
            
        """
        agent_list = list(self.agent_grid.Agents.values())

        return agent_list
    @property
    def agent_nodes(self):
        """
        Get a list of all nodes that agents are occupying.

        Returns
        -------
        nodes : np.ndarray
            Array of node IDs 
        """        
        nodes = np.array([agent.node for agent in self.agents])

        return nodes
    @property
    def agent_states(self):
        """
        Get a list of all nodes states.

        Returns
        -------
        states : list
            list of agent states
        """        
        states = np.array([agent.state for agent in self.agents])

        return states
    @property
    def ElemData(self):
        """
        Get a read-only dictionary of the element data.
        This pulls data from the agent_grid and merges `agent_grid.ElemData`
        and `agent_grid.ElemVectorData`

        Returns
        -------
        data : dict
            dictionary of element data
        """        
        data = dict(self.agent_grid.ElemData)
        data.update(dict(self.agent_grid.ElemVectorData))

        return data
    @property
    def NodeData(self):
        """
        Get a read-only dictionary of the node data.
        This pulls data from the agent_grid and merges `agent_grid.NodeData`
        and `agent_grid.NodeVectorData`

        Returns
        -------
        data : dict
            dictionary of element data
        """        
        data = dict(self.agent_grid.NodeData)
        data.update(dict(self.agent_grid.NodeVectorData))

        return data

def initialize(Mesh, TimeStep=1):
    """
    Generate an AgentGrid from a mesh

    Parameters
    ----------
    Mesh : mymesh.mesh
        Mesh object the defines the underlying structure and connectivity
        of the agent mesh

    Returns
    -------
    agent_grid : myabm.AgentGrid
        AgentGrid object initialized from the mesh
    """    
    # NodeEdgeConn = {i:[] for i in range(Mesh.NNode)}
    # h = np.linalg.norm(np.diff(Mesh.NodeCoords[Mesh.Edges[0]],axis=0))
    # for i,edge in enumerate(Mesh.Edges):
    #     NodeEdgeConn[edge[0]].append(i)
    #     NodeEdgeConn[edge[1]].append(i)

    # NodeEdgeConn = mymesh.utils.PadRagged(list(NodeEdgeConnDict.values()))
    # try:
    #     Mesh.NodeConn = np.asarray(Mesh.NodeConn, dtype=np.int64)
    # except:
    #     raise NotImplementedError('Mixed-element meshes not yet supported.')
    NodeConnDict = Dict.empty(key_type=int64, value_type=int64[:])

    ElemConnDict = Dict.empty(key_type=int64, value_type=int64[:])
    NodeNeighborDict = Dict.empty(key_type=int64, value_type=int64[:])
    MooreNeighborDict = Dict.empty(key_type=int64, value_type=int64[:])

    EdgeElemConnDict = Dict.empty(key_type=int64, value_type=int64[:])
    NodeEdgeConnDict = Dict.empty(key_type=int64, value_type=int64[:])


    for i in range(Mesh.NElem):
        NodeConnDict[i] = np.asarray(Mesh.NodeConn[i], dtype=np.int64)

    for i in range(Mesh.NNode):
        ElemConnDict[i] = np.asarray(Mesh.ElemConn[i], dtype=np.int64)
        NodeNeighborDict[i] = np.asarray(Mesh.NodeNeighbors[i], dtype=np.int64)
        MooreNeighborDict[i] = np.setdiff1d(np.unique([Mesh.NodeConn[e] for e in Mesh.ElemConn[i]]).astype(np.int64),i)

    for i in range(Mesh.NEdge):
        EdgeElemConnDict[i] = np.asarray(Mesh.EdgeElemConn[i], dtype=np.int64)
        
        edge = Mesh.Edges[i]
        if edge[0] in NodeEdgeConnDict:
            NodeEdgeConnDict[edge[0]] = np.append(NodeEdgeConnDict[edge[0]], i)
        else:
            NodeEdgeConnDict[edge[0]] = np.array([i], dtype=np.int64)

        if edge[1] in NodeEdgeConnDict:
            NodeEdgeConnDict[edge[1]] = np.append(NodeEdgeConnDict[edge[1]], i)
        else:
            NodeEdgeConnDict[edge[1]] = np.array([i], dtype=np.int64)
    
    agent_grid = AgentGrid(
                        Mesh.NNode, 
                        Mesh.NElem,
                        TimeStep,
                        NodeConnDict,           # NodeConn
                        np.asarray(Mesh.Edges, dtype=np.int64),
                        EdgeElemConnDict,# EdgeElemConn
                        NodeEdgeConnDict,                 # NodeEdgeConn    
                        ElemConnDict,           # ElemConn
                        NodeNeighborDict,      # Node Neighbors 
                        MooreNeighborDict, # Node Moore Neighborhood
                        )

    return agent_grid

# def get_AgentMesh(AgentGrid):

#     nodes = list(AgentGrid.NodeAgents.keys())
#     A = mesh(AgentGrid.NodeCoords[nodes])
#     A.NodeData['state'] = [AgentGrid.NodeAgents[i].state for i in nodes]

#     return A
