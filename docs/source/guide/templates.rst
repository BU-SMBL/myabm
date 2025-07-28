
Function Templates
==================

MyABM was designed to be modular so that different models or model-variants 
could be defined within a common framework. To that end, there several places 
where custom functions can be defined and plugged into a model. This page
contains templates demonstrating how those functions should be defined.

Action Templates
----------------

Agent Action Template
^^^^^^^^^^^^^^^^^^^^^

Agent actions are actions taken by individual agents. Typically each agent in
the model will take the same action or set of actions, but custom 
:ref:`schedules <Schedule Template>`
could be defined so that different agents take different actions. Agent actions
take two inputs, the agent and the agent grid, allowing for the agent to make
decisions based on its environment. 

Agent actions can be stored as :code:`model.agent_actions = (action1, action2, ...)`

.. note::

    Agent actions must be numba njit (no-python, Just in Time complied) functions.
    This allows actions to be performed over many agents much more efficiently,
    but requires more strict variable typing and only supported functions can
    be called from within.

.. code-block:: python
    :linenos:

    @numba.njit
    def agent_action(agent, grid):
        """
        
        Parameters
        ----------
        agent : myabm.agent
            Agent to perform the action.
        grid : myabm.AgentGrid
            Grid environment that the agent is in.

        Returns
        -------
        None.

        """
        
        ...

Grid Action Template
^^^^^^^^^^^^^^^^^^^^

Grid actions are performed on the whole agent grid, rather than individual 
agents. 

Grid actions can be stored as :code:`model.grid_actions = (action1, action2, ...)`

.. note::
    
    Grid actions must be numba njit (no-python, Just in Time complied) functions.
    This allows actions to be performed much more efficiently, but requires more
    strict variable typing and only supported functions can be called from 
    within.


.. code-block:: python
    :linenos:

    @numba.njit
    def agent_grid_action(agent, grid):
        """

        Parameters
        ----------
        grid : myabm.AgentGrid
            AgentGrid of the model that the action will be performed on
        
        Returns
        -------
            None.    
        """
        
        ...

Model Action Template
^^^^^^^^^^^^^^^^^^^^^

Model actions are similar to grid actions, but more flexible. They can be, but
are not required to be numba JIT compiled, allowing for a wider range of 
functionality, albeit at a potentially increased computational cost. Model
actions have access to and can manipulate everything stored within the model, 
including both the mesh and the agent grid (note that manipulations to the 
mesh should be done carefully, as the agent grid will not automatically be 
made aware of changes).

Model actions can be stored as :code:`model.model_actions = (action1, action2, ...)`

.. code-block:: python
    :linenos:

    def model_action(model):
        """

        Parameters
        ----------
        model : myabm.Model
            Model that the action will be performed on
        
        Returns
        -------
            None.    
        """
        
        ...

Schedule Templates
------------------

Schedules orchestrate the actions of the model, agent grid, and agents. The
default schedule is shown below. Importantly, the schedule is where model 
history can be updated, which is necessary to have in order to use 
:meth:`myabm.Model.export` or :meth:`myabm.Model.animate`. Custom schedules may 
rearrange the order of actions, perform other functions in between actions, 
introduce a sub-stepping loop, or introduce other functionality.

A custom schedule can be used when calling :meth:`myabm.Model.act` or 
:meth:`myabm.Model.iterate`.

.. code-block:: python
    :linenos:

    def default_schedule(model):
        """
        Default scheduler that performs agent actions, grid actions, and model
        actions. Actions must be defined in model.agent_actions, 
        model.grid_actions, and model.model_actions

        Parameters
        ----------
        model : myabm.Model
            Model to perform the schedule on
        """        

        model.agent_grid.run_agents(model.agent_actions)
        model.agent_grid.run_grid(model.grid_actions)
        for f in model.model_actions:
            f(model)

        # update history
        model.history['Agent Nodes'].append(model.agent_nodes)
        model.history['Agent States'].append(model.agent_states)
        model.history['ElemData'].append(copy.deepcopy(model.ElemData))
        model.history['NodeData'].append(copy.deepcopy(model.NodeData))
        model.history['Time'].append(model.history['Time'][-1]+model.agent_grid.TimeStep)
