"""
Schelling's Model of Segregation
================================

Implementation of an agent-based model of segregation by Thomas Schelling 
:cite:`Schelling1971`.
This model was one of the earliest agent-based model, and was initially 
implemented using coins and graph paper. 


Implementation of the Schelling's Segregation Model
---------------------------------------------------
Agent's in this model perform a single action where they migrate based on
the agent state of their neighbors. The decision of whether or not to move is 
based on a single parameter, `'B'`. A second agent parameter `'happy'` is used 
to track whether the agent is happy with its current neighbors

Migration is implemented as two actions: one agent action and one grid action.
The agent action :code:`decide` determines whether the agent is happy or wants to move.
Once all the agents have decided whether they want to move in a particular time
step, the grid action :code:`relocate` finds them all a new location to move to.
Note that in the :ref:`default schedule <Schedule Templates>`, the agent actions
are performed before the grid actions.
"""

# %%
import numpy as np
import numba
import mymesh
import sys
sys.path.append('../src')
from myabm import Model

# %%
@numba.njit
def decide(agent, grid):
    """
    Agent action to decide whether the agent is happy with its neighborhood

    Parameters
    ----------
    agent : myabm.Agent
        Agent to perform the action.
    grid : myabm.AgentGrid
        Grid environment that the agent is in.

    Returns
    -------
    None.

    """

    neighbor_agentstates = [grid.Agents[grid.NodeAgents[i]].state for i in grid.NodeMooreNeighbors[agent.node] if i in grid.NodeAgents]
    if len(neighbor_agentstates) == 0:
        # No neighbors, no need to move
        agent.parameters['happy'] = 1
        return

    same = np.sum(np.array([state == agent.state for state in neighbor_agentstates]))

    B = same/len(neighbor_agentstates)

    if B < agent.parameters['B']:
        agent.parameters['happy'] = 0
    else:
        agent.parameters['happy'] = 1

@numba.njit
def relocate(grid):
    """
    Grid action to relocate all unhappy agents

    Parameters
    ----------
    grid : myabm.AgentGrid
        Grid environment that the agent is in.

    Returns
    -------
    None.

    """

    for agent_id in grid.Agents:
        agent = grid.Agents[agent_id]
        
        if agent.parameters['happy'] == 1:
            continue
        
        else:
            # Search for a new location
            for newnode in np.random.permutation(grid.NNode):
                # Search all nodes
                if newnode in grid.NodeAgents:
                    # Occupied - continue searching
                    continue
                
                # Not occupied - check state of neighbors
                neighbor_agentstates = [grid.Agents[grid.NodeAgents[i]].state for i in grid.NodeMooreNeighbors[newnode] if i in grid.NodeAgents]


                if len(neighbor_agentstates) != 0:
                    same = np.sum(np.array([state == agent.state for state in neighbor_agentstates]))
                    newB = same/len(neighbor_agentstates)
                else:
                    # No neighbors - okay to move to
                    newB = 0

                if newB >= agent.parameters['B']:
                    # move
                    agent.parameters['happy'] = 1
                    grid.move_agent(agent, newnode)

                    break

# %%
# 2D model with two agent types
# -----------------------------
# 

# Create the mesh for the simulation
Mesh = mymesh.primitives.Grid2D([0,1,0,1],0.025)

# Initialize the model and assign actions
model = Model(Mesh, agent_parameters={'B': 0.6, 'happy':0})
model.agent_actions = (decide,)
model.grid_actions = (relocate,)

# Seed the model with agents with two different states
model.seed(600, state='0')
model.seed(600, state='1')

# Run the simulation for 30 steps
model.iterate(20)

# Create an animation
model.animate('segregation2d.gif', view='xy', show_mesh=False)


#%%
# 3D model on a torus
# -------------------
# The model can easily be implemented on different meshes, in 2D or 3D.

Mesh = mymesh.primitives.Torus([0,0,0], 1, 0.5, phi_resolution=80, theta_resolution=80)

model = Model(Mesh, agent_parameters={'B': 0.3, 'happy':0})
model.agent_actions = (decide,)
model.grid_actions = (relocate,)

model.seed(2000, state='0')
model.seed(2000, state='1')
model.seed(2000, state='2')

model.iterate(10)

model.animate('segregation3d.gif', show_mesh_edges=False, agent_kwargs={'render_points_as_spheres':True, 'point_size':8})


# %%
