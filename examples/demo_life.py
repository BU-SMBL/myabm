"""
Conway's Game of Life
=====================

John Conway's Game of Life (or just "Life") is a cellular automaton where agents are created
or die based on interactions with nearby agents, according to four rules:

    1. An agent with fewer than two neighbors dies (under-population)
    2. An agent with 2-3 neighbors lives to the next generation
    3. An agent with more than three neighbors dies (over-population)
    4. An agent is created in an unoccupied grid point if there are exactly three adjacent neighbors (reproduction)

From this simple set of rules, complex patterns and behaviors arise, including
Turing-completeness.

Implementation of the Game of Life
----------------------------------
One way to implement Life would to seed every point in the grid with agents
and then change the state of each agent from "alive" to "dead" or vice-versa.
Alternatively, a grid action can be defined that determines when new agents 
are created or existing agents die, as follows:
"""

#%%
import numpy as np
import numba

from mymesh import *
import sys
sys.path.append('../src')
from myabm import Model

@numba.njit
def life_action(agent, grid):

    # Get the states of neighboring agents in the Moore neighborhood
    neighbor_states = [grid.Agents[grid.NodeAgents[i]].state for i in grid.NodeMooreNeighbors[agent.node] if i in grid.NodeAgents]
    # Count the number of 'live' agents in the neighborhood
    nlive = np.sum(np.array([1 if state == 'live' else 0 for state in neighbor_states]))
    if agent.state == 'live':
        if nlive < 2:
            # Die by underpopulation
            grid.change_agent(agent, 'dead', True)
        elif nlive > 3:
            # Die by overpopulation
            grid.change_agent(agent, 'dead', True)
    elif agent.state == 'dead':
        if nlive == 3:
            # Live by reproduction
            grid.change_agent(agent, 'live', True)

# %%
# Random initialization of the Game of Life
# -----------------------------------------
#

Grid = primitives.Grid2D([0,1,0,1],0.025)

model = Model(Grid)
# model.grid_actions = (game_of_life,)
model.agent_actions = (life_action,)
model.seed(500, state='live')
model.seed(Grid.NNode-500, state='dead')
model.iterate(100)

color = dict(live='black', dead=None)
model.animate('game_of_life.gif', view='xy', show_mesh=False, state_color=color)

# %% 
# Gosper glider gun
# -----------------
# Bill Gosper's glider gun was the first pattern in the Game of Life that was
# found to grow infinitely, by generating an infinite sequence of "gliders".

Grid = primitives.Grid2D([0,1,0,1],0.025)
gosper_gun_nodes = np.array([ 154,  155,  195,  196,  563,  564,  565,  603,  
                            607,  643,  649,  684,  690,  728,  767,  771,  809, 
                            810,  811,  851,  975,  976,  977, 1016, 1017, 1018, 
                            1056, 1060, 1137, 1138, 1142, 1143, 1550,  1551, 
                            1591, 1592])

# Setup the model
model = Model(Grid)
model.agent_actions = (life_action,)

# Seed the grid
model.seed(len(gosper_gun_nodes), nodes=gosper_gun_nodes, state='live')
model.seed(model.NNode-len(gosper_gun_nodes), nodes=np.setdiff1d(np.arange(model.NNode), gosper_gun_nodes), state='dead')

# Run the simulation
model.iterate(200)

color = dict(live='black', dead=None)
model.animate('gosper_gun.gif', view='xy', show_mesh=False, state_color=color)


# %%
