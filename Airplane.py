#%%
import numpy as np
import numba

import ABM, ABM_run
from mymesh import *

# %%
@numba.njit
def unboard(agent, grid):

    node = agent.node
    point = grid.NodeCoords[node]

    edge_ids = grid.NodeEdgeConn[node]
    edge_ids = edge_ids[edge_ids!=-1]
    edge_set = set(edge_ids)
    node_set = set(grid.Edges[edge_set.pop()])
    while len(edge_set) > 0:
        node_set.update(grid.Edges[edge_set.pop()])

    if point[1] == 0:
        if point[0] == 0:
            # off plane
            del grid.NodeAgents[node]
        else:
            # Move towards front
            open_neighbors = np.array([n for n in node_set if (n not in grid.NodeAgents)])
            for n in open_neighbors:
                if grid.NodeCoords[n,0] < point[0] and grid.NodeCoords[n,1] == 0:
                    # Update data structure
                    del grid.NodeAgents[node]               # Remove cell from old node
                    agent.node = n                          # Update cell's node id
                    grid.NodeAgents[n] = agent              # Move cell to new node
                    return
    else:
        # Move towards aisle
        open_neighbors = np.array([n for n in node_set if (n not in grid.NodeAgents)])
        for n in open_neighbors:
            if np.sign(point[1])*(point[1] - grid.NodeCoords[n][1]) > 0:
                # Update data structure
                del grid.NodeAgents[node]               # Remove cell from old node
                agent.node = n                          # Update cell's node id
                grid.NodeAgents[n] = agent              # Move cell to new node
                return

#%%
Grid = primitives.Grid2D([0,1,-0.5,0.5],1/8)
# Grid = primitives.Torus([0,0,0], 1, .5, theta_resolution=60, phi_resolution=60)

Grid.NodeData['aisle'] = np.zeros(Grid.NNode)
Grid.NodeData['aisle'][Grid.NodeCoords[:,1] == 0] = 1
Grid.NodeData['seats'] = np.zeros(Grid.NNode)
Grid.NodeData['seats'][Grid.NodeCoords[:,1] != 0] = 1

#%%
AgentGrid = ABM.InitializeAgentGrid(Grid)
AgentGrid.seed(20)

AgentMeshes = [mesh(Grid.NodeCoords[list(AgentGrid.NodeAgents.keys())])]
#%%
t = 40
for i in range(t):
    AgentGrid.run_agents((unboard,))
    if len(AgentGrid.NodeAgents) == 0:
        break
    AgentMeshes.append(ABM.get_AgentMesh(AgentGrid))
Grid.write('grid.vtu')
ABM_run.export('plane', AgentMeshes=AgentMeshes)
# %%
