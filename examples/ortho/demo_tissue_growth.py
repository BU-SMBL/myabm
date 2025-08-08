"""
Tissue Growth
=============

Tissue has been shown to grow in a curvature-dependent manner, with faster
growth in regions of higher concave curvatures while being suppressed in 
regions of convex curvature. Curvature dependent tissue growth is implemented
through the :func:`myabm.ortho.actions.produce_oriented` cell action, and 
requires the model action: :func:`myabm.ortho.actions.update_curvature` to 
calculate the local curvatures.

This example replicates the experimental 
setup of :cite:`Bidan2013` to illustrate this behavior.

"""

#%%
import sys
sys.path.append('../src')
from myabm.ortho import OrthoModel, actions, setup, geometries
import mymesh
import numpy as np

# Setup model
func, bounds = geometries.cross_channel('medium')
h = 0.025 # Grid spacing
model = setup.implicit_scaffold(func, bounds, h, seeding_density=1e3) 

# Scaffold surfaces are smoothed by default, but in this case the geometry has 
# flat walls/sharp edges
model.model_parameters['Smooth Scaffold'] = False 
# Set actions
model.agent_actions = (actions.proliferate, actions.migrate_curvotaxis, actions.produce_oriented, actions.apoptose) # Set cell actions
model.model_actions = (actions.update_curvature,)

# Run the model
model.act(actions.update_curvature) # initialize curvature before iterating
model.iterate(35) # Iterate the model for 35 days

# Generate animation
model.animate('cross_channel.gif', fps=10, timestep=1, view='xy', show_mesh_edges=False,
            show_agents=False, show_timer=True)

# %%
