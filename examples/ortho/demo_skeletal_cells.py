"""
Skeletal Cell Behaviors
=======================

These demos illustrate the basic behaviors of skeletal cells in the ortho model,
proliferation, apoptosis, migration, tissue production, and differentiation. 
"""

#%%
import sys
sys.path.insert(0,'../src')
from myabm.ortho import OrthoModel, actions, setup, geometries
import mymesh
import numpy as np

#%%
# Cell Dynamics - Proliferation & Apoptosis
# -----------------------------------------
# 
# At each time step, cells have a random chance of proliferating or undergoing
# apoptosis. The probabilities are determined based on the parameters
# :code:`agent.parameters['ProlifRate']` and 
# :code:`agent.parameters['ApopRate']`, which are defined with units of 
# events/day. 

h = 0.025 # Grid spacing
model = setup.demo_block(h) # Create the demo block scaffold
model.agent_actions = (actions.proliferate, actions.apoptose) # Set cell actions
model.iterate(7, schedule=OrthoModel.substep_saver_schedule) # Iterate the model for 7 days
model.animate('celldynamics.gif', fps=50, view=[.8,.3,.3],              
                show_mesh_edges=True, agent_kwargs=dict(
                  render_points_as_spheres=True, point_size=40),
                show_timer=True)

#%%
# Migration - Random Walk
# -----------------------
# 
# Cells can have a random chance of migrating at each time step, based on the
# parameter :code:`agent.parameters['MigrRate']` (mm/day). The cells then 
# randomly choose where to migrate to from the available neighboring nodes.

h = 0.025 # Grid spacing
model = setup.demo_block(h) # Create the demo block scaffold
model.agent_actions = (actions.migrate,) # Set cell actions
model.iterate(1, schedule=OrthoModel.substep_saver_schedule) # Iterate the model for 1 day
model.animate('migration.gif', show_mesh_edges=True, view=[.8,.3,.3],
            agent_kwargs=dict(render_points_as_spheres=True, point_size=40), 
            show_timer=True)
#%%
# Tissue Production
# -----------------
# 
# Cells produce tissue by filling adjacent voxels with a tissue type 
# corresponding to the cell type. The basic :func:`myabm.ortho.actions.produce`
# function has cells produce tissue at a baseline rate of 
# :code:`agent.parameters['Production']` (mm\ :sup:`3`/day). 
# A curvature-dependent tissue production function is also available: 
# :func:`myabm.ortho.actions.produce_oriented`
# When tissue is being produced in previously unfilled space, once the tissue
# reaches :code:`model.grid_parameters['Tissue Threshold']`, cells automatically
# move to the new surface, creating the effect of cells producing tissue 
# beneath them.


h = 0.025 # Grid spacing
model = setup.demo_block(h) # Create the demo block scaffold
model.agent_grid.NodeData['Cells Allowed'][model.mesh.SurfNodes] = 0 # Keep the cell centered for the demo
model.agent_actions = (actions.produce,) # Set cell actions
model.iterate(10, schedule=OrthoModel.substep_saver_schedule) # Iterate the model for 10 days
model.animate('production.gif', fps=50, view=[.8,.3,.3], 
        agent_kwargs=dict(render_points_as_spheres=True, point_size=40), 
        show_timer=True, tissue_threshold=.01, tissue_opacity=True)

# %%
# Differentiation
# ---------------
# 
# Cells can differentiate into different cell types. Using the mechanobiological
# framework of Prendergast et al. (1997), marrow stromal cells (:code:`'msc'`)
# can differentiate into fibroblasts, chondrocytes, or osteoblasts based on 
# mechanical stimulus (octahedral shear strain and fluid flow). Here, the 
# stimulus is assigned randomly, rather than through mechanical simulation.

h = 0.025 # Grid spacing
# Set the required age before differentiation to 0 (default 6)
parameters = dict(DiffMaturity=0) 
# Create the demo block scaffold
model = setup.demo_block(h, ncells=9, agent_parameters=parameters) 
# Keep the cell centered for the demo
model.agent_grid.NodeData['Cells Allowed'][model.mesh.SurfNodes] = 0 
# Apply a random stimulus for differentiation
model.agent_grid.NodeData['Stimulus'] = np.random.rand(model.NNode)*4
model.agent_actions = (actions.differentiate_prendergast,) # Set cell actions
model.iterate(14, schedule=OrthoModel.substep_saver_schedule) # Iterate the model for 14 days
model.animate('differentiate.gif', fps=50, view=[.8,.3,.3], 
        agent_kwargs=dict(render_points_as_spheres=True, point_size=40), 
        show_timer=True, tissue_threshold=.01, tissue_opacity=True)

# %%
