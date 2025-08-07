#%%
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
model = setup.wellplate(96, h, media_volume=1) # Create the demo block scaffold
model.plotter().show()
# %%
