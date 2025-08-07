"""
Ortho Model
===========

pass
"""

#%%
import sys
sys.path.append('../src')
from myabm.ortho import OrthoModel, actions, setup, geometries
import mymesh


#%%
func, bounds = geometries.cross_channel('medium', height=0.5)
model = setup.implicit_scaffold(func, bounds, 0.025, seeding_density=1e3)

model.agent_actions = (actions.proliferate, actions.migrate, actions.produce_oriented, actions.apoptose)
model.model_actions = (actions.update_curvature,)
# # %%
# model.iterate(2)
# # %%
# model.export('test.xdmf')
# %%
