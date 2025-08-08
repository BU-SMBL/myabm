"""
*in vitro* cell culture
=======================



"""
#%%
from myabm.ortho import OrthoModel, actions, setup, geometries
import mymesh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#%%

h = 0.025 # Grid spacing (mm)
# Setup a well of a 96 well plate
model = setup.wellplate(96, h, media_volume=False) 

model.agent_actions = (actions.proliferate, actions.migrate, actions.apoptose)

model.iterate(5, schedule=OrthoModel.substep_saver_schedule)

model.animate('wellplate.gif', timestep=0.1, show_mesh_edges=False, agent_kwargs=dict(render_points_as_spheres=True, point_size=4))

# %%
# 
t = model.history['Time']
ncells = np.array([len(model.history['Agent Nodes'][i]) for i in range(len(model.history['Agent Nodes']))])

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(t, ncells, color='black')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Number of Cells')
ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(4,4))
ax.spines[['right', 'top']].set_visible(False)
plt.rcParams['font.family'] = 'arial'
plt.show()
# %%
