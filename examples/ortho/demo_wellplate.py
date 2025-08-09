"""
*in vitro* cell culture
=======================

Cell culture are often performed by seeding cells in well plates, which come in 
a few standard sizes (6, 12, 24, 38, 96, 384 wells/plate). Simulating cell 
behavior in a culture plate well can be a convenient way to see if the 
population of cells follows experimental expectations.

A function for generating standard well plate geometries is available in 
:func:`myabm.ortho.geometries.wellplate` and convenient setup function is 
available to easily create a well plate simulation 
(:func:`myabm.ortho.setup.wellplate`).

By default, the well plate geometry is essentially just a platform that cells
exist on, however a volume of space above the bottom of the well can be added
with the :code:`media_volume` optional input, which can either by set to
:code:`True` for a standard recommended media volume, or can be given a float
for a custom media volume. This can be useful if performing a simulation that 
considers nutrient availability or other biochemical factors.


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
# Analysis
# --------
# The model history can be examined to see the population dynamics over time
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
