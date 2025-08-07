"""
Curvotaxis
==========

Cells have been shown to migrate in response to a variety of cues, including 
chemical signaling ("chemotaxis") and substrate/ECM stiffness ("durotaxis"). 
:cite:`Pieuchot2018a` demonstrated that cells also exhibit "curvotaxis", or 
curvature-dependent migration. Both migration rate and preferential migration
direction were demonstrated to be curvature dependent, which can be implemented
in the model by weighting the probability of migration by the curvature at 
the cells current location and the probabilities of which site to move to by 
the curvatures at the surrounding sites. 

"""

# %%
import sys
sys.path.append('../src')
from myabm.ortho import OrthoModel, actions, setup, geometries
import mymesh
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
# %%
# Surface geometry for demonstrating curvotaxis
# ---------------------------------------------
# 
# :cite:`Pieuchot2018a` studied the migration of cells seeded on sinusoidal 
# substrates. These can be created by warping a 2D grid mesh to a sinusoidal 
# implicit function.
# To sufficiently resolve the geometry of the sinusoidal surface, a smaller
# grid size (0.005 mm) than the standard (0.025 mm) was used and the time
# step was adjusted proportionally. While reducing the grid size alters
# the contact inhibition of migration and allows for a locally higher than
# normal cell density, the low seeding density of the cells on the surface
# makes this impact negligible.

func, bounds = geometries.sinusoidal_surface(amplitude=0.01, period=0.1)
h = .005 # Grid spacing
dt = 0.02 * (h/.025)
surf = mymesh.primitives.Grid2D(bounds[:4],h)

# Shape the surface
xnodes = np.where((surf.NodeCoords[:,0] == bounds[0]) | 
                        (surf.NodeCoords[:,0] == bounds[1]))[0]
ynodes = np.where((surf.NodeCoords[:,1] == bounds[2]) | 
                        (surf.NodeCoords[:,1] == bounds[3]))[0]
constraints = np.vstack((np.column_stack(( # Constrain the boundaries to 
        xnodes, # nodes
        np.repeat(0, len(xnodes)), # axis
        np.zeros(len(xnodes)) # constraint value
        )),
        np.column_stack((
        ynodes, # nodes
        np.repeat(1, len(ynodes)), # axis
        np.zeros(len(ynodes)) # constraint value
        ))
    ))

surf = mymesh.implicit.SurfaceNodeOptimization(surf, func, h, iterate=200, 
                            constraint=constraints, smooth=False, InPlace=True)

# %%
# Running the curvotaxis simulation
# ---------------------------------
# 
# The :func:`myabm.ortho.actions.migrate_curvotaxis` agent action determines
# the rate of migration and the direction of migration with a 
# curvature-dependent random walk process. 
#
# The migration rate is determined by the mean curvature at the node the cell 
# is currently on (:math:`H_i`)by the equation:
#
# .. math::
#
#    k = k_{m} \left(m_a m_b^{H_i} + m_c \right)
#
# where :math:`k_m` is the baseline migration rate 
# (:code:`agent.parameters['MigrRate']`) and :math:`m_a, \ m_b, \ m_c` are 
# migration weights (:code:`agent.parameters['MigrationWeight0']`,
# :code:`agent.parameters['MigrationWeight1']`, 
# :code:`agent.parameters['MigrationWeight2']`). The default values
# of these parameters were obtained by curve fitting to migration rate data
# from :cite:`Pieuchot2018a`: :math:`k_m = 1`, :math:`m_a = 0.101`, 
# :math:`m_b = 1.083`, :math:`m_c = 0.475`.
#
# The migration direction is chosen based on the mean curvatures of the 
# neighboring nodes 
#
# .. math::
#   
#    p(H_n) = 1 - (\frac{1}{1 + \exp(m_d(H_n - m_e))})
#
# where :math:`m_d` and :math:`m_e` are migration weights 
# (:code:`agent.parameters['MigrationWeight3']`, 
# :code:`agent.parameters['MigrationWeight4']`).  The default values were
# determined by Bayesian optimization fitting the result of  simulations on
# multiple different sinusoidal substrates to the experimental data of 
# :cite:`Pieuchot2018a`: :math:`m_a = 0.232 \ mm`, # :math:`m_b = -19.975 \ mm^{-1}`
#
#
# Since this example is only studying migration over a 1-day period, other 
# cell behaviors, like proliferation and apoptosis, can be neglected.


model = OrthoModel(surf)
# Scale the time step to be proportional with the grid spacing
model.agent_grid.TimeStep = dt
model.model_parameters['SubStep'] = dt
model.seed(100, state='msc')
model.agent_grid.ElemData['Scaffold Fraction'][:] = 1
model.agent_grid.ElemData['Volume Fraction'][:] = 1

# Calculate curvatures
k1, k2 = mymesh.curvature.CubicFit(model.mesh.NodeCoords, 
                                    model.mesh.NodeConn, 
                                    model.mesh.NodeNeighbors, 
                                    model.mesh.NodeNormals)
model.agent_grid.NodeData['Mean Curvature'] = mymesh.curvature.MeanCurvature(k1, k2)

# Set actions
model.agent_actions = (actions.migrate_curvotaxis,) # Set cell actions

# Run the model
model.iterate(1, schedule=OrthoModel.substep_saver_schedule) 

# Generate animation
model.animate('curvotaxis.gif', fps=20, show_mesh_edges=False,
            show_agents=True, show_timer=True, 
            agent_kwargs=dict(render_points_as_spheres=True), 
            scalars='Mean Curvature', clim=(-10,10), cmap='coolwarm')

# %% 
# Comparison to experimental data
# -------------------------------
#
# The results of the simulation can be compared to the experimental results of 
# :cite:`Pieuchot2018a`, which looked at the distribution of cells along the 
# surface, binning by height (which for these sinusoidal surfaces is linearly
# correlated to mean curvature).
# 
# .. note::
#       
#       Due to the randomness of the model, the results can vary, but typically
#       averaging the results of a few repeated runs gives good agreement with
#       the experimental data


# Plot histogram to compare to experimental data
cellZ = model.mesh.NodeCoords[model.history['Agent Nodes'][-1],2]
hist, edges = np.histogram(cellZ, bins=5, range=(-0.005, 0.005))
centers = np.array([(edges[i]+edges[i+1])/2 for i in range(len(edges)-1)])

# Experimental data from Pieuchot et al. (2018) 
average_height = np.array([-4.0, -2.0, 0, 2.0, 4.0])
percent_nuclei = np.array([59.6644, 27.2483, 10.8054, 2.3490, 0])
percent_nuclei_std = np.array([6.1074, 2.8188, 1.8792, 1.8792, 0])      

colors = ['#00007E', '#1C15DB', '#7E5E7E', '#E1A92A', '#F7E6C4']

fig, ax = plt.subplots(figsize=(4,4))
bar_width = 0.7
bar1 = ax.bar(
    centers*1000-bar_width/1.9,
    hist,
    width=bar_width,
    color=colors[:len(centers)],
    edgecolor='white',
    hatch='////',
    label='Simulated',
)

bar2 = ax.bar(
    average_height+bar_width/1.9, 
    percent_nuclei,
    yerr=percent_nuclei_std,
    width=bar_width,
    color=colors[:len(average_height)],
    label='Pieuchot et al.',
)

ax.set_xlabel('Height (μm)')
ax.set_ylabel('% of nuclei')
ax.set_ylim(0, 100)
ax.legend()
plt.tight_layout()
plt.show()
# %%
