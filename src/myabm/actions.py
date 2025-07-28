import numpy as np
import numba
from numba.experimental import jitclass
from numba import int32, float32, float64
from numba.typed import Dict
from numba.types import string, DictType
from mymesh import *
import scipy

@numba.njit
def null(agent, grid):
    """
    Do nothing function, used for debugging

    Parameters
    ----------
    agent : myabm.agent
        Cell agent to perform the action.
    grid : myabm.AgentGrid
        Grid environment that the agent is in.

    Returns
    -------
    None.

    """
    
    pass