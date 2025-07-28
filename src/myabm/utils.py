"""
Utility functions to support various functionalities.

.. currentmodule:: myabm.utils

numba-compatibility
===================
.. autosummary::
    :toctree: submodules/

    dict_to_Dict

"""

import numba
from numba.typed import Dict

def dict_to_Dict(dictionary, key_type, value_type):
    """
    Convert python dictionaries to numba.typed.Dict

    Parameters
    ----------
    dictionary : dict
        Dictionary
    key_type : numba.core.types
        Numba type used for dictionary keys (e.g. `numba.types.string`)
    value_type : numba.core.types
        Numba type used for dictionary values (e.g. `numba.types.float64`)

    Returns
    -------
    numba_dict : numba.typed.typeddict.Dict
        Numba-compatible dictionary

    """    
    numba_dict = Dict.empty(key_type=key_type, value_type=value_type)

    for key in dictionary.keys():

        try:
            if key_type == numba.types.string:
                key = str(key)
            elif key_type in (numba.types.float32, numba.types.float64):
                key = float(key)
            elif key_type in (numba.types.int32, numba.types.int64):
                key = int(key)
            
        except:
            raise ValueError(f"dictionary contains keys of type {type(dictionary[key])} that aren't convertible to the key_type: {value_type}")

        try:
            if value_type == numba.types.string:
                value = str(dictionary[key])
            elif value_type in (numba.types.float32, numba.types.float64):
                value = float(dictionary[key])
            elif value_type in (numba.types.int32, numba.types.int64):
                value = int(dictionary[key])

            numba_dict[key] = value
        except:
            raise ValueError(f"dictionary contains values of type {type(dictionary[key])} that aren't convertible to the value_type: {value_type}")

    return numba_dict
