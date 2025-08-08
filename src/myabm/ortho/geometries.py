#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Fri Jun 13 19:18:28 2025
# @author: toj
"""
Predefined Model Geometries
===========================
.. autosummary::
    :toctree: submodules/

    square_channel
    cross_channel
    sinusoidal_surface
    wellplate
    demo_block
    strut

"""
from mymesh import implicit
import numpy as np

def square_channel(size, height=2):
    """
    A square pore channel, based on :cite:t:`Bidan2013`.

    Parameters
    ----------
    size : str, float
        Size of the pore, either 'medium'/'large' for the sizes used by 
        :cite:t:`Bidan2013` (perimeter = 4.71 mm,  6.28 mm), or a float for a 
        custom perimeter
    height : int, optional
        Height of the channel, by default 2.

    Returns
    -------
    func : callable
        implicit function of the form f(x,y,z)
    bounds : list
        Outer bounds of the grid, formatted as [xmin, xmax, ymin, ymax, zmin, zmax]

    """    
    if type(size) is str:
        if size == 'medium':
            Pmedium = 4.71 # mm
            Perimeter = Pmedium
        elif size == 'large':
            Plarge = 6.28 # mm
            Perimeter = Plarge
    else:
        Perimeter = size
        
    side = Perimeter/4
    outer = side/2+0.15
    square = implicit.box(-side/2, side/2, -side/2, side/2, 0, 2)
    
    bounds = [-outer,outer,-outer,outer,0,height]
    box = implicit.box(*bounds)
    func = implicit.difff(box,square)
    
    return func, bounds

def cross_channel(size, height=2):
    """
    A cross-shaped pore channel, based on :cite:t:`Bidan2013`.

    Parameters
    ----------
    size : str, float
        Size of the pore, either 'medium'/'large' for the sizes used by 
        :cite:t:`Bidan2013` (perimeter = 4.71 mm,  6.28 mm), or a float for a 
        custom perimeter
    height : int, optional
        Height of the channel, by default 2.

    Returns
    -------
    func : callable
        implicit function of the form f(x,y,z)
    bounds : list
        Outer bounds of the grid, formatted as [xmin, xmax, ymin, ymax, zmin, zmax]

    """    

    if type(size) is str:
        if size == 'medium':
            Pmedium = 4.71 # mm
            Perimeter = Pmedium
        elif size == 'large':
            Plarge = 6.28 # mm
            Perimeter = Plarge
    else:
        Perimeter = size
        
    side = Perimeter/12
    L = 1.5*side 
    W = 0.5*side
    
    # L = 3*P/28
    # W = P/28
    outer = L+0.1
    rect1 = implicit.box(-L, L, -W, W, 0, 2)
    rect2 = implicit.box(-W, W, -L, L, 0, 2)
    cross = implicit.unionf(rect1,rect2)
    bounds = [-outer,outer,-outer,outer,0,height]
    box = implicit.box(*bounds)
    func = implicit.difff(box,cross)
    
    return func, bounds

def sinusoidal_surface(amplitude, period):
    """
    Sinusoidal substrates based on :cite:t:`Pieuchot2018a`.

    Parameters
    ----------
    amplitude : float
        Distance between the max and min heights of the sinusoids
    period : float
        Period of osscilation 

    Returns
    -------
    func : callable
        implicit function of the form f(x,y,z)
    bounds : list
        Outer bounds of the grid, formatted as [xmin, xmax, ymin, ymax, zmin, zmax]
    """    

    func = lambda x,y,z : z - amplitude*np.sin(2*np.pi/period*x)/4 + amplitude*np.sin(2*np.pi/period*y)/4
    bounds = [0, 0.6, 0, 0.6, -0.04, 0.04]

    return func, bounds

def wellplate(size, media_volume=None):
    """
    Cell culture well plate geometries.
    Well plate specifications based on: `Useful Numbers for Cell Culture <https://www.thermofisher.com/us/en/home/references/gibco-cell-culture-basics/cell-culture-protocols/cell-culture-useful-numbers.html>`_

    +------------+-------------------------------------+---------------------------+---------------------------------------------------+
    | Well Plate | Surface Area (mm\ :superscript:`2`) | Growth Medium Volume (mL) | Recommended Seeding Density                       |
    +============+=====================================+===========================+===================================================+
    | 6-well     | 960                                 | 2                         | 0.3e6 cells/well, 313 cells/mm\ :superscript:`2`  |
    +------------+-------------------------------------+---------------------------+---------------------------------------------------+
    | 12-well    | 350                                 | 1                         | 0.1e6 cells/well, 285 cells/mm\ :superscript:`2`  |
    +------------+-------------------------------------+---------------------------+---------------------------------------------------+
    | 24-well    | 190                                 | 0.75                      | 0.05e6 cells/well, 263 cells/mm\ :superscript:`2` |
    +------------+-------------------------------------+---------------------------+---------------------------------------------------+
    | 48-well    | 110                                 | 0.3                       | 0.03e6 cells/well, 273 cells/mm\ :superscript:`2` |
    +------------+-------------------------------------+---------------------------+---------------------------------------------------+
    | 96-well    | 32                                  | 0.15                      | 0.01e6 cells/well, 313 cells/mm\ :superscript:`2` | 
    +------------+-------------------------------------+---------------------------+---------------------------------------------------+
    
    Parameters
    ----------
    size : int
        Well plate size, must be one of (6, 12, 24, 48, 96)
    media_volume : float, bool, NoneType, optional
        Specific media volume to override the default well
        recommendation, by default None

    Returns
    -------
    func : callable
        implicit function of the form f(x,y,z)
    bounds : list
        Outer bounds of the grid, formatted as [xmin, xmax, ymin, ymax, zmin, zmax]

    """    
    if size == 96: 
        area = 32 # mm2
        if media_volume is True:       
            vol = 0.1 * 1000 # ml -> mm3
            height = vol/area
        elif media_volume is None or media_volume is False:
            height = 0.025
        else:
            vol = media_volume
            height = vol/area
        
        r = np.sqrt(area/np.pi) # 96 well plate surface area = 0.32 cm2 = 32 mm2
        bounds = [-3.5,3.5,-3.5,3.5,0,height]
    
    elif size == 48:        
        
        area = 110 # mm2
        if media_volume is True:       
            vol = 0.3 * 1000 # ml -> mm3
            height = vol/area
        elif media_volume is None or media_volume is False:
            height = 0.025
        else:
            vol = media_volume
            height = vol/area
        
        r = np.sqrt(area/np.pi) 
        bounds = [-6,6,-6,6,0,height]

    else:
        raise ValueError('Invalid well plate size, must be one of: 96, 48, 24, 12, 6.')
        
    func = implicit.intersectionf(implicit.cylinder([0,0,0], r), implicit.zplane(0, -1))
    return func, bounds

def demo_block(h):
    """
    A small block of scaffold for demonstrating agent behaviors.

    Parameters
    ----------
    h : float
        Grid spacing

    Returns
    -------
    func : callable
        implicit function of the form f(x,y,z)
    bounds : list
        Outer bounds of the grid, formatted as [xmin, xmax, ymin, ymax, zmin, zmax]
    """    
    bounds = [0, 2*h, 0, 2*h, 0, 6*h]
    func = implicit.box(0, 2*h, 0, 2*h, 0, h)
    
    return func, bounds
    
def strut(L, d):
    """
    Strut lattice

    Parameters
    ----------
    L : float
        Unit cell size/period of the lattice.
    d : float
        Duty cycle, must be in the range [0, 1]. This controls the thickness
        of the strut walls, thickness = L*d. 

    Returns
    -------
    strut : callable
        Implicit function of the form f(x,y,z)
    bounds : list
        6 element list ([xmin, xmax, ymin, ymax, zmin, zmax]) that give the bounds
        of one strut unit of size (L + L*d). Note this is not the "unit cell" 
        bounds, which would be [0, L, 0, L, 0, L], rather this gives the 
        symmetric unit that has walls on both sides. Since the function is
        periodic, different bounds can be used to get a larger multi-unit
        structure.

    """
    
    # L = period, d=duty cycle
    s = 2*d - 1
    x = np.linspace(0,L+L*d, 100)
    u = x/L - s/2
    y = np.sign(2*(u - np.floor(u + 1/2))+s) # shifted saw wave
    
    saw = lambda u, s : 2*(u - np.floor(u + 1/2))+s
    square = lambda x : -np.sign(saw(x/L - s/2, s))
    
    strut = lambda x,y,z : square(x) + square(y) + square(z)
    
    bounds = [0, L+L*d, 0, L+L*d, 0, L+L*d]
    
    return strut, bounds