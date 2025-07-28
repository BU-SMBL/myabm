#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 09:50:10 2025

@author: toj
"""
import numpy as np

def prendergast_mat(matIDs):
    """
    Get tissue-specific material properties

    Parameters
    ----------
    matIDs : array_like
        Array of material IDs for each element
        
        - 1: Granulation Tissue/Neotissue
        - 2: Fibrous Tissue
        - 3: Cartilage
        - 4: Bone Marrow
        - 5: Immature woven bone
        - 6: Mature woven bone
        - 7: Lamellar bone
        - 8: Scaffold
        
    Returns
    -------
    E : np.ndarray
        Young's modulus (MPa).
    nu : np.ndarray
        Poisson's ratio.
    K : np.ndarray
        Permeability of fluid through solid (mm/s).
    FluidBulk : np.ndarray
        Bulk modulus of the fluid (MPa).
    FluidSpecWeight : np.ndarray
        Specific weight of the fluid (N/mm3).
    Porosity : np.ndarray
        Porosity of the solid (0-1).
    SolidBulk : np.ndarray
        Bulk modulus of the solid.

    """
    
    
    matIDs = np.asarray(matIDs)
    
    # Existing materials
    water = matIDs == 0   # pseudo-water
    neo = matIDs == 1   # Neotissue/granulation tissue
    fib = matIDs == 2   # fibrous tissue
    cart = matIDs == 3  # cartilage
    marr = matIDs == 4  # bone marrow
    ibone = matIDs == 5 # immature woven bone
    mbone = matIDs == 6 # mature moven bone
    lbone = matIDs == 7 # lamellar bone
    scaff = matIDs == 8 # scaffold
        
    # Young's modulus (MPa)
    Ewater = 1e-6
    Eneo = 0.2      # ID = 1
    Efib = 2        # ID = 2
    Ecart = 10      # ID = 3
    Emarr = 2       # ID = 4
    Eibone = 1000   # ID = 5
    Embone = 6000   # ID = 6
    Elbone = 13e3   # ID = 7
    Escaff = 12e3   # ID = 8
    
    # Poisson's ratio
    Nuwater = 0.167
    Nuneo = 0.167
    Nufib = 0.167
    Nucart = 0.167
    Numarr = 0.167
    Nuibone = 0.3
    Numbone = 0.3
    Nulbone = 0.3
    Nuscaff= 0.3
    
    # Permeability/Hydraulic Conductivity (mm/s)
    Kwater = 5e-2
    Kneo = 9.81e-8
    Kfib = 9.81e-8
    Kcart = 4.905e-8
    Kmarr = 9.81e-8
    Kibone = 9.81e-7 # Lacroix 2002
    Kmbone = 3.6297e-6 # Lacroix 2002
    Klbone = 9.81e-11 # Lacroix cortical bone
    Kscaff = 9.81e-11 # Lacroix cortical bone
    
    # Solid Grain Bulk Modulus (MPa)
    SBulkwater = 2300
    SBulkneo = 2300
    SBulkfib = 2300
    SBulkcart = 3400
    SBulkmarr = 2300
    SBulkibone = 13920
    SBulkmbone = 13920
    SBulklbone = 13920
    SBulkscaff = Escaff/(3*(1-2*Nuscaff))
    
    # Fluid Properties
    FluidBulk = 2300 # Bulk modulus of fluid (MPa)
    FluidSpecWeight = 9.81e-6 # Specific weight of water (N/mm^3)
    
    E = np.zeros(len(matIDs))
    nu = np.zeros(len(matIDs))
    K = np.zeros(len(matIDs))
    FluidBulk = np.repeat(FluidBulk,len(matIDs))
    FluidSpecWeight = np.repeat(FluidSpecWeight,len(matIDs))
    Porosity = np.zeros(len(matIDs))
    SolidBulk = np.zeros(len(matIDs))
    
    E[water] = Ewater
    E[neo] = Eneo
    E[fib] = Efib
    E[cart] = Ecart
    E[marr] = Emarr
    E[ibone] = Eibone
    E[mbone] = Embone
    E[lbone] = Elbone
    E[scaff] = Escaff
    
    nu[water] = Nuwater
    nu[neo] = Nuneo
    nu[fib] = Nufib
    nu[cart] = Nucart
    nu[marr] = Numarr
    nu[ibone] = Nuibone
    nu[mbone] = Numbone
    nu[lbone] = Nulbone
    nu[scaff] = Nuscaff
    
    K[water] = Kwater
    K[neo] = Kneo
    K[fib] = Kfib
    K[cart] = Kcart
    K[marr] = Kmarr
    K[ibone] = Kibone
    K[mbone] = Kmbone
    K[lbone] = Klbone
    K[scaff] = Kscaff
    
    Porosity[water] = 1
    Porosity[neo] = 0.8
    Porosity[fib] = 0.8
    Porosity[cart] = 0.8
    Porosity[marr] = 0.8
    Porosity[ibone] = 0.8
    Porosity[mbone] = 0.8
    Porosity[lbone] = 0.04
    Porosity[scaff] = 0.04
    
    SolidBulk[water] = SBulkwater
    SolidBulk[neo] = SBulkneo
    SolidBulk[fib] = Efib
    SolidBulk[cart] = Ecart
    SolidBulk[marr] = SBulkmarr
    SolidBulk[ibone] = SBulkibone
    SolidBulk[mbone] = SBulkmbone
    SolidBulk[lbone] = SBulklbone
    SolidBulk[scaff] = SBulkscaff

    return E, nu, K, FluidBulk, FluidSpecWeight, Porosity, SolidBulk

def prendergast_mat_stiffening(matIDs, maturities):
    """
    Get tissue-specific material properties with tissue maturation

    Parameters
    ----------
    matIDs : array_like
        Array of material IDs for each element
        
        - 1: Granulation Tissue/Neotissue
        - 2: Fibrous Tissue
        - 3: Cartilage
        - 4: Bone Marrow
        - 5: Immature woven bone
        - 6: Mature woven bone
        - 7: Lamellar bone
        - 8: Scaffold
        
        
    maturities : array_like
        Array of tissue maturities for each element. Tissue maturity is an effective age that is
        used to scale the material properties as a tissue matures. All properties for all tissues 
        are tuned to mature in 60 days.

    Returns
    -------
    E : np.ndarray
        Young's modulus (MPa).
    nu : np.ndarray
        Poisson's ratio.
    K : np.ndarray
        Permeability of fluid through solid (mm/s).
    FluidBulk : np.ndarray
        Bulk modulus of the fluid (MPa).
    FluidSpecWeight : np.ndarray
        Specific weight of the fluid (N/mm3).
    Porosity : np.ndarray
        Porosity of the solid (0-1).
    SolidBulk : np.ndarray
        Bulk modulus of the solid.

    """
    
    
    matIDs = np.asarray(matIDs)
    maturities = np.asarray(maturities)
    
    # Existing materials
    neo = matIDs == 1   # Neotissue/granulation tissue
    fib = matIDs == 2   # fibrous tissue
    cart = matIDs == 3  # cartilage
    marr = matIDs == 4  # bone marrow
    ibone = matIDs == 5 # immature woven bone
    mbone = matIDs == 6 # mature moven bone
    lbone = matIDs == 7 # lamellar bone
    scaff = matIDs == 8 # scaffold
        
    # Young's modulus (MPa)
    Eneo = 0.2      # ID = 1
    Efib = 2        # ID = 2
    Ecart = 10      # ID = 3
    Emarr = 2       # ID = 4
    Eibone = 1000   # ID = 5
    Embone = 6000   # ID = 6
    Elbone = 13e3   # ID = 7
    Escaff = 12e3   # ID = 8
    
    # Poisson's ratio
    Nuneo = 0.167
    Nufib = 0.167
    Nucart = 0.167
    Numarr = 0.167
    Nuibone = 0.3
    Numbone = 0.3
    Nulbone = 0.3
    Nuscaff= 0.3
    
    # Permeability/Hydraulic Conductivity (mm/s)
    Kneo = 9.81e-8
    Kfib = 9.81e-8
    Kcart = 4.905e-8
    Kmarr = 9.81e-8
    Kibone = 9.81e-7 # Lacroix 2002
    Kmbone = 3.6297e-6 # Lacroix 2002
    Klbone = 9.81e-11 # Lacroix cortical bone
    Kscaff = 9.81e-11 # Lacroix cortical bone
    
    # Solid Grain Bulk Modulus (MPa)
    SBulkneo = 2300
    SBulkfib = 2300
    SBulkcart = 3400
    SBulkmarr = 2300
    SBulkibone = 13920
    SBulkmbone = 13920
    SBulklbone = 13920
    SBulkscaff = Escaff/(3*(1-2*Nuscaff))
    
    # Fluid Properties
    FluidBulk = 2300 # Bulk modulus of fluid (MPa)
    FluidSpecWeight = 9.81e-6 # Specific weight of water (N/mm^3)
    
    E = np.zeros(len(matIDs))
    nu = np.zeros(len(matIDs))
    K = np.zeros(len(matIDs))
    FluidBulk = np.repeat(FluidBulk,len(matIDs))
    FluidSpecWeight = np.repeat(FluidSpecWeight,len(matIDs))
    Porosity = np.zeros(len(matIDs))
    SolidBulk = np.zeros(len(matIDs))
    
    E[neo] = Eneo
    E[fib] = np.minimum(Eneo*np.exp(maturities[fib]*np.log(Efib/Eneo)/60), Efib)
    E[cart] = np.minimum(Eneo*np.exp(maturities[cart]*np.log(Ecart/Eneo)/60), Ecart)
    E[marr] = Emarr
    E[ibone] = np.minimum(Eneo*np.exp(maturities[ibone]*np.log(Eibone/Eneo)/60), Eibone)
    E[mbone] = np.minimum(Eneo*np.exp(maturities[mbone]*np.log(Embone/Eneo)/60), Embone)
    E[lbone] = np.minimum(Eneo*np.exp(maturities[lbone]*np.log(Elbone/Eneo)/60), Elbone)
    E[scaff] = Escaff
    
    nu[neo] = Nuneo
    nu[fib] = Nufib
    nu[cart] = Nucart
    nu[marr] = Numarr
    nu[ibone] = np.minimum(Nuneo*np.exp(maturities[ibone]*np.log(Nuibone/Nuneo)/60), Nuibone)
    nu[mbone] = np.minimum(Nuneo*np.exp(maturities[mbone]*np.log(Numbone/Nuneo)/60), Numbone)
    nu[lbone] = np.minimum(Nuneo*np.exp(maturities[lbone]*np.log(Nulbone/Nuneo)/60), Nulbone)
    nu[scaff] = Nuscaff
    
    K[neo] = Kneo
    K[fib] = Kfib
    K[cart] = np.maximum(Kneo*np.exp(maturities[cart]*np.log(Kcart/Kneo)/60), Kcart)
    K[marr] = Kmarr
    K[ibone] = np.minimum(Kneo*np.exp(maturities[ibone]*np.log(Kibone/Kneo)/60), Kibone)
    K[mbone] = np.minimum(Kneo*np.exp(maturities[mbone]*np.log(Kmbone/Kneo)/60), Kmbone)
    K[lbone] = np.minimum(Kneo*np.exp(maturities[lbone]*np.log(Klbone/Kneo)/60), Klbone)
    K[scaff] = Kscaff
    
    Porosity[neo] = 0.8
    Porosity[fib] = 0.8
    Porosity[cart] = 0.8
    Porosity[marr] = 0.8
    Porosity[ibone] = 0.8
    Porosity[mbone] = 0.8
    Porosity[lbone] = np.maximum(0.8*np.exp(maturities[lbone]*np.log(.04/.8)/60), .04)
    Porosity[scaff] = 0.04
    
    SolidBulk[neo] = SBulkneo
    SolidBulk[fib] = np.minimum(SBulkneo*np.exp(maturities[fib]*np.log(SBulkfib/SBulkneo)/60), Efib)
    SolidBulk[cart] = np.minimum(SBulkneo*np.exp(maturities[cart]*np.log(SBulkcart/SBulkneo)/60), Ecart)
    SolidBulk[marr] = SBulkmarr
    SolidBulk[ibone] = np.minimum(SBulkneo*np.exp(maturities[ibone]*np.log(SBulkibone/SBulkneo)/60), Eibone)
    SolidBulk[mbone] = np.minimum(SBulkneo*np.exp(maturities[mbone]*np.log(SBulkmbone/SBulkneo)/60), Embone)
    SolidBulk[lbone] = np.minimum(SBulkneo*np.exp(maturities[lbone]*np.log(SBulklbone/SBulkneo)/60), Elbone)
    SolidBulk[scaff] = SBulkscaff

    return E, nu, K, FluidBulk, FluidSpecWeight, Porosity, SolidBulk

def prendergast_update(material, maturity, shear, flow, dt, resorption=True, lamellar=False, setpoints=(0.01, 0.267, 1, 3)):
    """
    Update materials and tissue maturities based on mechanical stimulus (octahedral shear strain and fluid flow)

    Parameters
    ----------
    material : array_like
        Array of material IDs for each element of the current tissue
    maturity : array_like
        Array of (effective) tissue maturities for each element of the current tissue
    shear : np.ndarray
        Array of element values of octahedral shear strain
    flow : np.ndarray
        Array of element values of relative fluid flow.
    dt : float
        Time step.

    Returns
    -------
    newMat : np.ndarray
        Array of material IDs for each element of the updated tissue
    newMaturity : np.ndarray
        Array of (effective) tissue ages for each element of the updated tissue

    """
    
    
    Eneo = 0.2      # ID = 1
    Efib = 2        # ID = 2
    Ecart = 10      # ID = 3
    Emarr = 2       # ID = 4
    Eibone = 1000   # ID = 5
    Embone = 6000   # ID = 6
    Elbone = 13e3   # ID = 7
    Escaff = 12e3   # ID = 8
    
    neo = material == 1
    fib = material == 2
    cart = material == 3
    marr = material == 4
    ibone = material == 5
    mbone = material == 6
    lbone = material == 7
    scaff = material == 8
    
    newMat = np.copy(material)
    prevMat = material
    newMaturity = np.copy(maturity)
    
    
    prevMod, _, _, _, _, _, _ = prendergast_mat(material, maturity)
    
    Stimulus = shear/.0375 + flow*1000/3

    # Update material assignments 
    newMat[(Stimulus <= setpoints[0])] = -1 # resorption or non-stiffening
    if lamellar:
        newMat[(Stimulus > setpoints[0]) & (Stimulus <= setpoints[2])] = 7 # Bone
    else:
        newMat[(Stimulus > setpoints[0]) & (Stimulus <= setpoints[1])] = 6 # Mature Bone
        newMat[(Stimulus > setpoints[1]) & (Stimulus <= setpoints[2])] = 5 # Mature Bone
    newMat[(Stimulus > setpoints[2]) & (Stimulus <= setpoints[3])] = 3 # Cartilage
    newMat[(Stimulus > setpoints[3])] = 2 # Fibrous
    newMat[scaff] = 8 # Scaffold remains 
    
    # Tissue type switching rules - prevents spontaneous transformation of e.g. bone into cartilage
    # Tissue intended to switch to fibrous from bone but stiffer than max fib stiffness switched to cartiage (subject to check) instead
    newMat[(newMat == 2) & (ibone | mbone | lbone) & (prevMod > Efib)] = 3
    # Tissue intended to switch to cartilage from bone but stiffer than max cartilage stiffness switched to bone
    newMat[(newMat == 3) & (ibone | mbone | lbone) & (prevMod > Ecart)] = prevMat[(newMat == 3) & (ibone | mbone | lbone) & (prevMod > Ecart)]
    # Tissue intended to switch to fibrous from cartilage but stiffer than fibrous stiffness switched to cartilage
    newMat[(newMat == 2) & (cart) & (prevMod > Efib)] = 3
    
    # Update tissue maturity 
    newMaturity[marr] = 0
    newMaturity[scaff] = 0
    newMaturity[(newMat == 2) & (fib)] += dt
    newMaturity[(newMat == 3) & (cart)] += dt
    newMaturity[((newMat == 5) | (newMat == 6) | (newMat == 7)) & (ibone | mbone | lbone)] += dt
    
    newMaturity[(newMat == 2) & (neo)] = dt     # Fibrous
    newMaturity[(newMat == 3) & (neo)] = dt     # Cartiage
    newMaturity[((newMat == 5) | (newMat == 6) | (newMat == 7)) & (neo)] = dt     # Bone
    
    # Tissue type switching
    to_fibrous = (newMat == 2) & ~(fib) & ~(neo) & ~(prevMat == 7) & ~(prevMat == 4)
    to_cartilage = (newMat == 3) & ~(cart) & ~(neo) & ~(prevMat == 7) & ~(prevMat == 4)
    to_bone = ((newMat == 5) | (newMat == 6) | (newMat == 7)) & ~(ibone | mbone | lbone) & ~(neo) & ~(prevMat == 7) & ~(prevMat == 4)
    to_ibone = ((newMat == 5) ) & ~(ibone) & ~(neo) & ~(prevMat == 7) & ~(prevMat == 4)
    to_mbone = ((newMat == 6) ) & ~(mbone) & ~(neo) & ~(prevMat == 7) & ~(prevMat == 4)
    to_lbone = ((newMat == 7) ) & ~(lbone) & ~(neo) & ~(prevMat == 7) & ~(prevMat == 4)
    
    # Determine maturity (effective age) based on stiffness of previous tissue type
    newMaturity[to_fibrous] = np.log(prevMod[to_fibrous]/0.2)/(np.log(Efib/0.2)/60) + dt       # Fibrous
    newMaturity[to_cartilage] = np.log(prevMod[to_cartilage]/0.2)/(np.log(Ecart/0.2)/60) + dt  # Cartiage
    newMaturity[to_ibone] = np.log(prevMod[to_ibone]/0.2)/(np.log(Eibone/0.2)/60) + dt            # Immature Bone
    newMaturity[to_mbone] = np.log(prevMod[to_mbone]/0.2)/(np.log(Embone/0.2)/60) + dt            # Immature Bone
    newMaturity[to_lbone] = np.log(prevMod[to_lbone]/0.2)/(np.log(Elbone/0.2)/60) + dt            # Immature Bone
    
    if resorption:
        # Resorption - tissue maturity decreases
        newMat[newMat == -1] = prevMat[newMat == -1]
        newMaturity[newMat == -1] -= dt ### TODO: assumes resorption occurs at the same rate as tissue maturation, maybe not a sound assumption
    else:
        # Non-stiffening - tissue doesn't mature
        newMat[newMat == -1] = prevMat[newMat == -1]
        newMaturity[newMat == -1] = maturity[newMat == -1]

    return newMat, newMaturity

