"""
An asexual model, consisting of discrete-time equations to simulate the dynamics 
of a two-genotype model with St and Eu morphs. Companion to Lightfoot et al.
"""

__author__ = 'Ata Kalirad, Arne Traulsen & Stefano Giaimo'

import numpy as np

import sys

if sys.version_info[0] < 3:
    raise Exception("Python 3 or a more recent version is required.")

# Two-genotype version

def asexual_eqs_two_gen(xStJ, xStA, xEuJ, xEuA, yStJ, yStA, yEuJ, yEuA, k, f, lam, c, chi, dx, dy, bxy, rxy, bxx=1., byy=1.): 
    """Calculate the change in the discrete equations of the asexual model (two-genotype version) in Lightfoot et al. from t to t+1. 

    Parameters
    ----------
    xStJ : float
        The number of juveniles of genotype x with St morph at t.
    xStA : float
        The number of adults of genotype x with St morph at t.
    xEuJ : float
        The number of juveniles of genotype x with Eu morph at t.
    xEuA : float
        The number of adults of genotype x with Eu morph at t.
    yStJ : float
        The number of juvenile of genotype y with St morph at t.
    yStA : float
        The number of adults of genotype y with St morph at t.
    yEuJ : float
        The number of juvenile of genotype y with Eu morph at t.
    yEuA : float
        The number of adults of genotype y with Eu morph at t.
    k : float
        The carrying capacity.
    f : float
        The fecundity.
    lam : float
        The speed of development from juvenile to adult.
    c : float
        The cost of predation
    chi : float
        Kin recognition.
    dx : float
        The bias of genotype x to preduce Eu morph. 
    dy : float
        The bias of genotype y to preduce Eu morph. 
    bxy : float
        The probability of encounter between individuals of x and y genotypes.
    rxy : float
        The genomic relatedness between individuals of x and y genotypes.
    bxx : float, optional
        The probability of encounter between individuals of x genotype, by default 1.
    byy : float, optional
        The probability of encounter between individuals of y genotype, by default 1.

    Returns
    -------
    numpy.ndarray
        An array that includes the frequencies at t+1.
    """
    xStJ_1 = xStJ + f * (1. - dx) * (xStA + xEuA) - lam * xStJ - c * xStJ * (xEuA * bxx * (1. - chi) + yEuA * bxy * (1. - chi * rxy))
    xEuJ_1 = xEuJ + f * dx * (xStA + xEuA) - lam * xEuJ - c * xEuJ * (xEuA * bxx * (1. - chi) + yEuA * bxy * (1. - chi * rxy))
    yStJ_1 = yStJ + f * (1. - dy) * (yStA + yEuA) - lam * yStJ - c * yStJ * (xEuA * bxy * (1. - chi * rxy) + yEuA * byy * (1. - chi))
    yEuJ_1 = yEuJ + f * dy * (yStA + yEuA) - lam * yEuJ - c * yEuJ * (xEuA * bxy * (1. - chi * rxy) + yEuA * byy * (1. - chi))     
    xStA_1 = xStA + lam * xStJ - xStA * (f/k) * (xStA + xEuA + yStA + yEuA)
    xEuA_1 = xEuA + lam * xEuJ - xEuA * (f/k) * (xStA + xEuA + yStA + yEuA)
    yStA_1 = yStA + lam * yStJ - yStA * (f/k) * (xStA + xEuA + yStA + yEuA)
    yEuA_1 = yEuA + lam * yEuJ - yEuA * (f/k) * (xStA + xEuA + yStA + yEuA)
    return np.array([xStJ_1, xStA_1, xEuJ_1, xEuA_1, yStJ_1, yStA_1, yEuJ_1, yEuA_1])

def asexual_sim_two_gen(xStJ, xStA, xEuJ, xEuA, yStJ, yStA, yEuJ, yEuA, k, f, lam, c, chi, dx, dy, bxy, rxy, t):
    """Numerically solve the discrete-time equations of the asexual model.

    Parameters
    ----------
    t : int
        The number of steps.

    Returns
    -------
    dict
        A dictionary of time series for all the phenotypes.
    """
    xStJ_t_series = []
    xStA_t_series = []
    xEuJ_t_series = []
    xEuA_t_series = []
    yStJ_t_series = []
    yStA_t_series = []
    yEuJ_t_series = []
    yEuA_t_series = []
    for i in range(t+1):
        xStJ_t_series.append(xStJ)
        xStA_t_series.append(xStA)
        xEuJ_t_series.append(xEuJ)
        xEuA_t_series.append(xEuA)
        yStJ_t_series.append(yStJ)
        yStA_t_series.append(yStA)
        yEuJ_t_series.append(yEuJ)
        yEuA_t_series.append(yEuA)
        sim = asexual_eqs_two_gen(xStJ, xStA, xEuJ, xEuA, yStJ, yStA, yEuJ, yEuA, k, f, lam, c, chi, dx, dy, bxy, rxy)
        xStJ = sim[0] 
        xStA = sim[1] 
        xEuJ = sim[2] 
        xEuA = sim[3] 
        yStJ = sim[4] 
        yStA = sim[5] 
        yEuJ = sim[6] 
        yEuA = sim[7]
    data = {}
    data['xStJ'] = np.array(xStJ_t_series)
    data['xStA'] = np.array(xStA_t_series)
    data['xEuJ'] = np.array(xEuJ_t_series)
    data['xEuA'] = np.array(xEuA_t_series)
    data['yStJ'] = np.array(yStJ_t_series)
    data['yStA'] = np.array(yStA_t_series)
    data['yEuJ'] = np.array(yEuJ_t_series)
    data['yEuA'] = np.array(yEuA_t_series)
    return data

# Three-genotype version

def asexual_eqs_three_gen(xStJ, xStA, xEuJ, xEuA, yStJ, yStA, yEuJ, yEuA, zStJ, zStA, zEuJ, zEuA, k, f, lam, c, chi, dx, dy, dz, bxy, bxz, byz, rxy, rxz, ryz, bxx=1., byy=1., bzz=1.):
    """Calculate the change in the discrete equations of the asexual model (three-genotype version) in Lightfoot et al. from t to t+1. 

    Parameters
    ----------
    zStJ : float
        The number of juveniles of genotype z with St morph at t.
    zStA : float
        The number of adults of genotype z with St morph at t.
    zEuJ : float
        The number of juveniles of genotype z with Eu morph at t.
    zEuA : float
        The number of adults of genotype x with Eu morph at t.
    dz : float
        The bias of genotype z to preduce Eu morph.
    bxy : float
        The probability of encounter between individuals of x and y genotypes.
    bxz : float
        The probability of encounter between individuals of x and z genotypes.
    byz : float
        The probability of encounter between individuals of y and z genotypes.
    rxy : float
        The genomic relatedness between individuals of x and y genotypes.
    rxz : float
        The genomic relatedness between individuals of x and z genotypes.
    ryz : float
        The genomic relatedness between individuals of  and y genotypes.

    Returns
    -------
    numpy.ndarray
        An array that includes the frequencies at t+1.
    """
    xStJ_1 = xStJ + f * (1. - dx)*(xStA + xEuA) - lam * xStJ - c * xStJ * (xEuA * bxx * (1. - chi) + yEuA * bxy * (1. - chi * rxy)+ zEuA * bxz * (1. - chi * rxz))
    xEuJ_1 = xEuJ + f * dx * (xStA + xEuA) - lam * xEuJ - c * xEuJ * (xEuA * bxx * (1. - chi) + yEuA * bxy * (1. - chi * rxy) + zEuA * bxz * (1. - chi * rxz))
    yStJ_1 = yStJ + f * (1. - dy) * (yStA + yEuA) - lam * yStJ - c * yStJ * (xEuA * bxy * (1. - chi * rxy) + yEuA * byy * (1. - chi) + zEuA * byz * (1. - chi * ryz))
    yEuJ_1 = yEuJ + f * dy *(yStA + yEuA) - lam * yEuJ - c * yEuJ * (xEuA * bxy * (1. - chi * rxy) + yEuA * byy * (1. - chi)  + zEuA * byz * (1. - chi * ryz))  
    zStJ_1 = zStJ + f * (1. - dz) * (zStA + zEuA) - lam * zStJ - c * zStJ * (xEuA * bxz * (1. - chi * rxz) + yEuA * byz * (1. - chi * ryz) + zEuA * bzz * (1. - chi))
    zEuJ_1 = zEuJ + f * dz *(zStA + zEuA) - lam * zEuJ - c * zEuJ * (xEuA * bxz * (1. - chi * rxz) +  yEuA * byz * (1. - chi * ryz) + zEuA * bzz * (1. - chi))  
    xStA_1 = xStA + lam * xStJ - xStA * (f/k) * (xStA + xEuA + yStA + yEuA + zStA + zEuA)
    xEuA_1 = xEuA + lam * xEuJ - xEuA * (f/k) * (xStA + xEuA + yStA + yEuA + zStA + zEuA)
    yStA_1 = yStA + lam * yStJ - yStA * (f/k) * (xStA + xEuA + yStA + yEuA + zStA + zEuA)
    yEuA_1 = yEuA + lam * yEuJ - yEuA * (f/k) * (xStA + xEuA + yStA + yEuA + zStA + zEuA)
    zStA_1 = zStA + lam * zStJ - zStA * (f/k) * (xStA + xEuA + yStA + yEuA + zStA + zEuA)
    zEuA_1 = zEuA + lam * zEuJ - zEuA * (f/k) * (xStA + xEuA + yStA + yEuA + zStA + zEuA)
    
    return np.array([xStJ_1, xStA_1, xEuJ_1, xEuA_1, yStJ_1, yStA_1, yEuJ_1, yEuA_1, zStJ_1, zStA_1, zEuJ_1, zEuA_1])

def asexual_sim_three_gen(xStJ, xStA, xEuJ, xEuA, yStJ, yStA, yEuJ, yEuA, zStJ, zStA, zEuJ, zEuA, k, f, lam, c, chi, dx, dy, dz, bxy, bxz, byz, rxy, rxz, ryz, t):
    """Numerically solve the discrete-time equations of the asexual model.

    Parameters
    ----------
    t : int
        The number of steps.

    Returns
    -------
    dict
        A dictionary of time series for all the phenotypes.
    """
    xStJ_t_series = []
    xStA_t_series = []
    xEuJ_t_series = []
    xEuA_t_series = []
    yStJ_t_series = []
    yStA_t_series = []
    yEuJ_t_series = []
    yEuA_t_series = []
    zStJ_t_series = []
    zStA_t_series = []
    zEuJ_t_series = []
    zEuA_t_series = []
    for i in range(t+1):
        xStJ_t_series.append(xStJ)
        xStA_t_series.append(xStA)
        xEuJ_t_series.append(xEuJ)
        xEuA_t_series.append(xEuA)
        yStJ_t_series.append(yStJ)
        yStA_t_series.append(yStA)
        yEuJ_t_series.append(yEuJ)
        yEuA_t_series.append(yEuA)
        zStJ_t_series.append(zStJ)
        zStA_t_series.append(zStA)
        zEuJ_t_series.append(zEuJ)
        zEuA_t_series.append(zEuA)
        sim = asexual_eqs_three_gen(xStJ, xStA, xEuJ, xEuA, yStJ, yStA, yEuJ, yEuA, zStJ, zStA, zEuJ, zEuA, k, f, lam, c, chi, dx, dy, dz, bxy, bxz, byz, rxy, rxz, ryz)
        xStJ = sim[0] 
        xStA = sim[1] 
        xEuJ = sim[2] 
        xEuA = sim[3] 
        yStJ = sim[4] 
        yStA = sim[5] 
        yEuJ = sim[6] 
        yEuA = sim[7]
        zStJ = sim[8] 
        zStA = sim[9] 
        zEuJ = sim[10] 
        zEuA = sim[11]
    data = {}
    data['xStJ'] = np.array(xStJ_t_series)
    data['xStA'] = np.array(xStA_t_series)
    data['xEuJ'] = np.array(xEuJ_t_series)
    data['xEuA'] = np.array(xEuA_t_series)
    data['yStJ'] = np.array(yStJ_t_series)
    data['yStA'] = np.array(yStA_t_series)
    data['yEuJ'] = np.array(yEuJ_t_series)
    data['yEuA'] = np.array(yEuA_t_series)
    data['zStJ'] = np.array(zStJ_t_series)
    data['zStA'] = np.array(zStA_t_series)
    data['zEuJ'] = np.array(zEuJ_t_series)
    data['zEuA'] = np.array(zEuA_t_series)
    return data