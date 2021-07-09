# -*- coding: utf-8 -*-
"""


@author: TomClifford
"""

#functions and classes come from https://github.com/GEMScienceTools/gmpe-smtk/blob/master/smtk/response_spectrum.py

#%%
import numpy as np

#
def response_spectrum(acc_g, dt, periods, damping=0.05, newmark_scheme='average'):
    """    
    Returns the response spectra for acceleration, velocity, displacement, pseudo-acceleration, and pseuso-velocity
    
	Parameters
	----------
    acc_g : floats in a list or Numpy.array
        [m/sec^2] acceleration time series for the ground motion
    dt : float
        [sec] time step
    periods : floats in a list or Numpy.array
        [sec] periods to evaluate the response spectra
    damping : float
        damping ratio, **default = 0.05**
    newmark_scheme : str, *optional*
        scheme for Newmark-beta method; available options are **average (default)**, constant, and central
    
	Returns
	-------
    ars : floats in an Numpy.array
        [m/sec^2] acceleration response spectrum
    vrs : floats in an Numpy.array
        [m/sec] velocity response spectrum
    drs : floats in an Numpy.array
        [m] displacement response spectrum
    psa : floats in an Numpy.array
        [m/sec^2] pseudo-acceleration response spectrum
    psv : floats in an Numpy.array
        [m/sec] pseudo-acceleration response spectrum
    
    """
    
    # preprocess
    num_per = len(periods) # number of periods
    num_steps = len(acc_g) # number of steps in record
    acc_g = np.asarray(acc_g) # convert ground motion to Numpy array
    periods = np.asarray(periods) # convert list of periods to Numpy array
    # acc_g = acc_g * g # convert to m/sec^2
    
    # system parameters
    omega = 2*np.pi/periods # rad/sec, angular frequency
    m = np.ones(num_per) # kg, mass, assumed to be 1 kg
    c = damping*2*m*omega # N/(m/sec), dashpot coefficient
    k = omega**2*m # N/m or kg/sec^2, spring stiffness
    
    # basic schemes for Newmark-Beta method
    # 1: (implicit) average constant acceleration - gamma = 1/2, beta = 1/4
    # 2: (implicit) linear acceleration - gamma = 1/2, beta = 1/6
    # 3: (explicit) central difference - gamma = 1/2, beta = 0
    # 
    # note: gamma applies to velocity, and beta applies to displacement
    gamma = 1/2
    if 'average' in newmark_scheme.lower():
        beta = 1/4
    elif 'linear' in newmark_scheme.lower():
        beta = 1/6
    elif 'central' in newmark_scheme.lower():
        beta = 0
    else:
        beta = 1/4 # default to average constant acceleration
    
    # Perform Newmark - Beta integration
    # Pre-allocate arrays
    acc_rel = np.zeros([num_steps, num_per]) # relative acceleration for SDOF 
    vel = np.zeros([num_steps, num_per]) # relative velocity for SDOF
    disp = np.zeros([num_steps, num_per]) # relative displacement for SDOF
    acc_total = np.zeros([num_steps, num_per]) # absolute acceleration for SDOF
    # first time step
    # initial velocities and displacements = zero
    acc_rel[:,0] = -acc_g[0] # initial acceleration = ground motion
    # Initial line
    for i in range(1, num_steps):
        # for solving relative acceleration
        numer = -m*acc_g[i] - c*(vel[i-1,:]+(1-gamma)*acc_rel[i-1,:]*dt) - k*(disp[i-1,:]+vel[i-1,:]*dt+(1-2*beta)/2*acc_rel[i-1,:]*dt**2)
        denom = m + c*gamma*dt + k*beta*dt**2
        # solve system response
        acc_rel[i,:] = numer/denom # m/sec^2
        vel[i,:] = vel[i-1,:] + (1-gamma)*acc_rel[i-1,:]*dt + gamma*acc_rel[i,:]*dt # m/sec
        disp[i,:] = disp[i-1,:] + vel[i-1,:]*dt + (1-2*beta)/2*acc_rel[i-1,:]*dt**2 + beta*acc_rel[i,:]*dt**2 # m
        acc_total[i,:] = acc_g[i] + acc_rel[i,:]
            
    # construct return dictionary
    spectra = {
        'Acceleration': np.max(np.fabs(acc_total), axis=0), # m/sec^2
        'Velocity': np.max(np.fabs(vel), axis=0), # m/sec
        'Displacement': np.max(np.fabs(disp), axis=0) # m
    }
    spectra['Pseudo-Velocity'] = omega*spectra['Displacement'] # m/sec
    spectra['Pseudo-Acceleration'] = omega**2*spectra['Displacement'] # m/sec^2
    
    #
    return spectra