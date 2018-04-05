import pickle
import numpy as np
import utils
import os
import pymultinest
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad
import matplotlib.pyplot as plt 
# Functions to model stellar spectra at any temperature:
def get_specs():
    temperatures = np.array([])
    counter = 0
    for t in ['6500','6250','6000','5750','5500','5250','5000','4750','4500','4250','4000']:
        ww,ff = np.loadtxt('stellar_models/'+t+'.dat',unpack=True)
        idx = np.where((ww>3000)&(ww<10000))[0]
        if counter == 0:
            wavs = ww[idx]
            fluxes = ff[idx]
            counter = 1
        else:
            fluxes = np.vstack((fluxes,ff[idx]))
        temperatures = np.append(temperatures,np.double(int(t)))
    return wavs,fluxes,temperatures

def spec_get(w,f,T,Ttarget):
    target_flux = np.zeros(len(w))
    for i in range(len(w)):
        func = interp1d(T,f[:,i])
        target_flux[i] = func(Ttarget) 
    return target_flux

wavs,fluxes,temperatures = get_specs()

# Get contrasts:
wlow,wup,cs,cserr = np.loadtxt('contrasts.dat',unpack=True,usecols=(0,1,2,3))
ws = (wlow+wup)/2.
ws_err = ws-wlow
csup = cserr
csdown = cserr

def model(T):
    stellar_flux = np.zeros(len(ws))
    f = spec_get(wavs,fluxes,temperatures,T)
    s = UnivariateSpline(wavs,f)
    for i in range(len(ws)):
        delta_wav = (ws[i]+ws_err[i]) - (ws[i]-ws_err[i])
        stellar_flux[i] = s.integral(ws[i]-ws_err[i],ws[i]+ws_err[i])/delta_wav
    return stellar_flux

# Define the prior (you have to transform your parameters, that come from the unit cube,
# to the prior you want):
def prior(cube, ndim, nparams):
    # Prior on Temperature of spot:
    cube[0] = utils.transform_uniform(cube[0],4000,6500)
    # Prior on temperature of WASP-19:
    cube[1] = utils.transform_normal(cube[1],5460,90)

def loglike(cube, ndim, nparams):
    # Extract parameters:
    T,Tw19 = cube[0],cube[1]
    if T>Tw19:
        return -np.inf
    # Evaluate the log-likelihood:
    loglikelihood = (-0.5*np.log(2.*np.pi*cserr**2) + (-0.5 * ((model(T)/model(Tw19) - cs) / cserr)**2)).sum()
    return loglikelihood

"""
Ts = range(4010,6500,10)
llike = np.zeros(len(Ts))
for i in range(len(Ts)):
    print Ts[i]
    llike[i] = (-0.5*np.log(2.*np.pi*cserr**2) + (-0.5 * ((model(Ts[i])/mw19 - cs) / cserr)**2)).sum()
    
plt.plot(Ts,llike)
plt.show()
"""

n_params = 2
out_file = 'out_mnest'

# Run MultiNest:
pymultinest.run(loglike, prior, n_params, n_live_points = 500,outputfiles_basename=out_file, resume = False, verbose = True)
# Get output:
output = pymultinest.Analyzer(outputfiles_basename=out_file, n_params = n_params)

print 'Psampling:'
# Get out parameters: this matrix has (samples,n_params+1):
psamples = output.get_equal_weighted_posterior()[:,:-1]
"""
T,Tw19 = psamples[-50:,0],psamples[-50:,1]
for i in range(len(T)):
    plt.plot(ws, model(T[i])/model(Tw19[i]), '-', color='blue', alpha=0.01, label='data')
plt.errorbar(ws,cs,yerr=cserr,fmt='.',color='black')
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('$F_s/F_*$')
plt.show()
"""
# Save the log-evidence:
a_lnZ = output.get_stats()['global evidence']
# Save the error on the log-evidence:
a_lnZ_err = output.get_stats()['global evidence error']
#plt.xlabel('Wavelength (Angstroms)')
#plt.ylabel('$F_s/F_*$')
"""
print '(ln) Evidence: ',a_lnZ,'+-',a_lnZ_err
plt.hist(T,bins=30,label='a',normed=True)
plt.show()
"""
out = {}
out['T'] = psamples[:,0]
out['Tw19'] = psamples[:,1]
out['lnZ'] = a_lnZ
out['lnZerr'] = a_lnZ_err
fout = open('out_mnest.pkl','w')
pickle.dump(out,fout)
fout.close()
