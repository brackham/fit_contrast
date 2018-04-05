import pickle
import matplotlib
import numpy as np
import utils
import os
import pymultinest
import seaborn as sns
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

def nonbin_model(T):
    stellar_flux = np.zeros(len(ws))
    f = spec_get(wavs,fluxes,temperatures,T)
    return f

sns.set_context("talk")
sns.set_style("ticks")
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['axes.linewidth'] = 1.2 
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rcParams['lines.markeredgewidth'] = 1 

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)

data_kwargs = {"zorder":100}
model_kwargs = {"zorder":0}

out = pickle.load(open('out_mnest.pkl','r'))
T,Tw19 = out['T'], out['Tw19']#[-500:,0],psamples[-500:,1]
for i in range(len(T)):
    if i == 0:
        themodel = nonbin_model(T[i])/nonbin_model(Tw19[i])
    else:
        themodel = np.vstack((themodel,nonbin_model(T[i])/nonbin_model(Tw19[i])))

mean_model = np.median(themodel,axis=0)
sigma_model = np.sqrt(np.var(themodel,axis=0))
plt.plot(wavs, mean_model, '-', color='grey', linewidth=1.5, label='data',**model_kwargs)
plt.fill_between(wavs, mean_model + 5*sigma_model, mean_model - 5*sigma_model, color='black', alpha=0.25,edgecolor="none")

print 'Spot temperature:',np.median(T),np.sqrt(np.var(T))
print 'WASP-19 temperature:',np.median(Tw19),np.sqrt(np.var(Tw19))
print 'Delta T',np.median(T-Tw19),np.sqrt(np.var(T-Tw19))
pal = sns.diverging_palette(250, 15, s=99, l=50,center='dark', n=4)
ax.set_prop_cycle('color',pal)

Tw19 = np.median(out['Tw19'])
for T in [5080,5180,5380,5450][::-1]:
        plt.plot(wavs, nonbin_model(T)/nonbin_model(Tw19), '-', linewidth=1, label='data',alpha=0.5,**model_kwargs)

#plt.plot(wavs,np.ones(len(wavs)),'--',color='grey',linewidth=1)
plt.errorbar(ws,cs,yerr=cserr,fmt='o',markersize=7,markeredgecolor='black',markerfacecolor='white',ecolor='black',elinewidth=1,**data_kwargs)

fsize = 8
plt.text(6800,0.715,'5080 K',fontsize=fsize,rotation=10)
plt.text(6800,0.78,'5180 K',fontsize=fsize,rotation=6)
plt.text(6800,0.915,'5380 K',fontsize=fsize,rotation=1)
plt.text(6800,0.965,'5450 K',fontsize=fsize)
plt.xlim([4000,9500])
plt.ylim([0.65,1.01])
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Spot contrast')
plt.tight_layout()
#plt.tight_layout()
plt.savefig('plot.pdf')
