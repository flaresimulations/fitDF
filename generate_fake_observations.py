
import os

import fitLF.models as models
import fitLF.analyse as analyse

import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle

import mpmath

plt.style.use('simple')


plt.style.use('simple')
fig = plt.figure(figsize=(3,3))
ax = fig.add_axes([0.15, 0.15, 0.8, 0.75 ])








# ID = 'test_2volumes'
ID = 'test_1volumes'

if not os.path.exists(ID): os.mkdir(ID)

np.random.seed(5)


# -------------------- define input LF

parameters = {'log10L*': 28.5, 'log10phi*': -2., 'alpha': -2.5}
LF = models.Schechter(parameters)

pickle.dump(parameters, open(ID+'/input_parameters.p','w'))





log10L_limits = [28.0, 29.0]
volumes = [(50.)**3, (500.)**3]

log10L_limits = [28.0]
volumes = [(50.)**3]


observations = []

mxphi = -100.

for log10L_limit, volume in zip(log10L_limits,volumes):


    logV = np.log10(volume)

    print '------------', log10L_limit, volume

    # -------------------- sample input LF and plot

    sample = LF.sample(volume, log10L_limit) # --- sample the IMF from log10L_limit -> \infty 
    
    observations.append({'log10L_limit':log10L_limit, 'volume':volume, 'sample': sample})

    # -------------------- this bins that sampled IMF and compares to the truth

    binw = 0.1
    bin_edges = np.arange(log10L_limit, log10L_limit+1.5, binw)
    bin_centres = bin_edges[:-1] + 0.5*(bin_edges[1:]-bin_edges[:-1]) 

    N = LF.N(volume, bin_edges) # --- the exact number of galaxies expected in each bin

    N_sample = models.bin(sample, volume, bin_edges) # --- bin the sampled LF with the same bins 

    for bc,n,n_sample in zip(bin_centres, N, N_sample): print bc, n, n_sample

    # -------------------- plot sampled and true LF

    


    c = np.random.rand(3,)

    # --- plot "true" LF

    ax.plot(bin_centres, np.log10(N) - np.log10(binw) - logV, c=c, lw=3, alpha = 0.2)

    # --- plot sampled LF values with poisson confidence intervals   

    for bc, n in zip(bin_centres, N_sample): 

        print n, models.poisson_confidence_interval(n, 0.68)

        if n>0:
            ax.plot([bc]*2, np.log10(models.poisson_confidence_interval(n, 0.68)) -np.log10(binw) - logV, c=c, lw=1, alpha = 1.0) 
        else:
            ax.arrow(bc, np.log10(models.poisson_confidence_interval(n, 0.68))[1] -np.log10(binw) - logV, 0.0, -0.5, color=c)


    phi = np.log10(N_sample) - np.log10(binw) - logV

    ax.scatter(bin_centres, phi, s=5, c=c, lw=1, alpha = 1.0)

    if np.max(phi)>mxphi: mxphi = np.max(phi)


pickle.dump(observations, open(ID+'/fake_observations.p', 'w')) # --- save sampled LFs






# --- plot input parameters

ax.axvline(parameters['log10L*'], c='k', alpha = 0.1)
ax.axhline(parameters['log10phi*'] + np.log10(volume), c='k', alpha = 0.1)


ax.set_ylim([np.log10(1./np.max(np.array(volumes)))-0.5, mxphi+0.5])

ax.set_xlabel(r"$\rm \log_{10}(L_{\nu}/erg\, s^{-1}\, Hz^{-1})$")
ax.set_ylabel(r"$\rm \log_{10}(\phi/{\rm Mpc^{-3}})$")


fig.savefig(ID+'/inputLF.pdf', dpi = 300)
    


