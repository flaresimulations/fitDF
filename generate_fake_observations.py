import os
import numpy as np
import matplotlib.pyplot as plt
import json

import fitDF.models as models
import fitDF.analyse as analyse


fig = plt.figure(figsize=(3,3))
ax = fig.add_axes([0.15, 0.15, 0.8, 0.75 ])

ID = 'example'
if not os.path.exists(ID): os.mkdir(ID)

np.random.seed(2)

# -------------------- define input function
parameters = {'D*': 28.5, 'log10phi*_1': -1, 'alpha_1': 0.5,
                          'log10phi*_2': -2.5, 'alpha_2': -2.5}
LF = models.DoubleSchechter(parameters)


with open('%s/input_parameters.json'%ID,'w') as f:
    json.dump(parameters,f)

log10L_limits = [27.0]
volumes = [(50.)**3]

observations = {}

mxphi = -100.

for log10L_limit, volume in zip(log10L_limits,volumes):


    logV = np.log10(volume)

    print('------------', log10L_limit, volume)

    # -------------------- sample input LF and plot
    # --- sample the IMF from log10L_limit -> \infty 
    sample = models.sample(LF, volume=volume, D_lowlim=log10L_limit) 
    
    observations[log10L_limit] = {'log10L_limit':log10L_limit, 'volume':volume, 'sample': sample}

    # -------------------- bin the sampled IMF and compares to the truth
    binw = 0.05
    bin_edges = np.arange(log10L_limit, log10L_limit+3, binw)
    bin_centres = bin_edges[:-1] + 0.5*(bin_edges[1:]-bin_edges[:-1]) 

    N = LF.N(volume, bin_edges) # --- the exact number of galaxies expected in each bin

    N_sample = models.bin(sample, volume, bin_edges) # --- bin the sampled LF with the same bins 

    # -------------------- plot sampled and true LF
    # --- plot "true" LF
    c = np.random.rand(3,)
    ax.plot(bin_centres, np.log10(N) - np.log10(binw) - logV, c=c, lw=3, alpha = 0.2)

    # --- plot sampled LF values with poisson confidence intervals   
    for bc, n in zip(bin_centres, N_sample): 
        print(n, models.poisson_confidence_interval(n, 0.68))

        if n>0:
            ax.plot([bc]*2, np.log10(models.poisson_confidence_interval(n, 0.68)) -np.log10(binw) - logV, c=c, lw=1, alpha = 1.0) 
        else:
            ax.arrow(bc, np.log10(models.poisson_confidence_interval(n, 0.68))[1] -np.log10(binw) - logV, 0.0, -0.5, color=c)


    phi = np.log10(N_sample) - np.log10(binw) - logV
    ax.scatter(bin_centres, phi, s=5, c=c, lw=1, alpha = 1.0)

    if np.max(phi)>mxphi: mxphi = np.max(phi)



# convert arrays to lists for json serialization
for l in observations.keys():
    observations[l]['sample'] = observations[l]['sample'].tolist() 

with open('%s/fake_observations.json'%ID,'w') as f:
    json.dump(observations,f)

# --- plot input parameters
ax.axvline(parameters['D*'], c='k', alpha = 0.1)
ax.axhline(parameters['log10phi*_1'], c='k', alpha = 0.1)
ax.axhline(parameters['log10phi*_2'], c='k', alpha = 0.1)

ax.set_ylim([np.log10(1./np.max(np.array(volumes)))-0.5, mxphi+0.5])

ax.set_xlabel(r"$\rm \log_{10}(L_{\nu}/erg\, s^{-1}\, Hz^{-1})$")
ax.set_ylabel(r"$\rm \log_{10}(\phi/{\rm Mpc^{-3}})$")

plt.show()
fig.savefig(ID+'/inputLF.pdf', dpi = 300)

