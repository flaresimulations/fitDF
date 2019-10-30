import fitDF.fitDF as fitDF
import fitDF.models as models
import fitDF.analyse as analyse

import numpy as np
import matplotlib.pyplot as plt
import scipy
import json

ID = 'example'

# -------------------- read in observations

with open('%s/fake_observations.json'%ID,'r') as f:
    fake_observations = json.load(f)


observations = [] # fit LF input list

binw = 0.1

for k,fake_obs in fake_observations.items():
    bin_edges = np.arange(fake_obs['log10L_limit'], fake_obs['log10L_limit']+1.5, binw)
    N_sample = models.bin(fake_obs['sample'], fake_obs['volume'], bin_edges)
    observations.append({'bin_edges': bin_edges, 'N': N_sample, 'volume': fake_obs['volume']})


model = models.Schechter()

# ----- Define Priors manually...
priors = {}
priors['log10phi*'] = scipy.stats.uniform(loc = -7.0, scale = 7.0) 
priors['alpha'] = scipy.stats.uniform(loc = -3.0, scale = 3.0) 
priors['D*'] = scipy.stats.uniform(loc = 26., scale = 5.0) 

# ---- ...or use utility function in fitDF
priors = models.LF_priors()

# -------------------- fit sampled LF and plot median fit
fitter = fitDF.fitter(observations, model=model, priors=priors, output_directory = ID)
fitter.fit(nsamples = 600, burn = 250)

# -------------------- make simple analysis plots
a = analyse.analyse(ID = ID, model=model, observations=observations)
fig = a.triangle(hist2d = True, ccolor='0.5')
fig.savefig('%s/triangle.png'%ID,dpi=150,bbox_inches='tight')

fig = a.LF(bins=np.arange(27,30,0.01), observations=True)
fig.savefig('%s/LF.png'%ID,dpi=150,bbox_inches='tight')


