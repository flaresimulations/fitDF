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
    bin_edges = np.arange(fake_obs['log10L_limit'], fake_obs['log10L_limit']+3, binw)
    N_sample = models.bin(fake_obs['sample'], fake_obs['volume'], bin_edges)
    observations.append({'bin_edges': bin_edges, 'N': N_sample, 'volume': fake_obs['volume']})


# ----- Define Priors manually...
priors = {}

model = models.DoubleSchechter()

priors['log10phi*_1'] = scipy.stats.uniform(loc = -5.0, scale = 6.0) 
priors['alpha_1'] = scipy.stats.uniform(loc = -1.0, scale = 3.0) 
priors['log10phi*_2'] = scipy.stats.uniform(loc = -5.0, scale = 4.0) 
priors['alpha_2'] = scipy.stats.uniform(loc = -4.0, scale =3.0) 
priors['D*'] = scipy.stats.uniform(loc = 26., scale = 5.0) 


# -------------------- fit sampled LF and plot median fit
fitter = fitDF.fitter(observations, model=model, priors=priors, output_directory = ID)
fitter.fit(nsamples = 800, burn = 1000)

# -------------------- make simple analysis plots
a = analyse.analyse(ID = ID, model=model, observations=observations)
fig = a.triangle(hist2d = True, ccolor='0.5')
plt.show()
fig.savefig('%s/triangle.png'%ID,dpi=150,bbox_inches='tight')

fig = a.LF(bins=np.arange(27,33,0.01), observations=True)
plt.show()
fig.savefig('%s/LF.png'%ID,dpi=150,bbox_inches='tight')


