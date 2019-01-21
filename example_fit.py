
import fitDF.fitDF as fitDF
import fitDF.models as models
import fitDF.analyse as analyse

import numpy as np
import matplotlib.pyplot as plt
import pickle

# plt.style.use('simple')

ID = 'example'

# -------------------- read in observations

fake_observations = pickle.load(open(ID+'/fake_observations.p','rb')) # list of dictionaries containing log10L_limit, volume, sample


observations = [] # fit LF input list

binw = 0.1

for fake_obs in fake_observations:

    print(fake_obs)

    bin_edges = np.arange(fake_obs['log10L_limit'], fake_obs['log10L_limit']+1.5, binw)

    N_sample = models.bin(fake_obs['sample'], fake_obs['volume'], bin_edges)

    observations.append({'bin_edges': bin_edges, 'N': N_sample, 'volume': fake_obs['volume']})


print(observations)

# -------------------- fit sampled LF and plot median fit
import scipy
priors = {}
priors['phi*'] = scipy.stats.uniform(loc = -7.0, scale = 7.0) 
priors['alpha'] = scipy.stats.uniform(loc = -3.0, scale = 3.0) 
priors['D*'] = scipy.stats.uniform(loc = 26., scale = 5.0) 

model = models.Schechter()

fitter = fitDF.fitter(observations, model=model, priors=priors, output_directory = ID)
fitter.fit(nsamples = 50, burn = 10)
# fitter.fit(nsamples = 2000, burn = 500, sample_save_ID = 'a_different_ID') # to save the samples as something other than samples.p


# -------------------- make simple analysis plots


# ranges controls the range shown on the triangle plot. If set to False it calculates a range based on the sample data. 

# ranges = {'log10phi*': [-6.0, 2.0], 'alpha': [-4.0, 0.0], 'log10L*': [27., 33.]}
# ranges = False
# 
# a = analyse.analyse(ID = ID)
# a.triangle(hist2d = True, ccolor='0.5', ranges = ranges)
# a.LF()
