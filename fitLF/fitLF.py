#!/usr/bin/env python

import numpy as np
import emcee
import scipy.stats
import pickle
from . import models
import scipy.misc
import math

class fitter():


    def __init__(self, observations, ID = 'test'):


        self.ID = ID
        self.observations = observations

        pickle.dump(observations, open(self.ID+'/observations.p', 'wb'))

        print('fitLFv0.8')
                 
        self.parameters = ['log10phi*','alpha','log10L*']

        # ----- define priors
        
        self.priors = {}
        
        # This distribution is constant between loc and loc + scale.
        
        self.priors['log10phi*'] = scipy.stats.uniform(loc = -7.0, scale = 7.0) 
        self.priors['alpha'] = scipy.stats.uniform(loc = -3.0, scale = 3.0) 
        self.priors['log10L*'] = scipy.stats.uniform(loc = 26., scale = 5.0) 



    def lnprob(self, params):

        p = {parameter:params[i] for i,parameter in enumerate(self.parameters)}
    
        model_LF = models.Schechter(p)

        lp = np.sum([self.priors[parameter].logpdf(p[parameter]) for parameter in self.parameters])
           
        if not np.isfinite(lp):
            return -np.inf
        
        lnlike = 0
        
        for obs in self.observations:
    
            N_exp = model_LF.N(obs['volume'], obs['bin_edges'])

            s = N_exp>0. # technically this should always be true but may break at very low N hence this 

            lnlike += np.sum(obs['N'][s] * np.log(N_exp[s]) - N_exp[s]) 
    
        return lp + lnlike
    
    
    def fit(self, nwalkers = 50, nsamples = 1000, burn = 200, sample_save_ID = 'samples'):
    
        print('Fitting -------------------------')
    
        # --- define number of parameters   
        self.ndim = 3
          
        # --- Choose an initial set of positions for the walkers.
        p0 = [ [self.priors[parameter].rvs() for parameter in self.parameters] for i in range(nwalkers)]

        # --- Initialize the sampler with the chosen specs. The "a" parameter controls the step size, the default is a=2.

        self.sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.lnprob, args=())
        
        pos, prob, state = self.sampler.run_mcmc(p0, burn)
        self.sampler.reset()
        
        self.sampler.run_mcmc(pos, nsamples)

        # --- save samples
    
        samples = {}
    
        chains = self.sampler.chain[:, :, ].reshape((-1, self.ndim))
    
        for ip, p in enumerate(self.parameters): 
        
            samples[p] = chains[:,ip]

        pickle.dump(samples, open(self.ID+'/'+sample_save_ID+'.p', 'wb'))
        
        return samples


