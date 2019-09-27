#!/usr/bin/env python

import numpy as np
import emcee
import scipy.stats
import pickle
from . import models
import scipy.misc
#import math


class fitter():

    def __init__(self, observations, model, priors, output_directory = 'test'):

        print('fitDFv0.9')

        # TODO: input tests
        self.output_directory = output_directory
        self.observations = observations
        self.model = model
        self.priors = priors
        self.parameters = priors.keys()


    def lnprob(self, params):

        p = {parameter:params[i] for i,parameter in enumerate(self.parameters)}
    
        self.model.update_params(p)

        lp = np.sum([self.priors[parameter].logpdf(p[parameter]) for parameter in self.parameters])
           
        if not np.isfinite(lp):
            return -np.inf
        
        lnlike = 0.
        
        for obs in self.observations:
            
            ## expected number of objects from model
            N_exp = self.model.N(obs['volume'], obs['bin_edges'])

            s = N_exp>0. # technically this should always be true but may break at very low N hence this 

            lnlike += np.sum(obs['N'][s] * np.log(N_exp[s]) - N_exp[s]) 
    
        return lp + lnlike
    
    
    def fit(self, nwalkers = 50, nsamples = 1000, burn = 200, sample_save_ID = 'samples'):
    
        print('Fitting -------------------------')
    
        # --- define number of parameters   
        self.ndim = len(self.priors.keys())
          
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

        pickle.dump(samples, open(self.output_directory+'/'+sample_save_ID+'.p', 'wb'))
        
        return samples


