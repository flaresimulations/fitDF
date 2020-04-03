#!/usr/bin/env python

import numpy as np
import emcee
import scipy.stats
import json
from . import models
import scipy.misc


class fitter():

    def __init__(self, observations, model, priors, output_directory = 'test', penalty = 'False'):

        print('fitDFv0.1')

        # TODO: input tests
        self.output_directory = output_directory
        self.observations = observations
        self.model = model
        self.priors = priors
        self.parameters = priors.keys()
        self.penalty = penalty


    def gaussian_lnlikelihood(self, observed, expected, sigma):
       
        output = -0.5 * np.sum((observed - expected) ** 2 / sigma**2 + np.log(sigma**2))
         
        return output
    
    
    def poissonian_lnlikelihood(self, observed, expected):
        
        output = np.nansum(observed * np.log(expected) - expected - (observed+0.5)*np.log(observed))
        
        if self.penalty:
        
            output += np.nansum((np.log10(expected) - np.log10(observed))**2)
            
        return output


    def lnprob(self, params, lnlikelihood):

        p = {parameter:params[i] for i,parameter in enumerate(self.parameters)}

        self.model.update_params(p)

        lp = np.sum([self.priors[parameter].logpdf(p[parameter]) for parameter in self.parameters])

        if not np.isfinite(lp):
            return -np.inf

        lnlike = 0.

        for obs in self.observations:

            ## expected number of objects from model
            N_exp = self.model.N(obs['volume'], obs['bin_edges'])

            s = np.logical_and(N_exp>0., obs['N']>0.) # technically this should always be true but may break at very low N hence this

            if 'sigma' in obs.keys():
                lnlike += lnlikelihood(obs['N'][s], N_exp[s], obs['sigma'][s])
            else:
                lnlike += lnlikelihood(obs['N'][s], N_exp[s])

        return lp + lnlike


    def fit(self, nwalkers = 50, nsamples = 1000, 
            burn = 200, sample_save_ID = 'samples', lnlikelihood=None):
        
        # set the log-likelihood method
        if lnlikelihood is None:
            lnlikelihood = self.poissonian_lnlikelihood

        print('Fitting -------------------------')

        # --- define number of parameters
        self.ndim = len(self.priors.keys())

        # --- Choose an initial set of positions for the walkers.
        p0 = [ [self.priors[parameter].rvs() for parameter in self.parameters] for i in range(nwalkers)]

        # --- Initialize the sampler with the chosen specs. The "a" parameter controls the step size, the default is a=2.

        self.sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.lnprob, kwargs={'lnlikelihood': lnlikelihood})

        pos, prob, state = self.sampler.run_mcmc(p0, burn)
        self.sampler.reset()

        self.sampler.run_mcmc(pos, nsamples)

        # --- save samples

        samples = {}

        chains = self.sampler.chain[:, :, ].reshape((-1, self.ndim))

        for ip, p in enumerate(self.parameters):

            samples[p] = chains[:,ip]

        self.save_samples(samples,sample_save_ID)

        return samples


    def save_samples(self, samples, save_ID):
        
        samples = {key: arr.tolist() for key,arr in samples.items()}

        with open('%s/%s.json'%(self.output_directory,save_ID),"w") as f:
            json.dump(samples,f)
        

