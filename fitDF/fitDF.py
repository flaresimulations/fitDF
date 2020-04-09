#!/usr/bin/env python

import numpy as np
import emcee
import scipy.stats
import json
from . import models
import scipy.misc


class fitter():

    def __init__(self, observations, model, priors, output_directory = 'test', penalty = 'False'):

        print('fitDFv0.2')

        # TODO: input tests
        self.output_directory = output_directory
        self.observations = observations
        self.model = model
        self.priors = priors
        self.parameters = priors.keys()
        self.penalty = penalty
        self.lnlikelihood = self.poissonian_lnlikelihood


    def gaussian_lnlikelihood(self, observed, expected, sigma):
       
        output = -0.5 * np.sum((observed - expected) ** 2 / sigma**2 + np.log(sigma**2))
         
        return output
    
    
    def poissonian_lnlikelihood(self, observed, expected):
        
        output = np.nansum(observed * np.log(expected) - expected - (observed+0.5)*np.log(observed))
        
        if self.penalty:
        
            output += np.nansum((np.log10(expected) - np.log10(observed))**2)
            
        return output


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

            s = np.logical_and(N_exp>0., obs['N']>0.) # technically this should always be true but may break at very low N hence this

            if 'sigma' in obs.keys():
                lnlike += self.lnlikelihood(obs['N'][s], N_exp[s], obs['sigma'][s])
            else:
                lnlike += self.lnlikelihood(obs['N'][s], N_exp[s])

        return lp + lnlike


    def fit(self, nwalkers = 50, nsamples = 1000, 
            burn = 200, thin = 15, sample_save_ID = 'samples', 
            use_autocorr=False, verbose=False):
        """
        Run sampler

        Args:
        nwalkers (int)
        samples (int)
        burn (int): manually set burn-in  
        thin (int): manually set thinning
        sample_save_ID (str)
        use_autocorr (bool): use autocorrelation time to set burn and thin parameters
        verbose (bool)
        """

        if verbose: print('Fitting -------------------------')

        # --- define number of parameters
        self.ndim = len(self.priors.keys())

        # --- Choose an initial set of positions for the walkers.
        p0 = np.asarray([ [self.priors[parameter].rvs() for parameter in self.parameters] for i in range(nwalkers)])

        self.sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.lnprob)

        if verbose:
            self.sampler.run_mcmc(p0, nsamples, progress=True)
        else:
            self.sampler.run_mcmc(p0, nsamples)


        try:
            _autocorr = self.sampler.get_autocorr_time()
            if verbose: print("Autocorrelation time:", _autocorr)
        except Exception as e:
            print(e)
            _autocorr = None
        
        if (_autocorr is not None) & use_autocorr:
            if verbose: print("Using autocorrelation time to set burn-in and thinning")
            burn = int(np.max(_autocorr) * 5)
            thin = int(np.max(_autocorr) / 2)
            if verbose: print("burn:",burn,"\nthin:",thin)
        else:
            if verbose: print("burn:",burn,"\nthin:",thin)

        # --- save samples

        samples = {}

        chains = self.sampler.get_chain(discard=burn, thin=thin, flat=True)

        for ip, p in enumerate(self.parameters):
            samples[p] = chains[:,ip]

        self.save_samples(samples,sample_save_ID)

        return samples


    def save_samples(self, samples, save_ID):
        
        samples = {key: arr.tolist() for key,arr in samples.items()}

        with open('%s/%s.json'%(self.output_directory,save_ID),"w") as f:
            json.dump(samples,f)
        

