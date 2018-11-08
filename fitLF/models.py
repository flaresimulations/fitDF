import numpy as np
import mpmath
import scipy.stats

   

geo=(4.*np.pi*(100.*10.*3.0867*10**16)**2)#factor relating the L to M in cm^2

def L(M):
    
    return 10**(-0.4*(M+48.6))*geo

def log10L(M):
    
    return -0.4*(M+48.6)+np.log10(geo)
    
    
def M(log10L):

    return -2.5*(log10L - np.log10(geo))-48.6


def poisson_confidence_interval(n,p):
    
    #http://ms.mcmaster.ca/peter/s743/poissonalpha.html
        
    #e.g. p=0.68 for 1 sigma
    
    #agrees with http://hyperphysics.phy-astr.gsu.edu/hbase/math/poifcn.html
        
    # see comments on JavaStat page
    
    #  scipy.stats.chi2.ppf((1.-p)/2.,2*n)/2. also known
    
    if n>0:   
        interval=[scipy.stats.chi2.ppf((1.-p)/2.,2*n)/2.,scipy.stats.chi2.ppf(p+(1.-p)/2.,2*(n+1))/2.]       
    
    else:
        
        #this bit works out the case for n=0
        
        ul=(1.-p)/2.
        
        prev=1.0
        for a in np.arange(0.,5.0,0.001):
        
            cdf=scipy.stats.poisson.cdf(n,a)
        
            if cdf<ul and prev>ul:
                i=a
        
            prev=cdf
        
        interval=[0.,i]
    
    
    return np.array(interval)





class Schechter():

    def __init__(self,sp): #schechter_params={'L*':,'phi*':,'alpha':} or {'M*':,'phi*':,'alpha':}

        # print sp


        if 'M*' in sp.keys():
            sp['L*'] = 10**(-0.4*(sp['M*']+48.6))*geo
            
        if 'L*' in sp.keys():
            sp['log10L*'] = np.log10(sp['L*'])
          
        if 'log10(L*)' in sp.keys():
            sp['log10L*'] = sp['log10(L*)']
          
        if 'log10(phi*)' in sp.keys():
            sp['log10phi*'] = sp['log10(phi*)']
          
        sp['phi*'] =10**sp['log10phi*']
             
        self.sp=sp
               
    def log10phi(self, log10L):
     
        y = log10L - self.sp['log10L*']
        
        alpha = self.sp['alpha']
     
        return np.log10(self.sp['phi*']) + np.log10(np.log(10.))  + y*(alpha+1.) + -10**y/np.log(10.)
     
     
    def CulmPhi(self,log10L):
    
        y = log10L - self.sp['log10L*']
        x = 10**y
        alpha = self.sp['alpha']
    
        # num = float(mpmath.gammainc(alpha+1.,x))*self.sp['phi*'] # integral from log10L to L=\infty
        num = float(scipy.integrate.quad(gamma, x,np.inf,args=alpha)[0])*self.sp['phi*']

        return num   
            
        
    def CDF(self, log10L_limit, normed = True):
    
        log10Ls = np.arange(self.sp['log10L*']+5.,log10L_limit-0.01,-0.01)
    
        CDF = np.array([self.CulmPhi(log10L) for log10L in log10Ls])
    
        if normed: CDF /= CDF[-1]
    
        return log10Ls, CDF 
        

    def N(self, volume, bin_edges):
    
        # --- return the exact number of galaxies expected in each bin
    
        CulmN = np.array([self.CulmPhi(x) for x in bin_edges])*volume
        
        return -(CulmN[1:] - CulmN[0:-1])


    # -------------------------------------------
    # ----------- sample the luminosity function  in a given volume.


    def sample(self, volume, log10L_limit): # --- define volume
            
        L, CDF = self.CDF(log10L_limit, normed=False)

        n2 = self.CulmPhi(log10L_limit)*volume

        n = np.random.poisson(volume * CDF[-1]) # --- I don't think this is strictly correct but I can't think of a better approach

        nCDF = CDF/CDF[-1]
        
        log10L_sample = np.interp(np.random.random(n), nCDF, L)
        
        return log10L_sample

    


def bin(log10L_sample, volume, bins):
    
        # --- bins can either be the number of bins or the bin_edges

        N_sample, bin_edges = np.histogram(log10L_sample, bins = bins, normed=False)
         
        return N_sample
    



  
 


