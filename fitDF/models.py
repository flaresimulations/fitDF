import numpy as np
import scipy.stats
import scipy.special
import scipy.integrate

geo=(4.*np.pi*(100.*10.*3.0867*10**16)**2)#factor relating the L to M in cm^2

def L(M):
    """
    Convert Absolute Magnitude to Luminosity (erg s^-1 cm^-2)
    """
    return 10**(-0.4*(M+48.6))*geo

def log10L(M):
    """
    Convert Absolute Magnitude to Log Luminosity (erg s^-1 cm^-2)
    """
    return -0.4*(M+48.6)+np.log10(geo)
    
    
def M(log10L):
    """
    Convert Log Luminosity to Absolute Magnitude
    """
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

    def __init__(self, sp=None):

        if sp is None: 
            self.sp = {'D*': None, 'log10phi*': None, 'alpha': None}
        else:
            self.sp = sp


    def update_params(self, sp):        
        self.sp = sp

 
    def log10phi(self, D):

        y = D - self.sp['D*']

        return self.sp['log10phi*'] + np.log10(np.log(10.)) \
                    + y*(self.sp['alpha']+1.) + -10**y/np.log(10.)

     
    @staticmethod
    def _integ(x,a):
        return x**(a-1) * np.exp(-x)


    def CulmPhi(self,D):
        """
        Args:
            D (array, float)
        """
    
        y = D - self.sp['D*']
        x = 10**y
        alpha = self.sp['alpha']

        gamma = scipy.integrate.quad(self._integ, x,np.inf,args=alpha+1)[0]
        num = gamma*(10**self.sp['log10phi*'])

        return num   
            

    def N(self, volume, bin_edges):
        """
        return the exact number of galaxies expected in each bin

        Args:
            volume (float)
            bin_edges (array, float)
        """ 

    
        CulmN = np.array([self.CulmPhi(x) for x in bin_edges])*volume
        
        return -(CulmN[1:] - CulmN[0:-1])


def _CDF(model, D_lowlim, normed = True):

    log10Ls = np.arange(model.sp['D*']+5.,D_lowlim-0.01,-0.01)

    CDF = np.array([model.CulmPhi(log10L) for log10L in log10Ls])

    if normed: CDF /= CDF[-1]

    return log10Ls, CDF 


def sample(model, volume, D_lowlim):
        
    D, cdf = _CDF(model, D_lowlim, normed=False)

    n2 = model.CulmPhi(D_lowlim)*volume

    # --- Not strictly correct but I can't think of a better approach
    n = np.random.poisson(volume * cdf[-1]) 

    ncdf = cdf/cdf[-1]
    
    D_sample = np.interp(np.random.random(n), ncdf, D)
    
    return D_sample


def LF_priors():
    """
    Define some dummy priors for a Luminosity Function fit
    """
    print("Initialising dummy priors")
    priors = {}
    priors['log10phi*'] = scipy.stats.uniform(loc = -7.0, scale = 7.0)
    priors['alpha'] = scipy.stats.uniform(loc = -3.0, scale = 3.0)
    priors['D*'] = scipy.stats.uniform(loc = 26., scale = 5.0)

    return priors

 


def bin(log10L_sample, volume, bins):
    
        # --- bins can either be the number of bins or the bin_edges

        N_sample, bin_edges = np.histogram(log10L_sample, bins = bins, normed=False)
         
        return N_sample
    
