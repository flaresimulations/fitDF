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



def phi_to_N(phi, volume, bin_lims):
    """
    phi (Mpc^-3 dex^-1)
    N -> no units
    """
    dex = bin_lims[1:] - bin_lims[:-1]
    return phi * volume * dex



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
        phi = np.log(10) * 10**self.sp['log10phi*'] * np.exp(-10**y) * 10**(y*(self.sp['alpha'] + 1))

        return np.log10(phi)


    @staticmethod
    def _integ(x,a,D):
        return 10**((a+1)*(x-D)) * np.exp(-10**(x-D))


    def binPhi(self,D1,D2):
        """
        Integrate function between set limits
        """

        args = (self.sp['alpha'],self.sp['D*'])
        gamma = scipy.integrate.quad(self._integ, D1, D2, args=args)[0]

        return gamma * 10**self.sp['log10phi*'] * np.log(10)


    def N(self, volume, bin_edges):
        """
        return the exact number of galaxies expected in each bin

        Args:
            volume (float)
            bin_edges (array, float)
        """
        N = np.array([self.binPhi(x1,x2) for x1,x2 \
             in zip(bin_edges[:-1],bin_edges[1:])])*volume

        return N


class Schechter_Mags():

    def __init__(self, sp=None):

        if sp is None:
            self.sp = {'D*': None, 'log10phi*': None, 'alpha': None}
        else:
            self.sp = sp


    def update_params(self, sp):
        self.sp = sp


    def log10phi(self, D):

        y = 0.4*(D - self.sp['D*'])

        return np.log10(0.4*np.log(10)) + self.sp['log10phi*'] - y*(self.sp['alpha']+1.) \
        - (np.log10(np.e)) * (10**(-y))


    @staticmethod
    def _integ(x,a):
        return x**(a) * np.exp(-x)


    def binPhi(self,D1,D2):
        """
        Integrate function between set limits
        """
        x1 = 10**(-0.4*(D1 - self.sp['D*']))
        x2 = 10**(-0.4*(D2 - self.sp['D*']))

        gamma = scipy.integrate.quad(self._integ, x1, x2, args=self.sp['alpha'])[0]
        return 0.4 * np.log(10) * gamma * (10**self.sp['log10phi*'])


    def N(self, volume, bin_edges):
        """
        return the exact number of galaxies expected in each bin

        Args:
            volume (float)
            bin_edges (array, float)
        """
        N = np.array([self.binPhi(x2,x1) for x1,x2 \
             in zip(bin_edges[:-1],bin_edges[1:])])*volume

        return N


class DoubleSchechter():
    """
    Five parameter model, with a single value for the break (D*)

    see e.g. Baldry+11 (arXiv:1111.5707)
    """

    def __init__(self, sp=None):

        if sp is None:
            self.sp = {'D*': None, 'log10phi*_1': None, 'alpha_1': None,
                       'log10phi*_2': None, 'alpha_2': None}
        else:
            self.sp = sp


    def update_params(self, sp):
        self.sp = sp


    def log10phi(self, D):

        y =10**(D - self.sp['D*'])

        _temp = 10**self.sp['log10phi*_1'] * y**(self.sp['alpha_1']+1)
        _temp += 10**self.sp['log10phi*_2'] * y**(self.sp['alpha_2']+1)

        return -y / np.log(10) + np.log10(_temp) + np.log10(np.log(10))


    @staticmethod
    def _integ(x,a1,a2,phi1,phi2,D):
        y=x-D
        return (phi1 * 10**(a1*y) + phi2 * 10**(a2*y)) * np.exp(-10**y)


    def binPhi(self,D1,D2):
        """
        Integrate function between set limits
        """
        args=(self.sp['alpha_1'] + 1,
              self.sp['alpha_2'] + 1,
              10**self.sp['log10phi*_1'],
              10**self.sp['log10phi*_2'],
              self.sp['D*'])

        N = scipy.integrate.quad(self._integ, D1, D2, args=args)[0]

        return N * np.log(10)


    def N(self, volume, bin_edges):
        """
        return the exact number of galaxies expected in each bin

        Args:
            volume (float)
            bin_edges (array, float)
        """
        N = np.array([self.binPhi(x1,x2) for x1,x2 \
             in zip(bin_edges[:-1],bin_edges[1:])])*volume

        return N


class DoubleSchechter_Mags():

    def __init__(self, sp=None):

        if sp is None:
            self.sp = {'D*': None, 'log10phi*_1': None, 'alpha_1': None,
                       'log10phi*_2': None, 'alpha_2': None}
        else:
            self.sp = sp


    def update_params(self, sp):
        self.sp = sp


    def log10phi(self, D):

        y = 10**(-0.4*(D - self.sp['D*']))

        _temp = 10**self.sp['log10phi*_1'] * y**(self.sp['alpha_1']+1)
        _temp += 10**self.sp['log10phi*_2'] * y**(self.sp['alpha_2']+1)

        return -y + np.log10(_temp) + np.log10(0.4*np.log(10.))


    @staticmethod
    def _integ(x,a1,a2,phi1,phi2):
        return phi1*((x**a1) + (phi2/phi1) * (x**a2)) * np.exp(-x)


    def binPhi(self,D1,D2):
        """
        Integrate function between set limits
        """
        x1 = 10**(-0.4*(D1 - self.sp['D*']))
        x2 = 10**(-0.4*(D2 - self.sp['D*']))

        args=(self.sp['alpha_1'],self.sp['alpha_2'],
              10**self.sp['log10phi*_1'],10**self.sp['log10phi*_2'])

        gamma = scipy.integrate.quad(self._integ, x1, x2, args=args)[0]

        return 0.4 * np.log(10) * gamma


    def N(self, volume, bin_edges):
        """
        return the exact number of galaxies expected in each bin

        Args:
            volume (float)
            bin_edges (array, float)
        """
        N = np.array([self.binPhi(x2,x1) for x1,x2 \
             in zip(bin_edges[:-1],bin_edges[1:])])*volume

        return N

class DPL_Mags():

    def __init__(self, sp=None):

        if sp is None:
            self.sp = {'D*': None, 'log10phi*': None, 'alpha_1': None,
            'alpha_2': None}
        else:
            self.sp = sp


    def update_params(self, sp):
        self.sp = sp


    def log10phi(self, D):

        y = 10**(0.4*(D - self.sp['D*']))

        _temp = self.sp['log10phi*'] - np.log10(y**(self.sp['alpha_1']+1) + y**(self.sp['alpha_2']+1))

        return _temp


    @staticmethod
    def _integ(x,D,a1,a2):
        return 1/(10**(0.4*(x-D)*(a1+1)) + 10**(0.4*(x-D)*(a2+1)))


    def binPhi(self,D1,D2):
        """
        Integrate function between set limits
        """

        args=(self.sp['D*'],self.sp['alpha_1'],self.sp['alpha_2'])

        gamma = scipy.integrate.quad(self._integ, D1, D2, args=args)[0]

        return gamma * (10**self.sp['log10phi*'])


    def N(self, volume, bin_edges):
        """
        return the exact number of galaxies expected in each bin

        Args:
            volume (float)
            bin_edges (array, float)
        """
        N = np.array([self.binPhi(x1,x2) for x1,x2 \
             in zip(bin_edges[:-1],bin_edges[1:])])*volume

        return N

    

def _CDF(model, D_lowlim, normed = True, inf_lim=30):

    log10Ls = np.arange(model.sp['D*']+5.,D_lowlim-0.01,-0.01)

    CDF = np.array([model.binPhi(log10L,inf_lim) for log10L in log10Ls])

    if normed: CDF /= CDF[-1]

    return log10Ls, CDF


def sample(model, volume, D_lowlim, inf_lim=100):

    D, cdf = _CDF(model, D_lowlim, normed=False, inf_lim=inf_lim)

    n2 = model.binPhi(D_lowlim, inf_lim)*volume

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
