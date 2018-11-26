import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
import pickle
import scipy

from . import models


# sys.path.insert(0, os.path.abspath('..'))

parameter_labels = {}

parameter_labels['log10L*'] = r'\log_{10}(L_{*})'
parameter_labels['log10phi*'] = r'\log_{10}(\phi_{*}/Mpc^{-3})'
parameter_labels['alpha'] = r'\alpha'


class analyse():

    def __init__(self, ID = 'test', sample_save_ID = 'samples'):

        self.ID = ID

        self.samples = pickle.load(open(self.ID+'/'+sample_save_ID+'.p', 'rb')) 

        self.parameters = self.samples.keys()

        # --- calculate median fit
        
        self.median_fit = {}

        for ip, p in enumerate(self.parameters): 
        
            self.median_fit[p] = np.percentile(self.samples[p], 50)

            print(p, np.percentile(self.samples[p], 16), np.percentile(self.samples[p], 50), np.percentile(self.samples[p], 84))


        # --- load observations

        self.observations = pickle.load(open(self.ID+'/observations.p', 'rb')) 

        # --- try to load input parameters

        try:
        
            self.input_parameters = pickle.load(open(self.ID+'/input_parameters.p', 'rb')) 
        
        except:
        
            self.input_parameters = False

        print(self.input_parameters)
    
 
    def LF(self, output_filename = False):
    
        # plt.style.use('simple')
        
        fig = plt.figure(figsize=(3,3 ))

        ax = fig.add_axes([0.15, 0.15, 0.8, 0.75 ])

        bw = 0.01
        log10L = np.arange(27, 31.0, bw)

        # --- plot input LF if available

        if self.input_parameters:

            ax.axvline( self.input_parameters['log10L*'], c='k', alpha = 0.1)
            ax.axhline( self.input_parameters['log10phi*'], c='k', alpha = 0.1)
            
            inputLF = models.Schechter(self.input_parameters)
            
            ax.plot(log10L, inputLF.log10phi(log10L), c='k', lw=3, alpha = 0.2)
    
    
        # --- plot median-fit
    
        medianLF = models.Schechter(self.median_fit)
        ax.plot(log10L, medianLF.log10phi(log10L), c='b', lw=1, alpha = 0.5)
    
    
        # --- plot observations
        
        mxphi = -100.
        mxlogL = 0.
        mnlogL = 100.
        
        for obs in self.observations:


            logV = np.log10(obs['volume'])

            bin_edges = obs['bin_edges']
        
            bin_width = bin_edges[1]-bin_edges[0]
        
            bin_centres = bin_edges[0:-1] + 0.5*bin_width
        
            c = np.random.rand(3,)
        
        
            for bc, n in zip(bin_centres, obs['N']): 
    
                if n>0:
                    ax.plot([bc]*2, np.log10(models.poisson_confidence_interval(n, 0.68)/bin_width) - logV, c=c, lw=1, alpha = 1.0) 
                else:
                    ax.arrow(bc, np.log10(models.poisson_confidence_interval(n, 0.68)/bin_width)[1] - logV, 0.0, -0.5, color=c)

            phi = np.log10(obs['N']/bin_width) - logV

            ax.scatter(bin_centres, phi, c=c, s=5)
    
            if np.max(phi)>mxphi: mxphi = np.max(phi)
            if bin_centres[-1]>mxlogL: mxlogL = bin_centres[-1]
            if bin_centres[0]<mnlogL: mnlogL = bin_centres[0]
        
        volumes = np.array([obs['volume'] for obs in self.observations])
        
        ax.set_ylim([np.log10(1./np.max(np.array(volumes)))-0.5, mxphi+0.5])
        ax.set_xlim([mnlogL-0.25, mxlogL+0.25])
        
        
        
#         ax.set_xlim([self.observations['bin_centres'][0]-0.1, self.observations['bin_centres'][-1]+0.1])
#         ax.set_ylim([np.log10(np.min(self.observations['phis'][self.observations['phis']>0]))-0.3, np.log10(np.max(self.observations['phis']))+0.3])
    
        ax.set_xlabel(r"$\rm \log_{10}(L_{\nu}/erg\, s^{-1}\, Hz^{-1})$")
#         ax.set_ylabel(r"$\rm \log_{10}(\phi/Mpc^{-3})$")
        ax.set_ylabel(r"$\rm \log_{10}(\phi/Mpc^{-3}\,dex^{-1})$")
    
        if output_filename: 
            fig.savefig(output_filename, dpi = 300)
        else:
            fig.savefig(self.ID+'/LF.pdf', dpi = 300)
        
    
    

    def triangle(self, bins = 50, contours = True, hist2d = True, output_filename = False, ccolor = '0.0', ranges = False):
    
        n = len(self.parameters)
    
        # ---- initialise figure

        plt.rcParams['mathtext.fontset'] = 'stixsans'
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.size'] = 7 # perhaps should depend on number of parameters to be plotted

        plt.rcParams['ytick.labelsize'] = 4 # perhaps should depend on number of parameters to be plotted
        plt.rcParams['xtick.labelsize'] = 4 # perhaps should depend on number of parameters to be plotted
    
        plt.rcParams['ytick.direction'] = 'in'    # direction: in, out, or inout
        plt.rcParams['xtick.direction'] = 'in'    # direction: in, out, or inout
    
        plt.rcParams['ytick.minor.visible'] = True
        plt.rcParams['xtick.minor.visible'] = True
    

        fig, axes = plt.subplots(n,n, figsize = (4,4))

        left  = 0.125  # the left side of the subplots of the figure
        right = 0.9    # the right side of the subplots of the figure
        bottom = 0.1   # the bottom of the subplots of the figure
        top = 0.9      # the top of the subplots of the figure
        wspace = 0.02   # the amount of width reserved for blank space between subplots
        hspace = 0.02   # the amount of height reserved for white space between subplots
    
        fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


        # ---- loop over parameters
        
        for i,ikey in enumerate(self.parameters):
            for j,jkey in enumerate(self.parameters):
                
                # axes[i,j].text(0.5,0.5,str(i)+str(j), transform=axes[i,j].transAxes) # label panels
                
                axes[i,j].locator_params(axis = 'x', nbins=3)
                axes[i,j].locator_params(axis = 'y', nbins=3)
      
                # pi = self.parameters[ikey]
                # pj = self.parameters[jkey]
            
                if i!=0 and j==0 and j<n-1:
                    #axes[i,j].set_ylabel(r'${\rm'+parameter_labels[pi]+'}$')
                    axes[i,j].set_ylabel(r'${\rm%s}$'%ikey)
                    
                if i==(n-1):
                    #axes[i,j].set_xlabel(r'${\rm'+parameter_labels[pj]+'}$')
                    axes[i,j].set_xlabel(r'${\rm%s}$'%jkey)


                if j == i:
                    median = np.percentile(self.samples[ikey], 50)
                
                    if ranges:
                        range = ranges[ikey]
                    
                    else:
                        IQR = np.percentile(self.samples[ikey], 75) - np.percentile(self.samples[ikey], 25)
                        range = [median-3*IQR, median+3*IQR]

                    
                
                    N,b,p = axes[i,j].hist(self.samples[ikey], bins = bins, range = range, color = '0.7', edgecolor = '0.7')
                
                    mxN = np.max(N)
                
                    if self.input_parameters: axes[i,j].axvline(self.input_parameters[jkey], lw = 1, c = 'k', alpha = 0.2)
                
                    axes[i,j].scatter(median,mxN*1.3,c='k',s=5)
                
                
                    axes[i,j].plot([np.percentile(self.samples[ikey], 16.), np.percentile(self.samples[ikey], 84.)],[mxN*1.3]*2,c='k',lw=1)
                
                    axes[i,j].set_xlim(range)
                    axes[i,j].set_ylim([0.,mxN*2.])
                    
                    axes[i,j].spines['right'].set_visible(False)
                    axes[i,j].spines['top'].set_visible(False)
                    axes[i,j].spines['left'].set_visible(False)
                    axes[i,j].yaxis.set_ticks_position('none')
                    axes[i,j].axes.get_yaxis().set_ticks([])


                elif j < i:
                    
                    if self.input_parameters:
                    
                        if hist2d: 
                            c = '1.0'
                        else:
                            c = 'k'
                        
                        axes[i,j].axhline(self.input_parameters[ikey], lw = 1, c = c, alpha = 0.2)
                        axes[i,j].axvline(self.input_parameters[jkey], lw = 1, c = c, alpha = 0.2)
                    
    
                    if ranges:

                        rangei = ranges[ikey]
                        rangej = ranges[jkey]                    
                    
                    else:
    
                        IQR = np.percentile(self.samples[ikey], 75) - np.percentile(self.samples[ikey], 25)
                        median = np.percentile(self.samples[ikey], 50)
                        rangei = [median-3*IQR, median+3*IQR]
    
                        IQR = np.percentile(self.samples[jkey], 75) - np.percentile(self.samples[jkey], 25)
                        median = np.percentile(self.samples[jkey], 50)
                        rangej = [median-3*IQR, median+3*IQR]
                    
    
                    H, xe, ye = np.histogram2d(self.samples[jkey], self.samples[ikey], bins = bins, range = [rangej, rangei]) 
                        
                    H = H.T
  
                    xlims = [xe[0], xe[-1]]
                    ylims = [ye[0], ye[-1]]
    
                    axes[i,j].set_xlim(xlims)
                    axes[i,j].set_ylim(ylims)
    
                    if hist2d: 
                        X, Y = np.meshgrid(xe, ye)
                        axes[i,j].pcolormesh(X, Y, H, cmap = 'plasma',linewidth=0,rasterized=True) #

                    if j != 0: axes[i,j].set_yticklabels([])


                    # --- add contours
                    
                    if contours: 
                    
                        norm=H.sum() # Find the norm of the sum
                        # Set contour levels
                        # contour1=0.99 
                        contour2=0.95
                        contour3=0.68

                        # Set target levels as percentage of norm
                        # target1 = norm*contour1
                        target2 = norm*contour2
                        target3 = norm*contour3

                        # Take histogram bin membership as proportional to Likelihood
                        # This is true when data comes from a Markovian process
                        def objective(limit, target):
                            w = np.where(H>limit)
                            count = H[w]
                            return count.sum() - target

                        # Find levels by summing histogram to objective
                        # level1= scipy.optimize.bisect(objective, H.min(), H.max(), args=(target1,))
                        level2= scipy.optimize.bisect(objective, H.min(), H.max(), args=(target2,))
                        level3= scipy.optimize.bisect(objective, H.min(), H.max(), args=(target3,))

                        # For nice contour shading with seaborn, define top level
                        level4=H.max()
                        levels=[level2,level3]

                        bwi = (xe[1]-xe[0])/2.
                        bwj = (ye[1]-ye[0])/2.

                        if hist2d: ccolor = '1.0'

                        axes[i,j].contour(xe[:-1]+bwi, ye[:-1]+bwj, H, levels=levels, linewidths=0.5, colors=ccolor)
                        
                        
                        

                else:
               
                    axes[i,j].set_axis_off() 



        if output_filename: 
            fig.savefig(output_filename, dpi = 300)
        else:
            fig.savefig(self.ID+'/triangle.pdf', dpi = 300)

        fig.clf()
  
  
  
  
#     def triangle_wP(self, bins = 50, output_filename = 'triangle.pdf'):
#     
#         n = len(self.parameters)
#     
#         # ---- initialise figure
# 
#         plt.rcParams['mathtext.fontset'] = 'stixsans'
#         plt.rcParams['text.usetex'] = False
#         plt.rcParams['font.size'] = 7 # perhaps should depend on number of parameters to be plotted
# 
#         plt.rcParams['ytick.labelsize'] = 4 # perhaps should depend on number of parameters to be plotted
#         plt.rcParams['xtick.labelsize'] = 4 # perhaps should depend on number of parameters to be plotted
#     
#         plt.rcParams['ytick.direction'] = 'in'    # direction: in, out, or inout
#         plt.rcParams['xtick.direction'] = 'in'    # direction: in, out, or inout
#     
#         plt.rcParams['ytick.minor.visible'] = True
#         plt.rcParams['xtick.minor.visible'] = True
#     
# 
#         fig, axes = plt.subplots(n,n, figsize = (3,3))
# 
#         left  = 0.125  # the left side of the subplots of the figure
#         right = 0.9    # the right side of the subplots of the figure
#         bottom = 0.1   # the bottom of the subplots of the figure
#         top = 0.9      # the top of the subplots of the figure
#         wspace = 0.02   # the amount of width reserved for blank space between subplots
#         hspace = 0.02   # the amount of height reserved for white space between subplots
#     
#         fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
# 
# 
# 
#         # ---- loop over parameters
# 
#         for i in np.arange(n):
#             for j in np.arange(n):
#     
#                 axes[i,j].locator_params(axis = 'x', nbins=3)
#                 axes[i,j].locator_params(axis = 'y', nbins=3)
#       
#                 pi = self.parameters[i]
#                 pj = self.parameters[j]
#                     
#                     
#                 
# 
#                 if i != n-1: axes[i,j].set_xticklabels([])
#     
# #                 if i == n-1: 
# #                     axes[i,j].set_ylabel(r'${\rm'+parameter_labels[pi]+'}$')
# #     
# #                 if j == 0: 
# #                 
# #                     print i, pi
# #                 
# #                     axes[i,j].set_xlabel(r'${\rm'+parameter_labels[pj]+'}$')
# #                     # axes[i,j].get_yaxis().set_label_coords(-0.1*n,0.5)
#   
#   
#                 if j==0 and j<n-1:
#                 
#                     axes[i,j].set_ylabel(r'${\rm'+parameter_labels[pi]+'}$')
#                     
#                 if i==0 and j==n-1:
#                 
#                     axes[i,j].set_xlabel(r'${\rm'+parameter_labels[pj]+'}$')
# 
# 
#                 if j < i:
#                     
#     
#                     IQR = np.percentile(self.samples[pi], 75) - np.percentile(self.samples[pi], 25)
#                     median = np.percentile(self.samples[pi], 50)
#                     rangei = [median-3*IQR, median+3*IQR]
#     
#                     IQR = np.percentile(self.samples[pj], 75) - np.percentile(self.samples[pj], 25)
#                     median = np.percentile(self.samples[pj], 50)
#                     rangej = [median-3*IQR, median+3*IQR]
#                     
#     
#                     H, xe, ye = np.histogram2d(self.samples[pj], self.samples[pi], bins = bins, range = [rangej, rangei]) 
#                         
#                     X, Y = np.meshgrid(xe, ye)
#   
#                     xlims = [xe[0], xe[-1]]
#                     ylims = [ye[0], ye[-1]]
#     
#                     axes[i,j].set_xlim(xlims)
#                     axes[i,j].set_ylim(ylims)
#     
#                     axes[i,j].pcolormesh(X, Y, H.T, cmap = 'plasma',linewidth=0,rasterized=True) #
# 
#                     if j != 0: axes[i,j].set_yticklabels([])
# 
# 
#                     # --- add contours
#                     
#                     norm=H.sum() # Find the norm of the sum
#                     # Set contour levels
#                     # contour1=0.99 
#                     contour2=0.95
#                     contour3=0.68
# 
#                     # Set target levels as percentage of norm
#                     # target1 = norm*contour1
#                     target2 = norm*contour2
#                     target3 = norm*contour3
# 
#                     # Take histogram bin membership as proportional to Likelihood
#                     # This is true when data comes from a Markovian process
#                     def objective(limit, target):
#                         w = np.where(H>limit)
#                         count = H[w]
#                         return count.sum() - target
# 
#                     # Find levels by summing histogram to objective
#                     # level1= scipy.optimize.bisect(objective, H.min(), H.max(), args=(target1,))
#                     level2= scipy.optimize.bisect(objective, H.min(), H.max(), args=(target2,))
#                     level3= scipy.optimize.bisect(objective, H.min(), H.max(), args=(target3,))
# 
#                     # For nice contour shading with seaborn, define top level
#                     level4=H.max()
#                     levels=[level2,level3]
# 
#                     bwi = (xe[1]-xe[0])/2.
#                     bwj = (ye[1]-ye[0])/2.
# 
#                     axes[i,j].contour(xe[:-1]+bwi, ye[:-1]+bwj, H, levels=levels, linewidths=0.5, colors='1.0')
# 
#                 else:
#                
#                     axes[i,j].set_axis_off() 
# 
# 
# 
#         fig.savefig(output_filename, dpi = 300)
# 
#         fig.clf()
    
  
  
  
