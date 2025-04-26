import numpy as np
import matplotlib.pyplot as plt
import libstempo as T
import libstempo.plot as LP, libstempo.toasim as LT
from libstempo import spharmORFbasis as anis
import glob
import sys
import math
import json
import os
import scipy.interpolate as interp
from enterprise.signals import gp_signals
from enterprise_extensions import model_utils, blocks
import dynesty
from enterprise.signals import signal_base
from enterprise.pulsar import Pulsar
from enterprise_extensions.frequentist import optimal_statistic as opt_stat
import enterprise.signals.parameter as parameter
from enterprise.signals import white_signals

import corner
from enterprise_extensions import sampler as sp
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

import scipy.constants as sc
from scipy.stats import rv_histogram
import la_forge.core as co
from ceffyl import densities
from ceffyl import Ceffyl

#input data directory
datadir = sys.argv[1]
#number of frequency components
comp = int(sys.argv[2])
#chain directory
datadir_out = sys.argv[3]
#spectrum directory
datadir_spec = sys.argv[4]

parfiles = sorted(glob.glob(datadir + '*.par'))
timfiles = sorted(glob.glob(datadir + '*.tim'))

psrs = []
ephemeris = None
for p, t in zip(parfiles, timfiles):
    psr = Pulsar(p, t, ephem=ephemeris)
    psrs.append(psr)

# find the maximum time span to set GW frequency sampling
Tspan = model_utils.get_tspan(psrs)

# free-spec MCMC chain
chaindir = datadir_out
freespec = co.Core(chaindir=chaindir)  # open the file

params = freespec.params  # access parameter names

#check which frequencies are available
check = f'g'
res = [idx for idx in params if idx[0].lower() == check.lower()]

freqs = np.arange(1, comp+1)/Tspan
num_frq = len(freqs)

# crn free spec param names
hd_rho_labels = [f'gw_crn_log10_rho_{ii}' for ii in range(len(res))]  

# accessing crn free spec rhos
hd_chain = freespec(hd_rho_labels)  # returns array of burn-in chain

# plotting violin plots
rho_bins = np.linspace(-9, -4, num=400) # rho bins
drho = rho_bins[1]-rho_bins[0]

# turning freespec into rv hist and sampling
epsilon = 1e-250  # make histograms for each rho
hd_histograms = np.array([np.histogram(hd_chain[:,ii], 
                                    rho_bins, density=True)[0]
                       for ii in range(hd_chain.shape[1])])

bin_mid = (rho_bins[1:] + rho_bins[:-1]) / 2.0
rvhist = [rv_histogram([hist, rho_bins]) for hist in hd_histograms]

# draw samples from histograms for each rho
rv_samples_hd = [rvh.rvs(size=100000) for rvh in rvhist]

fig = plt.figure()
ax = fig.add_subplot(111, rasterized=True)

yrfac = np.log10(365.25 * 86400)
    
# free-spec violinplot of random samples
v2 = ax.violinplot(rv_samples_hd, positions=np.log10(freqs), 
                    widths=0.05, showextrema=False)

# axis labels
ax.set_xlabel(r'$\log_{10}$(Frequency [Hz])'), 
           #fontsize=12)
ax.set_ylabel(r'$\log_{10}$(Excess timing delay [s])'), 
           #fontsize=12);

plt.savefig(datadir_out + "free_spec.png", dpi=300)

#saving file in la_forge format
os.system("mkdir " + datadir_spec)
chaindir = datadir_out + ''
freespec = co.Core(chaindir=chaindir)  # open the file
freespec.set_rn_freqs(freqs, Tspan, num_frq)

freespec.save(datadir_spec + "forg_dat.core")

kdes = densities.DE_factory(coredir=datadir_spec,
                            recursive=False, pulsar_names=['freespec'])

kdes.setup_densities(outdir=datadir_spec)
                            
pta = Ceffyl.ceffyl(datadir=datadir_spec)
