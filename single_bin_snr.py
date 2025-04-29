import numpy as np
import matplotlib.pyplot as plt
import libstempo as T
import libstempo.plot as LP, libstempo.toasim as LT
from libstempo import spharmORFbasis as anis
import glob
import math
import json
import sys
import scipy.interpolate as interp
from enterprise.signals import gp_signals
from enterprise_extensions import model_utils, blocks
import dynesty
from enterprise.signals import signal_base
from enterprise.pulsar import Pulsar
from enterprise_extensions.frequentist import optimal_statistic as opt_stat
import enterprise.signals.parameter as parameter
from enterprise.signals import white_signals
import optimal_statistic as opt_stat

import corner
from enterprise_extensions import sampler as sp
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

#input data directory
datadir = sys.argv[1]
#chain directory
chaindir = sys.argv[2]
#number of components
n_comp = np.int(sys.argv[3])

parfiles = sorted(glob.glob(datadir + '*.par'))
timfiles = sorted(glob.glob(datadir + '*.tim'))

psrs = []
ephemeris = None
for p, t in zip(parfiles, timfiles):
    psr = Pulsar(p, t, ephem=ephemeris)
    psrs.append(psr)
    
# find the maximum time span to set GW frequency sampling
Tspan = model_utils.get_tspan(psrs)

# Here we build the signal model
# First we add the timing model
s = gp_signals.TimingModel()

# Then we add the white noise
# We use different white noise parameters for every backend/receiver combination
# The white noise parameters are held constant
efac = parameter.Constant(1.0)
s += white_signals.MeasurementNoise(efac=efac)

# Finally, we add the common red noise, which is modeled as a Fourier series with 30 frequency components
# The common red noise has a power-law PSD with spectral index of 4.33
s += blocks.common_red_noise_block(psd='spectrum', prior='log-uniform', Tspan=Tspan,
                                   components=n_comp, name='gw_crn', orf = None)

# We set up the PTA object using the signal we defined above and the pulsars
pta = signal_base.PTA([s(p) for p in psrs])

chainname = 'chain_1'
chain = np.loadtxt(chaindir + chainname + '.txt')

burn = int(0.3*chain.shape[0])

#corner.corner(chain[burn:,-4],
#                      bins =30,
#                      plot_datapoints=False, plot_density=True, 
#                      plot_contours=False,fill_contours=False,
#                      show_titles = True, use_math_text=True, verbose=True)

fs = (np.arange(n_comp) + 1) / Tspan
parts = plt.violinplot(
    chain[burn:,:-4], positions=fs, widths=0.07*fs)
plt.savefig(chaindir + "violin_spec.png")   
plt.clf()

ostat = opt_stat.OptimalStatistic(psrs, pta=pta, orf='hd')


chain = np.genfromtxt(chaindir + chainname + ".txt")
chain_r = chain[burn:, :-4]

param_dict = {}
param_dict[pta.param_names[0][:-2]] = chain_r[1002]


for i,p in enumerate(pta.param_names):
    param_dict[p] = chain_r[300][i]

#print(param_dict)
snr_list = []
#freq_list = 10**np.arange(-9, -6, 0.1)
freq_list = fs
for ii in range(10):
    snr_list = []
    for j in range(len(freq_list)):
        param_dict = {}
        param_dict[pta.param_names[0][:-2]] = chain_r[8*ii]
        for i,p in enumerate(pta.param_names):
            param_dict[p] = chain_r[8*ii]
        xi, rho, sig, OS, OS_sig = ostat.compute_os(params = param_dict, psd = "spectrum", fgw=freq_list[j])
        snr_list = np.append(snr_list, OS/OS_sig)
    plt.plot(fs, snr_list, color="black", alpha=0.3)
#plt.yscale("log")
plt.xlabel("Frequency, Hz")
plt.ylabel("OS SNR") 
plt.savefig(chaindir + "freq_res_os.png", dpi=300)
    
