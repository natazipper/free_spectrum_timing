import time
start_time = time.time()
import random

import numpy as np
import glob
import sys
import matplotlib.pyplot as plt
import scipy.linalg as sl
import scipy

import libstempo as T
import libstempo.plot as LP
import toasim as LT
from enterprise.pulsar import Pulsar, Tempo2Pulsar
#import libstempo.toasim as LT
from libstempo.libstempo import GWB
from itertools import combinations
import json
import os

from astropy import units as u
from astropy.coordinates import SkyCoord

import corner

from enterprise.signals import gp_signals
from enterprise_extensions import model_utils
#import blocks_new as blocks
from enterprise_extensions import blocks
from enterprise.signals import signal_base
import enterprise.constants as const
from enterprise.pulsar import Pulsar, Tempo2Pulsar
from enterprise_extensions.frequentist import optimal_statistic as opt_stat
import enterprise.signals.parameter as parameter
from enterprise.signals import white_signals

import corner
from enterprise_extensions import sampler as sp
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc


#input data directory
datadir_in = sys.argv[1]
#output directory
datadir_out = sys.argv[2]
#number of iteration in mcmc
iter_num = int(sys.argv[3])
#number of components
comp = 30

#parameters to play with
Term = "both"
dzeta = 10

pop_a=5
pop=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

mjd_toas = np.arange(38738, 69818, 30)


#for numb in range(Npsr):
#    p_name = datadir_out + prefix_psr + str(numb) + '.par'
#    t_name = datadir_out + prefix_psr + str(numb) + '.tim'
#    parfiles.append(p_name)
#    timfiles.append(t_name)

parfiles = sorted(glob.glob(datadir_in + '*.par'))
Npsr = len(parfiles)
print(parfiles)


#psrs = []
#for numb in range(Npsr):
#    psr = T.tempopulsar(parfiles[numb], timfiles[numb])
#    psrs.append(psr)

err = 0.1

psrs = []

for ii in range(0,Npsr):

    # years of observations>
    psr = LT.fakepulsar(parfile=parfiles[ii],
            obstimes=mjd_toas, toaerr=err)
            
    LT.make_ideal(psr)

    #Generate white noise
    LT.add_efac(psr,efac=1.0)
    
#    add_rednoise(psr, 2e-14, 2.1)

    # add to list
    psrs.append(psr)

for i in range(Npsr):
    #psrs[i].fit()
    LP.plotres(psrs[i])
    
plt.savefig(datadir_out + "residuals_white.png", dpi=300)
plt.clf()

aleph = random.uniform(0, 2*np.pi)*180/np.pi
delet = random.uniform(-np.pi/2, np.pi/2)*180/np.pi

cord = SkyCoord(ra=aleph*u.deg, dec=delet*u.deg)

mc = 1*10**9
dist = 1.
T_lim_min = 31080
f_gw = [3.72396206e-10, 5.17442850e-10, 7.18984509e-10, 9.99025735e-10,
 1.38814176e-09, 1.92881672e-09, 2.68008214e-09, 3.72396206e-09,
 5.17442850e-09, 7.18984509e-09, 9.99025735e-09, 1.38814176e-08,
 1.92881672e-08, 2.68008214e-08, 3.72396206e-08]
fgw = 2*f_gw[4]
phase0 = random.uniform(0, 2*np.pi) #Initial Phase of GW source [radians]
psi = random.uniform(0, np.pi) #Polarization of GW source [radians]
inc = np.pi/4 #Inclination of GW source [radians]
evolve = False
pdist = []
for i in range(Npsr):
    pdist.append(np.random.normal(loc=pop[pop_a], scale=0.01))
    
ecl = SkyCoord.transform_to(cord, 'geocentricmeanecliptic')
gwtheta = ecl.lon.rad
gwphi = ecl.lat.rad
    
#for numb in range(Npsr):
#    LT.add_cgw(psrs[numb], gwtheta=gwtheta, gwphi=gwphi, mc=mc, dist=dist, fgw=fgw, phase0=phase0, psi=psi, inc=inc, dzeta=dzeta, Term=Term, #pdist=pdist[numb], psrTerm=False, evolve=False)

for i in range(Npsr):
    #psrs[i].fit()
    LP.plotres(psrs[i])
    
plt.savefig(datadir_out + "residuals.png", dpi=300)
plt.clf()

Psrs = []
for ii in psrs:
    ii.fit()
    psr = Tempo2Pulsar(ii)
    Psrs.append(psr)
    
os.system("mkdir " + datadir_out)
os.system("mkdir " + datadir_out + "final")
    
# find the maximum time span to set GW frequency sampling
Tspan = model_utils.get_tspan(Psrs)

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
                                   components=comp, name='gw_crn', orf = None)
#s += blocks.red_noise_block(psd='spectrum', prior='log-uniform', components=30)

# We set up the PTA object using the signal we defined above and the pulsars
pta = signal_base.PTA([s(p) for p in Psrs])

for Psr in psrs:
    
    Psr.fit()
    Psr.savepar(datadir_out + "final/" + Psr.name + '.par')
    Psr.savetim(datadir_out + "final/" + Psr.name + '.tim')
    T.purgetim(datadir_out + "final/" + Psr.name + '.tim')

def run_sampler(pta, iter_num, outdir = ''):

    N = int(iter_num)                                    # number of samples
    x0 = np.hstack(p.sample() for p in pta.params)  # initial parameter vector
    ndim = len(x0)                                  # number of dimensions
    print('x0 =', x0)

    # initial jump covariance matrix
    cov = np.diag(np.ones(ndim) * 0.01**2)
    
    #initialize the sampler object
    sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, outDir=outdir, resume=False)
    
    # additional jumps
    jp = sp.JumpProposal(pta)
    sampler.addProposalToCycle(jp.draw_from_prior, 5)
    
    sel_sig = ["rn", "red_noise", "dm_gp", "fcn", "chrom-rn", "srn", "dm_srn", "freechrom-srn", "chrom-srn",
                        "dm-expd", "freechrom-expd", "chrom-expd",
                        "dm-y", "freechrom-y", "chrom-y",
                        "gw"]
    for s in sel_sig:
        if any([s in p for p in pta.param_names]):
            #pnames = [p.name for p in pta.params if s in p.name]
            #print('Adding %s prior draws with parameters :'%s, pnames, '\n')
            print('Adding %s prior draws.'%s)
            sampler.addProposalToCycle(jp.draw_from_par_prior(s), 10)

        
    sampler.sample(x0, N, SCAMweight=40, AMweight=25, DEweight=55) # these weights relate to frequency of jumps

    # write a list of the parameters to a text file
    # and a list of the parameter groupings used
    filename = outdir + '/params.txt'
    np.savetxt(filename,list(map(str, pta.param_names)), fmt='%s')
    
    return None

print(pta.params)

run_sampler(pta, iter_num, datadir_out)

chainname = 'chain_1'
chain = np.loadtxt(datadir_out + chainname + '.txt')

burn = int(0.3*chain.shape[0])


#corner.corner(chain[burn:,-4],
#                      bins =30,
#                      plot_datapoints=False, plot_density=True, 
#                      plot_contours=False,fill_contours=False,
#                      show_titles = True, use_math_text=True, verbose=True)

fs = (np.arange(comp)+1) / Tspan
parts = plt.violinplot(
    chain[burn:,:-4], positions=fs, widths=0.07*fs)
plt.xlabel("Frequency, Hz", fontsize=12)
plt.ylabel("Spectrum", fontsize=12)
plt.xscale("log")
plt.savefig(datadir_out + "violin.png", dpi=300)
plt.clf()

#calculating 1-sigma uncertainties
std_lst = np.std(chain[burn:,:-4], axis=0)
mn_lst = np.mean(chain[burn:,:-4], axis=0)
np.savetxt(datadir_out + "free_spec.txt", np.vstack((fs, mn_lst, std_lst)).T)
plt.clf()


ostat = opt_stat.OptimalStatistic(psrs, pta=pta, orf='hd')


chain = np.genfromtxt(datadir_out + chainname + ".txt")
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
