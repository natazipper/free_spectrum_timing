import numpy as np
import matplotlib.pyplot as plt
import libstempo as T
import libstempo.plot as LP, libstempo.toasim as LT
from libstempo import spharmORFbasis as anis
import glob
import math
import json
import os
import sys
import scipy.interpolate as interp
import argparse
from enterprise.signals import gp_signals
from enterprise_extensions import model_utils
import blocks_new as blocks
#from enterprise_extensions import blocks
from enterprise.signals import signal_base
import enterprise.constants as const
from enterprise.pulsar import Pulsar, Tempo2Pulsar
from enterprise_extensions.frequentist import optimal_statistic as opt_stat
import enterprise.signals.parameter as parameter
from enterprise.signals import white_signals

import corner
from enterprise_extensions import sampler as sp
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', dest='datadir', type=str, help='Folder with the initial par/tim')
parser.add_argument('--comp', dest='comp', type=int, help='Number of frequency components')
parser.add_argument('--iter_num', dest='iter_num', type=float, help='Number of mcmc iterations')
parser.add_argument('--datadir_out', dest='datadir_out', type=str, help='Output directory for files')
parser.add_argument('--iter_real', dest='iter_real', type=float, help='Number of an iteration (to run the code)')
parser.add_argument('--cust_spec', dest='cust_spec', type=str, help='File with the customised spectrum')
parser.add_argument('--amp', dest='amp', type=float, default=1e-7, help='Amplitude of the spectrum in Omega_gw')
parser.add_argument('--nt', dest='nt', type=float, default=2.3, help='Slope of the spectrum on Omega_gw')
args = parser.parse_args()

#input data directory
datadir = args.datadir
#number of frequency components
comp = args.comp
#number of iteration in mcmc
iter_num = args.iter_num
#output directory
datadir_out = args.datadir_out
#number of iter
iter_real = args.iter_real
#customised spectrum
add_spec = args.cust_spec
#slope of the omega_gw
nt = args.nt
#amplitude of the omega_gw
amp = args.amp

if add_spec:
    print("Using customised Omega_gw")
else:
    print("Using nt={} and amp={} for the Omega_gw".format(nt, amp))

H0 = 3.*1e-18
day = 24 * 3600
year = 365.25 * day

def extrap1d(interpolator):
    """
    Function to extend an interpolation function to an
    extrapolation function.

    :param interpolator: scipy interp1d object

    :returns ufunclike: extension of function to extrapolation
    """

    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]  # +(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]  # +(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return np.array(list(map(pointwise, np.array(xs))))

    return ufunclike


def add_rednoise(psr, components=10, userSpec=None, tspan=None, seed=None):
    """
    Add red noise with P(f) = A^2 / (12 pi^2) (f year)^-gamma,
    using `components` Fourier bases.
    Optionally take a pseudorandom-number-generator seed.
    :param psr: libstempo pulsar object.
    :param A: Power law red noise amplitude at 1/yr.
    :param gamma: Power law red noise spectral index.
    :param components: Number of frequency components to use in red noise
        injection.
    :param tspan: Time span to use for red noise injection [days]. Frequencies
        are injected from 1/tspan to components/tspan. Time span of pulsar used
        when not furnished.
    :param seed: Random number seed for reproducibility.
    """

    if seed is not None:
        np.random.seed(seed)

    t = psr.toas()
    minx, maxx = np.min(t), np.max(t)
    if tspan is None:
        x = (t - minx) / (maxx - minx)
        T = (day / year) * (maxx - minx)
    else:
        x = (t - minx) / tspan
        T = (day / year) * tspan

    size = 2 * components
    F = np.zeros((psr.nobs, size), "d")
    f = np.zeros(size, "d")

    for i in range(components):
        F[:, 2 * i] = np.cos(2 * math.pi * (i + 1) * x)
        F[:, 2 * i + 1] = np.sin(2 * math.pi * (i + 1) * x)

        f[2 * i] = f[2 * i + 1] = (i + 1) / T
    
    freqs = userSpec[:, 0]
    freqs = freqs * year
    if len(userSpec[:, 0]) != len(freqs):
         raise ValueError("Number of supplied spectral points does not match number of frequencies!")
    else:
         fspec_in = interp.interp1d(np.log10(freqs), np.log10(userSpec[:, 1]), kind="linear")
         fspec_ex = extrap1d(fspec_in)
         hcf = 10.0 ** fspec_ex(np.log10(f))
         
    hcf_cr = np.sqrt(3. * H0**2 * hcf / 2 / np.pi**2) #inject in Omega, converts to hcf
    
    norm = hcf_cr**2 * year**4/ 12/ np.pi**2/f**5 * 1/T
    prior = norm

    y = np.sqrt(prior) * np.random.randn(size)
    
    psr.stoas[:] += (1.0 / day) * np.dot(F, y)


def computeORFMatrix(psr):
    """
    Compute ORF matrix.

    :param psr: List of pulsar object instances

    :returns: Matrix that has the ORF values for every pulsar
             pair with 2 on the diagonals to account for the
             pulsar term.

    """

    # begin loop over all pulsar pairs and calculate ORF
    npsr = len(psr)
    ORF = N.zeros((npsr, npsr))
    phati = N.zeros(3)
    phatj = N.zeros(3)
    ptheta = [N.pi / 2 - p["DECJ"].val for p in psr]
    pphi = [p["RAJ"].val for p in psr]
    for ll in range(0, npsr):
        phati[0] = N.cos(pphi[ll]) * N.sin(ptheta[ll])
        phati[1] = N.sin(pphi[ll]) * N.sin(ptheta[ll])
        phati[2] = N.cos(ptheta[ll])

        for kk in range(0, npsr):
            phatj[0] = N.cos(pphi[kk]) * N.sin(ptheta[kk])
            phatj[1] = N.sin(pphi[kk]) * N.sin(ptheta[kk])
            phatj[2] = N.cos(ptheta[kk])

            if ll != kk:
                xip = (1.0 - N.sum(phati * phatj)) / 2.0
                ORF[ll, kk] = 3.0 * (1.0 / 3.0 + xip * (N.log(xip) - 1.0 / 6.0))
            else:
                ORF[ll, kk] = 2.0

    return ORF

        
def createFreq(
    psr,
    seed=None,
    npts=600,
    howml=1
):
    """
    Function to create GW-induced residuals from a stochastic GWB as defined
    in Chamberlin, Creighton, Demorest, et al. (2014).

    :param psr: pulsar object for single pulsar
    :param Amp: Amplitude of red noise in GW units
    :param gam: Red noise power law spectral index
    :param noCorr: Add red noise with no spatial correlations
    :param seed: Random number seed
    :param turnover: Produce spectrum with turnover at frequency f0
    :param clm: coefficients of spherical harmonic decomposition of GW power
    :param lmax: maximum multipole of GW power decomposition
    :param f0: Frequency of spectrum turnover
    :param beta: Spectral index of power spectram for f << f0
    :param power: Fudge factor for flatness of spectrum turnover
    :param userSpec: User-supplied characteristic strain spectrum
                     (first column is freqs, second is spectrum)
    :param npts: Number of points used in interpolation
    :param howml: Lowest frequency is 1/(howml * T)

    :returns: list of residuals for each pulsar
    """

    if seed is not None:
        np.random.seed(seed)

    # number of pulsars
    Npulsars = len(psr)

    # gw start and end times for entire data set
    start = np.min([p.toas().min() * 86400 for p in psr])# - 86400
    stop = np.max([p.toas().max() * 86400 for p in psr])# + 86400

    # duration of the signal
    dur = stop - start

    # get maximum number of points
    if npts is None:
        # default to cadence of 2 weeks
        npts = dur / (86400 * 14)

    # make a vector of evenly sampled data points
    ut = np.linspace(start, stop, npts)

    # time resolution in days
    dt = dur / npts

    # Define frequencies spanning from DC to Nyquist.
    # This is a vector spanning these frequencies in increments of 1/(dur*howml).
    f = np.arange(1/ dur, 1 / (2 * dt), 1 / (dur * howml))
    f[0] = f[1]  # avoid divide by 0 warning
    Nf = len(f)
    
    return f
    
    
parfiles = sorted(glob.glob(datadir + '*.par'))
Npsr = len(parfiles)
print(parfiles)

psrs = []


for ii in range(0,Npsr):

    # years of observations>
    psr = LT.fakepulsar(parfile=parfiles[ii],
            obstimes=np.arange(53000,53000+10*365.25,28.), toaerr=0.001)

    # We now remove the computed residuals from the TOAs, obtaining (in effect) a perfect realization of the deterministic timing model. The pulsar parameters will have changed somewhat, so `make_ideal` calls `fit()` on the pulsar object.
    LT.make_ideal(psr)

    #Generate white noise
    LT.add_efac(psr,efac=1.0)
    
#    add_rednoise(psr, 2e-14, 2.1)

    # add to list
    psrs.append(psr)
    

#LT.createGWB(psrs, Amp=Amp, gam=gamma, howml=100)

#make spectrum
#define the spectrum

#r = 10**(-13.1)
nt = nt
amp = amp
#fstar = 7.7*1e-17
howml=10
freq = createFreq(psrs, howml=howml)

if add_spec is None:
     spec = amp * (freq/const.fyr)**nt 
     userSpec = np.asarray([freq, spec]).T

else:
    freq_orig, spec_orig = np.genfromtxt(add_spec, unpack=True)
    f_interp = interp.interp1d(freq_orig, spec_orig)
    spec = f_interp(freq.astype(np.float64))
    userSpec = np.asarray([freq, spec]).T


#spec = 100*np.genfromtxt("exact.txt")
#np.savetxt("freq.txt", freq)

#spec = 1e-50*np.ones(len(freq))
#spec[2*howml] = 3e-5*np.ones(1)
#spec[3*howml] = 3e-5*np.ones(1)

#userSpec is in Omega_GW units; freq, spec

for psr in psrs:
     add_rednoise(psr, components = comp, userSpec=userSpec)
     

#for Psr in psrs:
    
#    Psr.fit()
#    Psr.savepar(datadir_out + Psr.name + '.par')
#    Psr.savetim(datadir_out + Psr.name + '.tim')
#    T.purgetim(datadir_out + Psr.name + '.tim')


#for Psr in psrs:

#    Psr.savepar(datadir + Psr.name + '.par')
#    Psr.savetim(datadir + Psr.name + '.tim')
#    T.purgetim(datadir + Psr.name + '.tim')
    
    
#parfiles = sorted(glob.glob(datadir + '*.par'))
#timfiles = sorted(glob.glob(datadir_out + '*.tim'))

#Psrs = []
#ephemeris = None
#for p, t in zip(parfiles, timfiles):
#    Psr = Pulsar(p, t, ephem=ephemeris)
#    Psrs.append(psr)

Psrs = []
for ii in psrs:
    psr = Tempo2Pulsar(ii)
    Psrs.append(psr)
    
os.system("mkdir " + datadir_out)

#injected spectrum
plt.plot(freq, spec)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Frequency, Hz", fontsize=15)
plt.ylabel("Spectrum Omega", fontsize=15)
plt.title("Injected spectrum")
plt.savefig(datadir_out + "inject.png", dpi=300)
plt.clf()
    
# find the maximum time span to set GW frequency sampling
Tspan = model_utils.get_tspan(Psrs)
#start = np.min([p.toas().min() * 86400 for p in Psrs])# - 86400
#stop = np.max([p.toas().max() * 86400 for p in Psrs])# + 86400

# duration of the signal
#Tspan = stop - start

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
                                   components=comp, name='gw_crn', orf = 'crn')
#s += blocks.red_noise_block(psd='spectrum', prior='log-uniform', components=30)

# We set up the PTA object using the signal we defined above and the pulsars
pta = signal_base.PTA([s(p) for p in Psrs])

def run_sampler(pta, iter_num, outdir = ''):

    N = int(iter_num)                                    # number of samples
    x0 = np.hstack([p.sample() for p in pta.params])
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

#hcf = np.sqrt(3. * H0**2 * spec / 2 / np.pi**2 / freq**2)

spec_sf = np.log10(np.sqrt(spec**2/12/np.pi**2/freq**3/Tspan))

#corner.corner(chain[burn:,-4],
#                      bins =30,
#                      plot_datapoints=False, plot_density=True, 
#                      plot_contours=False,fill_contours=False,
#                      show_titles = True, use_math_text=True, verbose=True)

fs = (np.arange(comp)+1) / Tspan
parts = plt.violinplot(
    chain[burn:,:-4], positions=fs, widths=0.07*fs)
plt.plot(freq, np.log10(spec))
plt.xlabel("Frequency, Hz", fontsize=15)
plt.ylabel(r"$\Omega_{GW}$", fontsize=15)
plt.xscale("log")
plt.xlim(2e-9, 1e-7)
#plt.ylim(-11, -3)
#plt.show()
plt.savefig(datadir_out + "violin.png", dpi=300)
#plt.clf()

#calculating 1-sigma uncertainties
std_lst = np.std(chain[burn:,:-4], axis=0)
mn_lst = np.mean(chain[burn:,:-4], axis=0)
np.savetxt(datadir_out + "free_spec_" + str(iter_real) + ".txt", np.vstack((fs, mn_lst, std_lst)).T)


