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

amp = 1e-9 #amplitude at 1 year
nt = 2.4
amp_g = 10*amp #in comparison to the amplitude at 1 year
sigma_g = 0.2
centr_fr = 5e-8

def add_loggauss(freq, amp_g, sigma_g, centr_fr):
    signal = 1/freq/sigma_g*np.exp(-(np.log(freq)-np.log(centr_fr))**2/(2*sigma_g**2))
    signal = signal/np.max(signal)
    return amp_g*signal

freq = np.arange(1e-10, 1e-5, 1e-10)

spec = amp * (freq/const.fyr)**nt
gaus_sig = add_loggauss(freq, amp_g, sigma_g, centr_fr)
spec = spec + gaus_sig
np.savetxt("cust_spec.txt", np.vstack((freq, spec)).T)

plt.plot(freq, spec)
plt.xscale("log")
plt.yscale("log")
plt.show()
