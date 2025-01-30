import numpy as np
import scipy as sp

from .lanczos import *
from .nystrom_pcg import *
from .sols import *
from .sqrt import *

import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams['text.latex.preamble'] = r'\usepackage{newtxtext}\usepackage[scaled=.9]{DejaVuSansMono}'
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

c0='#1d1d1d'; c1='#1e3264'; c2='#c82336'; c3='#198c71'; c4='#ef9646'; c5='#1ca9d2'