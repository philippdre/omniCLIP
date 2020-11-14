"""
    omniCLIP is a CLIP-Seq peak caller

    Copyright (C) 2017 Philipp Boss

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from scipy.optimize import fmin_tnc
from scipy.special import logsumexp

from omniCLIP.omni_stat import FitBinoDirchEmmisionProbabilities as FBProb


def pred_log_lik(counts, state, EmissionParameters, single_mix=None):
    """Compute the log_likelihood for counts."""
    params = EmissionParameters['Diag_event_params']
    alpha = params['alpha'][state]

    # Selecting the appropratie function
    if EmissionParameters['diag_event_mod'] == 'DirchMult':
        fitFun = FBProb.ComputeStateProbForGeneMD_unif
    elif EmissionParameters['diag_event_mod'] == 'DirchMultK':
        fitFun = FBProb.ComputeStateProbForGeneMD_unif_rep
    else:
        return None

    # Check which function to use for prediction the log-likelihood
    if single_mix is None:
        if params['nr_mix_comp'] > 1:  # Check for multiple mix components
            # Iterate over the mixtures and sum up the probabilities
            Prob = np.zeros((params['nr_mix_comp'], counts.shape[1]))
            for curr_mix_comp in range(0, params['nr_mix_comp']):
                # Compute the MultDirch component of the probability
                Prob[curr_mix_comp, :] = fitFun(
                    counts, alpha[:, curr_mix_comp],
                    state, EmissionParameters)

                # Compute the mixture component
                Prob[curr_mix_comp, :] += np.log(
                    params['mix_comp'][state][curr_mix_comp])

            # Sum the probabilities
            Prob = logsumexp(Prob, axis=0)

        else:
            Prob = fitFun(counts, alpha[:, 0], state, EmissionParameters)

    else:  # Extract a single component from the mixture
        Prob = fitFun(counts, alpha[:, single_mix], state, EmissionParameters)

    return Prob


def estimate_multinomial_parameters(Counts, NrOfCounts,
                                    EmissionParameters, x_0):
    """Estimate the DirchMult parameters of a mixture component."""
    if len(Counts.shape) == 1:
        Counts = np.expand_dims(Counts, axis=1)
    args_TC = (Counts, NrOfCounts, EmissionParameters)
    Bounds = tuple([(1e-100, None) for i in range(0, len(x_0))])
    disp = 1 if EmissionParameters['verbosity'] > 0 else 0

    if EmissionParameters['diag_event_mod'] == 'DirchMult':
        fitFun = FBProb.MD_f_joint_vect_unif
    elif EmissionParameters['diag_event_mod'] == 'DirchMultK':
        fitFun = FBProb.MDK_f_joint_vect_unif
    else:
        return None

    alpha = fmin_tnc(
        fitFun, x_0, fprime=FBProb.MD_f_prime_joint_vect_unif,
        args=args_TC, bounds=Bounds, disp=disp, maxfun=50)[0]

    return alpha
