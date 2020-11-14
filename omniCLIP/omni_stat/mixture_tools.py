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

from copy import deepcopy
import itertools
import multiprocessing
import numpy as np
from scipy.special import logsumexp

from omniCLIP.omni_stat import diag_event_model
from omniCLIP.omni_stat import emission_prob


def em(counts, nr_of_counts, EmissionParameters, x_0=None, First=False, max_nr_iter=15, tol=0.0001, rand_sample_size=10, verbosity=1):
    """Perform the EM algorithm."""
    template_state = 3
    fg_state, bg_state = emission_prob.get_fg_and_bck_state(EmissionParameters, final_pred=True)
    check = False

    OldEmissionParameters = deepcopy(EmissionParameters)
    for curr_state in list(counts.keys()):
        # Only compute the the emission probabilities once
        if EmissionParameters['diag_bg']:
            if curr_state != fg_state:
                if True:
                    if check is True:
                        print('Using template state ' + str(curr_state))
                        EmissionParameters['Diag_event_params']['mix_comp'][curr_state] = deepcopy(EmissionParameters['Diag_event_params']['mix_comp'][template_state])
                        EmissionParameters['Diag_event_params']['alpha'][curr_state] = deepcopy(EmissionParameters['Diag_event_params']['alpha'][template_state])
                        continue
                    else:
                        print('setting template state ' + str(curr_state))
                        check = True
                        template_state = curr_state
                else:
                    template_state = 3
                    check = True
                    EmissionParameters['Diag_event_params']['mix_comp'][curr_state] = deepcopy(EmissionParameters['Diag_event_params']['mix_comp'][template_state])
                    EmissionParameters['Diag_event_params']['alpha'][curr_state] = deepcopy(EmissionParameters['Diag_event_params']['alpha'][template_state])
                    continue
        print('Estimating state ' + str(curr_state))

        curr_counts = counts[curr_state]
        curr_nr_of_counts = nr_of_counts[curr_state]

        alpha, mixtures = Parallel_estimate_mixture_params(OldEmissionParameters, curr_counts, curr_nr_of_counts, curr_state, rand_sample_size, max_nr_iter, nr_of_iter=20, stop_crit=1.0, nr_of_init=10, verbosity=verbosity)
        EmissionParameters['Diag_event_params']['alpha'][curr_state] = alpha
        EmissionParameters['Diag_event_params']['mix_comp'][curr_state] = mixtures

    return EmissionParameters


def Parallel_estimate_mixture_params(EmissionParameters, curr_counts_orig, curr_nr_of_counts_orig, curr_state, rand_sample_size, max_nr_iter, nr_of_iter=20, stop_crit=1.0, nr_of_init=10, verbosity=1):
    """Estimate the dirichlet multinomial mixture parameters."""
    # 1) Copy old parameters and use it as initialisation for the first iteration
    alphas_list = []
    mixtures_list = []
    lls_list = []
    curr_counts = deepcopy(curr_counts_orig)
    curr_nr_of_counts = deepcopy(curr_nr_of_counts_orig)

    if len(curr_counts.shape) == 1:
        curr_counts = np.expand_dims(curr_counts, axis=1)

    if np.sum(np.sum(curr_counts, axis=0) > 0) > 0:
        curr_nr_of_counts = curr_nr_of_counts[:, np.sum(curr_counts, axis=0) >0]
        curr_counts = curr_counts[:, np.sum(curr_counts, axis=0) >0]

    # Test for fitting distributions only on diag events
    if np.sum(np.sum(curr_counts, axis=0) > 10) > 10:
        curr_nr_of_counts = curr_nr_of_counts[:, np.sum(curr_counts, axis=0) > 10]
        curr_counts = curr_counts[:, np.sum(curr_counts, axis=0) > 10]

    tracks_per_rep = EmissionParameters['Diag_event_params']['alpha'][curr_state].shape[0]
    NrOfReplicates = curr_counts.shape[0] // tracks_per_rep

    if len(curr_counts.shape) == 1:
        curr_counts = np.expand_dims(curr_counts, axis=1)

    # Save old lls mixtures and alphas
    mixtures = deepcopy(EmissionParameters['Diag_event_params']['mix_comp'][curr_state])

    scored_counts = score_counts(curr_counts, curr_state, EmissionParameters)
    scored_counts += np.tile(np.log(mixtures[:, np.newaxis]), (1, scored_counts.shape[1]))
    ll = np.sum(np.sum(logsumexp(scored_counts, axis=0) + np.log(curr_nr_of_counts)))

    alphas_list.append(deepcopy(EmissionParameters['Diag_event_params']['alpha'][curr_state]))
    mixtures_list.append(deepcopy(EmissionParameters['Diag_event_params']['mix_comp'][curr_state]))
    lls_list.append(ll)

    data = zip(itertools.repeat(stop_crit), itertools.repeat(rand_sample_size), itertools.repeat(max_nr_iter), list(range(nr_of_init)), itertools.repeat(EmissionParameters), itertools.repeat(curr_state), itertools.repeat(curr_counts), itertools.repeat(curr_nr_of_counts)   )

    if EmissionParameters['nb_proc'] == 1:
        results = [Parallel_estimate_single_mixture_params(args) for args in data]
    else:
        print("Spawning processes")
        pool = multiprocessing.get_context("spawn").Pool(EmissionParameters['nb_proc'], maxtasksperchild=5)
        results = pool.imap(Parallel_estimate_single_mixture_params, data, chunksize=1)
        pool.close()
        pool.join()
        print("Collecting results")
        results = [res for res in results]

    alphas_list += [res[0] for res in results]
    mixtures_list += [res[1] for res in results]
    lls_list += [res[2] for res in results]

    # Select which alpha had the highest ll
    max_ll_pos = np.argmax(np.array(lls_list))

    alpha = alphas_list[max_ll_pos]
    mixtures = mixtures_list[max_ll_pos]
    return alpha, mixtures


def Parallel_estimate_single_mixture_params(args):
    """Estimate thedirichlet multinomial mixture parameters."""

    stop_crit, rand_sample_size, max_nr_iter, curr_init, EmissionParameters, curr_state, curr_counts, curr_nr_of_counts = args
    # Compute the curr mixture, ll and alpha
    # Initialiste the parameters
    old_ll = 0
    ll = -10

    OldAlpha = deepcopy(EmissionParameters['Diag_event_params']['alpha'][curr_state])
    mixtures = deepcopy(EmissionParameters['Diag_event_params']['mix_comp'][curr_state])

    if curr_init > 0:
        OldAlpha = np.random.uniform(low=0.0001, high=0.1, size=OldAlpha.shape)
        for i in range(OldAlpha.shape[1]):
            OldAlpha[np.random.randint(OldAlpha.shape[0]-1), i] = np.random.random() * 10.0
            OldAlpha[-2, i] = np.random.random() * 1.0
            OldAlpha[-1, i] = np.random.random() * 10.0
        mixtures = np.random.uniform(low=0.0001, high=1.0, size=mixtures.shape)
        mixtures /= np.sum(mixtures)
    if EmissionParameters['Diag_event_params']['nr_mix_comp'] == 1:
        # Case that only one mixture component is given
        EmissionParameters['Diag_event_params']['alpha'][curr_state][:, 0] = diag_event_model.estimate_multinomial_parameters(curr_counts, curr_nr_of_counts, EmissionParameters, OldAlpha[:])
        # Compute ll
        scored_counts = score_counts(curr_counts, curr_state, EmissionParameters)
        scored_counts += np.tile(np.log(mixtures[:, np.newaxis]), (1, scored_counts.shape[1]))
        ll = np.sum(np.sum(logsumexp(scored_counts, axis=0) + np.log(curr_nr_of_counts)))

        OldAlpha = deepcopy(EmissionParameters['Diag_event_params']['alpha'][curr_state])
        mixtures = deepcopy(EmissionParameters['Diag_event_params']['mix_comp'][curr_state])

    else:
        zero_ix = []
        for iter_nr in range(max_nr_iter):
            print('em-iteration ' + str(iter_nr))

            scored_counts = score_counts(curr_counts, curr_state, EmissionParameters)
            scored_counts += np.tile(np.log(mixtures[:, np.newaxis]), (1, scored_counts.shape[1]))
            # 2) Compute the mixture components
            # Compute the normalisation factor
            normalised_likelihood = logsumexp(scored_counts, axis=0)

            old_ll = ll
            ll = np.sum(np.sum(logsumexp(scored_counts, axis=0) + np.log(curr_nr_of_counts)))

            if np.abs(old_ll - ll) < stop_crit:
                if len(zero_ix) == 0:
                    break

            normalised_scores = scored_counts - np.tile(normalised_likelihood, (scored_counts.shape[0], 1))
            un_norm_mixtures = logsumexp(normalised_scores, b=np.tile(curr_nr_of_counts, (scored_counts.shape[0], 1)), axis = 1)

            mixtures = np.exp(un_norm_mixtures - logsumexp(un_norm_mixtures))

            # 3) Compute for eachcount the most likely mixture component
            curr_weights = np.exp(normalised_scores)
            curr_weights = (curr_weights == np.tile(np.max(curr_weights, axis=0), (curr_weights.shape[0], 1))) *1.0

            zero_mix = np.sum(curr_weights, axis=1) == 0
            zero_ix = np.where(zero_mix)[0].tolist()

            EmissionParameters['Diag_event_params']['mix_comp'][curr_state] = mixtures
            # Get number of positions that are used. (In case there are fewer entries that rand_sample_size in counts)
            rand_size = min(rand_sample_size, curr_counts.shape[1])
            for i in zero_ix:
                random_ix = np.random.choice(curr_counts.shape[1], rand_size, p=(curr_nr_of_counts[0, :] / np.float(np.sum(curr_nr_of_counts[0, :]))))
                curr_counts = np.hstack([curr_counts, curr_counts[:, random_ix]])
                curr_nr_of_counts = np.hstack([curr_nr_of_counts, np.ones((1, rand_size))])
                temp_array = np.zeros((normalised_scores.shape[0], rand_size))
                temp_array[i, :] = i
                normalised_scores = np.hstack([normalised_scores, temp_array])
                temp_array = np.zeros((curr_weights.shape[0], rand_size))
                temp_array[i, :] = 1
                curr_weights = np.hstack([curr_weights, temp_array])

            # 4) Compute the dirichlet-multinomial parameters
            for curr_mix_comp in range(EmissionParameters['Diag_event_params']['nr_mix_comp']):
                local_counts = curr_counts
                local_nr_counts = curr_nr_of_counts * curr_weights[curr_mix_comp, :]
                local_counts = local_counts[:, local_nr_counts[0, :] > 0]
                local_nr_counts = local_nr_counts[0, local_nr_counts[0, :] > 0]
                if len(local_counts.shape) == 1:
                    local_counts = np.expand_dims(local_counts, axis=1)
                curr_alpha = diag_event_model.estimate_multinomial_parameters(local_counts, local_nr_counts, EmissionParameters, OldAlpha[:, curr_mix_comp])

                if curr_mix_comp in zero_ix:
                    OldAlpha[:, curr_mix_comp] = np.random.uniform(low=0.0001, high=0.1, size=OldAlpha[:, curr_mix_comp].shape)
                    OldAlpha[np.random.randint(OldAlpha.shape[0]), curr_mix_comp] = np.random.random() * 10.0
                    OldAlpha[-2, curr_mix_comp] = np.random.random() * 1.0
                    OldAlpha[-1, curr_mix_comp] = np.random.random() * 10.0
                else:
                    OldAlpha[:, curr_mix_comp] = curr_alpha

            if (len(zero_ix) > 0) and (iter_nr + 2 < max_nr_iter):
                # Treat the case where some mixtures have prob zero
                mixtures[zero_ix] = np.mean(mixtures)
                mixtures /= np.sum(mixtures)
                EmissionParameters['Diag_event_params']['mix_comp'][curr_state] = deepcopy(mixtures)
            EmissionParameters['Diag_event_params']['alpha'][curr_state] = deepcopy(OldAlpha)
            # Check if convergence has been achieved.

        mixtures[zero_ix] = np.min(mixtures[mixtures > 0])
        mixtures /= np.sum(mixtures)

    return [deepcopy(OldAlpha), mixtures, ll]


def score_counts(counts, state, EmissionParameters):
    """Scores the counts for each mixture component."""

    nr_mixture_components = EmissionParameters['Diag_event_params']['nr_mix_comp']
    # Initialize the return array
    scored_counts = np.zeros((nr_mixture_components, counts.shape[1]))

    # Compute for each state the log-likelihood of the counts
    for mix_comp in range(nr_mixture_components):
        scored_counts[mix_comp, :] = diag_event_model.pred_log_lik(counts, state, EmissionParameters, single_mix=mix_comp)
        scored_counts[mix_comp, :] += np.log(EmissionParameters['Diag_event_params']['mix_comp'][state][mix_comp])

    return scored_counts
