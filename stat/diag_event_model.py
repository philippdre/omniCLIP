'''
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
'''


import FitBinoDirchEmmisionProbabilities
import numpy as np
from scipy.optimize import fmin_tnc
from scipy.special import logsumexp

##@profile
#@profile 
def pred_log_lik(counts, state, EmissionParameters, single_mix=None):
	'''
	This function computes the log_likelihood for counts
	'''
	
	alpha = EmissionParameters['Diag_event_params']['alpha'][state]

	#Check which function to use for prediction the log-likelihood
	if single_mix == None:
		if EmissionParameters['Diag_event_params']['nr_mix_comp'] > 1: #Check whether multiple mixuter components are used 
			if EmissionParameters['Diag_event_type'] == 'DirchMult':
				#Iterate over the mixtures and sum up the probabilities

				#Initialise the array
				Prob = np.zeros((EmissionParameters['Diag_event_params']['nr_mix_comp'], counts.shape[1]))
				for curr_mix_comp in range(0, EmissionParameters['Diag_event_params']['nr_mix_comp']):
					#Compute the MultDirch component of the probability
					Prob[curr_mix_comp, :] = FitBinoDirchEmmisionProbabilities.ComputeStateProbForGeneMD_unif(counts, alpha[:, curr_mix_comp], state, EmissionParameters)
					#compute the mixture component
					Prob[curr_mix_comp, :] += np.log(EmissionParameters['Diag_event_params']['mix_comp'][state][curr_mix_comp])

				#Sum the probabilities				
				Prob = logsumexp(Prob, axis=0)
			elif EmissionParameters['Diag_event_type'] == 'DirchMultK':
				#Iterate over the mixtures and sum up the probabilities

				#Initialise the array
				Prob = np.zeros((EmissionParameters['Diag_event_params']['nr_mix_comp'], counts.shape[1]))
				for curr_mix_comp in range(0, EmissionParameters['Diag_event_params']['nr_mix_comp']):
					#Compute the MultDirch component of the probability
					Prob[curr_mix_comp, :] = FitBinoDirchEmmisionProbabilities.ComputeStateProbForGeneMD_unif_rep(counts, alpha[:, curr_mix_comp], state, EmissionParameters)
					#compute the mixture component
					Prob[curr_mix_comp, :] += np.log(EmissionParameters['Diag_event_params']['mix_comp'][state][curr_mix_comp])

				#Sum the probabilities	
				Prob = logsumexp(Prob, axis=0)
				
			else:
				Prob = None
		else:
			if EmissionParameters['Diag_event_type'] == 'DirchMult':
				Prob = FitBinoDirchEmmisionProbabilities.ComputeStateProbForGeneMD_unif(counts, alpha[:, 0], state, EmissionParameters)
			elif EmissionParameters['Diag_event_type'] == 'DirchMultK':
				Prob = FitBinoDirchEmmisionProbabilities.ComputeStateProbForGeneMD_unif_rep(counts, alpha[:, 0], state, EmissionParameters)
			else:
				Prob = None
	else: #Extract a single component from the mixture
		if EmissionParameters['Diag_event_type'] == 'DirchMult':
			Prob = FitBinoDirchEmmisionProbabilities.ComputeStateProbForGeneMD_unif(counts, alpha[:, single_mix], state, EmissionParameters)
		elif EmissionParameters['Diag_event_type'] == 'DirchMultK':
			Prob = FitBinoDirchEmmisionProbabilities.ComputeStateProbForGeneMD_unif_rep(counts, alpha[:, single_mix], state, EmissionParameters)
		else:
			Prob = None

	return Prob

##@profile
#@profile 
def estimate_multinomial_parameters(Counts, NrOfCounts, EmissionParameters, OldAlpha):
	'''
	This function estimates for a mixture component the DirchMult parameters
	'''

	x_0  = OldAlpha
	if len(Counts.shape) == 1:
		Counts = np.expand_dims(Counts, axis=1)
	args_TC = (Counts, NrOfCounts, EmissionParameters)
	Bounds = tuple([(1e-100, None) for i in range(0,len(x_0))])
	if EmissionParameters['Verbosity'] > 0:
		disp = 1
	else:
		disp = 0
	if EmissionParameters['Diag_event_type'] == 'DirchMult':
		alpha = np.zeros_like(x_0)
		alpha = fmin_tnc(FitBinoDirchEmmisionProbabilities.MD_f_joint_vect_unif, x_0, fprime=FitBinoDirchEmmisionProbabilities.MD_f_prime_joint_vect_unif, args=args_TC, bounds=Bounds, disp=disp, maxfun=50)[0]
	elif EmissionParameters['Diag_event_type'] == 'DirchMultK':
		alpha = np.zeros_like(x_0)
		alpha = fmin_tnc(FitBinoDirchEmmisionProbabilities.MDK_f_joint_vect_unif, x_0, fprime=FitBinoDirchEmmisionProbabilities.MD_f_prime_joint_vect_unif, args=args_TC, bounds=Bounds, disp=disp, maxfun=50)[0]
	else:
		alpha = None

	return alpha
