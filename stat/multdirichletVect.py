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


from scipy.special import gammaln
import numpy as np
import pdb


#@profile 
def pdf(k, alpha):
    Ret = np.exp(log_pdf(k, alpha))
    return np.exp(Ret)


#@profile 
def log_pdf(k, alpha):
    Pos = gammaln(np.sum(k) + 1) + gammaln(np.sum(alpha)) + np.sum(gammaln(alpha + k))
    Neg = np.sum(gammaln(k + 1)) + gammaln(np.sum(alpha + k)) + np.sum(gammaln(alpha))
    log = Pos - Neg
    if np.isinf(log) or np.isnan(log):
        pdb.set_trace()
    return log


#@profile 
def TwoBinomlog_pdf(k1, k2, alpha):
    '''
    This is the pdf for the case when two multinomial distributions are observed
    '''

    k = k1 + k2
    Pos = gammaln(np.sum(k1) + 1) + gammaln(np.sum(k2) + 1) + gammaln(np.sum(alpha)) + np.sum(gammaln(alpha + k))
    Neg = np.sum(gammaln(k1 + 1)) + np.sum(gammaln(k2 + 1)) + gammaln(np.sum(alpha + k)) + np.sum(gammaln(alpha))
    log = Pos - Neg

    return log


#@profile 
def log_pdf_vect(k, alpha):
    '''
    This function computes the log pdf for an array of counts
    '''

    #First check whether the input is one dimnesional and set parameters acordingly
    alpha = np.expand_dims(alpha, axis=1)
    if len(k.shape) == 1:
        Ks = 1
        k = np.expand_dims(k, axis=1)
    else:
        Ks = k.shape[1]
    Pos = gammaln(np.sum(k, axis=0) + 1) + np.tile(gammaln(np.sum(alpha)), (1, Ks)) + np.sum(gammaln(np.tile(alpha, (1, Ks)) + k), axis=0)
    Neg = np.sum(gammaln(k + 1), axis=0) + gammaln(np.sum(np.tile(alpha, (1, Ks)) + k, axis=0)) + np.tile(np.sum(gammaln(alpha)), (1, Ks))
    log = Pos - Neg

    return log


#@profile 
def expand_k(k):
    '''
    Thid function adds one dimension to k if k is only a 1-dim  array
    '''

    if len(k.shape) == 1:
        Ks = 1
        k = np.expand_dims(k, axis=1)
    else:
        Ks = k.shape[1]
    return k, Ks

##@profile
#@profile 
def log_pdf_vect_rep(Counts, alpha, tracks_per_rep, NrOfReplicates):
    '''
    This function computes the log pdf for an array of counts
    '''

    #First check whether the input is one dimnesional and set parameters acordingly
    alpha = np.expand_dims(alpha, axis=1)

    #Compute the 'collapsed' counts per diagnostic events
    k = Counts[0 : tracks_per_rep, :].copy()
    k, Ks = expand_k(k)
    for rep in range(1, int(NrOfReplicates)):  
        new_k, Ks = expand_k(Counts[rep * tracks_per_rep:(rep + 1) * tracks_per_rep, :])
        k += new_k

    #Compute the factors that are independent of the replicates
    Pos = np.sum(gammaln(k + np.tile(alpha, (1, Ks))), axis=0) + np.tile(gammaln(np.sum(alpha)), (1, Ks)) 
    Neg = np.tile(np.sum(gammaln(alpha)), (1, Ks)) + gammaln(np.sum(np.tile(alpha, (1, Ks)) + k, axis=0))

    for rep in range(int(NrOfReplicates)):
        k = Counts[rep * tracks_per_rep:(rep + 1) * tracks_per_rep, :].copy()
        k, Ks = expand_k(k)
        Pos += gammaln(np.sum(k, axis=0) + 1)
        Neg += np.sum(gammaln(k + 1), axis=0)

    log = Pos - Neg

    return log

##@profile
#@profile 
def TwoBinomlog_pdf_vect(k1, k2, alpha):
    '''
    This is the pdf for an array of counts for the case when two multinomial distributions are observed
    '''

    alpha = np.expand_dims(alpha, axis=1)
    if len(k1.shape) == 1:
        Ks = 1
        k1 = np.expand_dims(k1, axis=1)
        k2 = np.expand_dims(k2, axis=1)
    else:
        Ks = k1.shape[1]
    k = k1 + k2
    Pos = gammaln(np.sum(k1, axis=0) + 1) + gammaln(np.sum(k2, axis=0) + 1) + np.tile(gammaln(np.sum(alpha)),(1,Ks)) + np.sum(gammaln(np.tile(alpha, (1, Ks)) + k), axis=0)
    Neg = np.sum(gammaln(k1 + 1), axis=0) + np.sum(gammaln(k2 + 1), axis=0) + gammaln(np.sum(np.tile(alpha, (1, Ks)) + k, axis=0)) + np.tile(np.sum(gammaln(alpha)), (1, Ks))
    log = Pos - Neg
    return log


#@profile 
def __init__(self, Parameters):
    self.name = 'MultDirichlet'
    self.parameters = [Parameters]
    self.frozen = frozen
    def log_probability(self, sample):
        alpha = self.parameters[0]
        Pos = gammaln(np.sum(k) + 1) + gammaln(np.sum(alpha)) + np.sum(gammaln(alpha + k))
        Neg = np.sum(gammaln(k + 1)) + gammaln(np.sum(alpha + k)) + np.sum(gammaln(alpha))
        Log = Pos - Neg
        return Log
    def from_sample(self, items, weights=None):
        if weights is None:
            weights = numpy.ones_like(items, dtype=float)
        self.parameters = [float(numpy.dot(items, weights)) / len(items)]
    def sample(self):
        
        return random.random() < self.parameters[0]



