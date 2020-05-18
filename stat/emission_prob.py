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


from scipy import special
from scipy.sparse import *
from scipy.sparse import csc_matrix
from scipy.stats import nbinom
import h5py
import itertools
import multiprocessing
import numpy as np
import resource
import scipy as sp
import sparse_glm
import time


 
def ComputeNBMeanVar(ExprPar):
    '''
    Compute the mean and the variance of a NB distribution with parameter n and p
    '''

    n = ExprPar[1]
    p = ExprPar[0]

    M = nbinom.mean(n, p)
    V = nbinom.var(n , p)
    return M, V


 
def EstimateExpressionParametersMix(NewEmissionParameters, Counts, NrOfCounts, NrOfStates, verbosity=1):
    '''
    This function determines the mean and variances of the counts for all states
    '''

    #Compute the mean and the variances for the states
    Means = {}
    Vars = {}
    for State in [0,1]:
        Means[State] = [0,0]
        Vars[State] = [0,0]
        for j in range(2):#Iterate over replicates
            vals = np.sum(Counts[State][list(range(j,6,2)), :], axis = 0)
            weights = NrOfCounts[State]
            #Compute the weigted average and weighted squared average
            Means[State][j] = np.sum(vals * weights) / np.sum(weights)
            SqrdMean = np.sum((vals ** 2) * weights) / np.sum(weights)
            Vars[State][j] = SqrdMean - (Means[State][j] ** 2)
            if Vars[State][j] < 0.000001:
                Vars[State][j] = 0.000001
            if Means[State][j] < 0.0000001:
                Means[State][j] = 0.0000001
            if Means[State][j] >= Vars[State][j]:
                Vars[State][j] = Means[State][j] + 0.0000001

    #Compute the paramaeters from the mean and average
    NewEmissionParameters['ExpBck'] = [np.max(np.sum(Counts[State][list(range(0,6,2)), :], axis = 0)), np.max(np.sum(Counts[State][list(range(1,6,2)), :], axis = 0))]
    NewEmissionParameters['ExpTC'] = [NB_parameter_estimation(Means[1][0], Vars[1][0]), NB_parameter_estimation(Means[1][1], Vars[1][1])]
    NewEmissionParameters['ExpNonTC'] = [NB_parameter_estimation(Means[0][0], Vars[0][0]), NB_parameter_estimation(Means[0][1], Vars[0][1])]

    M, V = ComputeNBMeanVar(NewEmissionParameters['ExpTC'][0])
    if verbosity > 1:
        print("Mean expr TC: " + str(M))
        print("Var expr TC: " + str(V))
    M, V = ComputeNBMeanVar(NewEmissionParameters['ExpNonTC'][0])
    if verbosity > 1:
        print("Mean expr NonTC: " + str(M))
        print("Var expr NonTC: " + str(V))

        print("Max Bck: " + str(NewEmissionParameters['ExpBck'][0]))

    return NewEmissionParameters


 
def EstimateExpressionParametersNB(NewEmissionParameters, Counts, NrOfCounts, NrOfStates, verbosity=1):
    '''
    This function determines the mean and variances of the counts for all states
    '''

    #Compute the mean and the variances for the states
    Means = {}
    Vars = {}
    for State in range(NrOfStates):
        Means[State] = [0,0]
        Vars[State] = [0,0]
        for j in range(2):#Iterate over replicates
            vals = np.sum(Counts[State][list(range(j,6,2)), :], axis = 0)
            weights = NrOfCounts[State]
            #pdb.set_trace()
            if State >3:
                sort_ix = np.argsort(vals)
                cutoff = round(np.sum(weights) * 0.01)
                #pdb.set_trace()
                curr_pos = len(vals) - 1
                while cutoff > 0:
                    if weights[0, sort_ix[curr_pos]] >= cutoff:
                        weights[0, sort_ix[curr_pos]] -= cutoff
                        cutoff = 0
                    else:
                        cutoff -= weights[0, sort_ix[curr_pos]]
                        weights[0, sort_ix[curr_pos]] = 0
                    curr_pos -= 1
                zero_ix =  (weights > 0 )[0,:]
                vals = vals[zero_ix]
                weights = weights[0, zero_ix]

            #Compute the weigted average and weighted squared average
            Means[State][j] = np.sum(vals * weights) / np.sum(weights)
            SqrdMean = np.sum((vals ** 2) * weights) / np.sum(weights)
            Vars[State][j] = SqrdMean - (Means[State][j] ** 2)
            if Vars[State][j] < 0.000001:
                Vars[State][j] = 0.000001
            if Means[State][j] < 0.0000001:
                Means[State][j] = 0.0000001
            if Means[State][j] >= Vars[State][j]:
                Vars[State][j] = Means[State][j] + 0.0000001

    #Compute the paramaeters from the mean and average
    NewEmissionParameters['ExpBck'] = [NB_parameter_estimation(Means[2][0], Vars[2][0]), NB_parameter_estimation(Means[2][1], Vars[2][1])]
    NewEmissionParameters['ExpTC'] = [NB_parameter_estimation(Means[1][0], Vars[1][0]), NB_parameter_estimation(Means[1][1], Vars[1][1])]
    NewEmissionParameters['ExpNonTC'] = [NB_parameter_estimation(Means[0][0], Vars[0][0]), NB_parameter_estimation(Means[0][1], Vars[0][1])]

    M, V = ComputeNBMeanVar(NewEmissionParameters['ExpTC'][0])
    if verbosity > 1:
        print("Mean expr TC: " + str(M))
        print("Var expr TC: " + str(V))
    M, V = ComputeNBMeanVar(NewEmissionParameters['ExpNonTC'][0])
    if verbosity > 1:
        print("Mean expr NonTC: " + str(M))
        print("Var expr NonTC: " + str(V))
    M, V = ComputeNBMeanVar(NewEmissionParameters['ExpBck'][0])
    if verbosity > 1:
        print("Mean expr Bck: " + str(M))
        print("Var expr Bck: " + str(V))

    return NewEmissionParameters


 
def EstimateExpressionParametersNBJoint(NewEmissionParameters, Counts, NrOfCounts, NrOfStates):
    '''
    This function determines the mean and variances of the counts for all states
    '''

    #Compute the mean and the variances for the states
    Means = {}
    Vars = {}
    for State in [1,2]:
        Means[State] = [0,0]
        Vars[State] = [0,0]
        for j in range(2):#Iterate over replicates
            if State == 1 or State == 0:
                #pdb.set_trace()
                vals = np.hstack((Counts[1][j, :], Counts[0][j, :]))
                weights = np.hstack((NrOfCounts[1], NrOfCounts[0]))[0, :]
            else:
                vals = Counts[State][j, :]
                weights = NrOfCounts[State][0, :]
            if State >3:
                sort_ix = np.argsort(vals)
                cutoff = round(np.sum(weights) * 0.01)
                curr_pos = len(vals) - 1
                while cutoff > 0:
                    if weights[0, sort_ix[curr_pos]] >= cutoff:
                        weights[0, sort_ix[curr_pos]] -= cutoff
                        cutoff = 0
                    else:
                        cutoff -= weights[0, sort_ix[curr_pos]]
                        weights[0, sort_ix[curr_pos]] = 0
                    curr_pos -= 1
                zero_ix =  (weights > 0 )[0,:]
                vals = vals[zero_ix]
                weights = weights[0, zero_ix]

            #Compute the weigted average and weighted squared average
            Means[State][j] = np.sum(vals * weights) / np.sum(weights)
            SqrdMean = np.sum((vals ** 2) * weights) / np.sum(weights)
            Vars[State][j] = SqrdMeanean - (Means[State][j] ** 2)
            if Vars[State][j] < 0.000001:
                Vars[State][j] = 0.000001
            if Means[State][j] < 0.0000001:
                Means[State][j] = 0.0000001
            if Means[State][j] >= Vars[State][j]:
                Vars[State][j] = Means[State][j] + 0.0000001
    
    #Compute the paramaeters from the mean and average
    NewEmissionParameters['ExpBck'] = [NB_parameter_estimation(Means[2][0], Vars[2][0]), NB_parameter_estimation(Means[2][1], Vars[2][1])]
    NewEmissionParameters['ExpTC'] = [NB_parameter_estimation(Means[1][0], Vars[1][0]), NB_parameter_estimation(Means[1][1], Vars[1][1])]
    NewEmissionParameters['ExpNonTC'] = [NB_parameter_estimation(Means[1][0], Vars[1][0]), NB_parameter_estimation(Means[1][1], Vars[1][1])]

    M, V = ComputeNBMeanVar(NewEmissionParameters['ExpTC'][0])
    if verbosity > 1:
        print("Mean expr TC: " + str(M))
        print("Var expr TC: " + str(V))
    M, V = ComputeNBMeanVar(NewEmissionParameters['ExpNonTC'][0])
    if verbosity > 1:
        print("Mean expr NonTC: " + str(M))
        print("Var expr NonTC: " + str(V))
    M, V = ComputeNBMeanVar(NewEmissionParameters['ExpBck'][0])
    if verbosity > 1:
        print("Mean expr Bck: " + str(M))
        print("Var expr Bck: " + str(V))

    return NewEmissionParameters


 
def NB_parameter_estimation(mean, var):
    '''
    This function computes the parameters p and r for the negative binomial distribution
    '''

    mean = np.float64(mean)
    var = np.float64(var)
    p = mean / var

    n = (mean ** 2) / (var - mean)

    return p, n


 
def estimate_expression_param(expr_data, verbosity=1):
    EmissionParameters, Sequences, Background, Paths, sample_size, bg_type = expr_data 
    '''
    This function estimates the parameters for the expression GLM
    '''
    print ('Start estimation of expression parameters')
    # 1) Get the library size
    bg_type = EmissionParameters['BckType']
    lib_size = EmissionParameters['LibrarySize']
    bck_lib_size = EmissionParameters['BckLibrarySize']
    start_params = EmissionParameters['ExpressionParameters'][0]
    disp = EmissionParameters['ExpressionParameters'][1]


    # 2) Estimate dispersion
    print ('Constructing GLM matrix')
    t = time.time() 
    # 3) Compute sufficient statistics
    if verbosity > 0:
        print('Estimating expression parameters: before GLM matrix construction')
        print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    try:
        Sequences.close()
    except:
        pass
    try:
        Background.close()
    except:
        pass     

    Sequences = h5py.File(EmissionParameters['DataOutFile_seq'], 'r')
    Background = h5py.File(EmissionParameters['DataOutFile_bck'], 'r')

    A, w, Y, rep = construct_glm_matrix(EmissionParameters, Sequences, Background, Paths)

    print('Estimating expression parameters: GLM matrix constrution')
    if verbosity > 0:
        print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        #pdb.set_trace()
        print('Done: Elapsed time: ' + str(time.time() - t))
    
    #make sure that matrix A is in the right format
    if not sp.sparse.isspmatrix_csc(A):
        A = csc_matrix(A)

    if verbosity > 0:
        print('Estimating expression parameters: before GLM matrix')
        print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    #create the offset for the library size
    offset = np.zeros_like(rep)
    for i in range(EmissionParameters['NrOfReplicates']):
        offset[rep == (i + 1)] = lib_size[str(i)]
    if bg_type != 'None':
        for i in range(EmissionParameters['NrOfBckReplicates']):
            offset[rep == -(i + 1)] = bck_lib_size[str(i)]

    # 4) Fit GLM
    print ('Fitting GLM')
    t = time.time()

    print('Estimating expression parameters: before fitting')
    if verbosity > 0:
        print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    start_params, disp = fit_glm(A, w, Y, offset, sample_size, disp, start_params, norm_class=EmissionParameters['norm_class'])

    if verbosity > 0:
        print('Estimating expression parameters: after fitting')
        print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    del A, w, Y, offset

    if verbosity > 0:
        print('Estimating expression parameters: after cleanup')
    
        print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        print('Done: Elapsed time: ' + str(time.time() - t))
    
    # 5) Process the output
    EmissionParameters['ExpressionParameters'] = [start_params, disp]
    print ('Finished expression parameter estimation')

    return EmissionParameters


 
def get_gene_expr(Sequences, Background, sample_size, EmissionParameters):
    '''
    This function gets the gene expressions
    '''

    exp_mat = np.zeros((EmissionParameters['NrOfReplicates'], sample_size))

    #Determine number of nonzero positions in genes
    nr_non_zeros = np.zeros(len(list(Sequences.keys())))
    gene_ix = {}
    for i, gene in enumerate(Sequences.keys()):
        gene_ix[gene] = i 
        counts = tools.StackData(Sequences, gene, add = 'all')
        nr_non_zeros[i] = np.sum(np.sum(counts, axis = 0) > 0)

    #Define which genes to sample
    sample = np.random.multinomial(sample_size, nr_non_zeros/np.sum(nr_non_zeros))

    #Sample the expressions
    curr_counter = 0
    for j in range(len(sample)):
        if sample[j] == 0:
            continue
        temp_counts = counts[:, np.sum(counts, axis = 0) > 0]
        pos = np.random.choice(list(range(temp_counts.shape[1])), sample[j])
        pos.sort()
        exp_mat[:, curr_counter:(curr_counter + sample[j])] = temp_counts[:, pos]
        curr_counter += sample[j]

    return exp_mat


 
def estimate_disp_from_replicates(exp_mat, lib_size):
    ''' 
    This function estimates the dispersion for a set of replicates
    '''
    lib_size_vect = np.zeros(len(list(lib_size.keys())))
    for i, key in enumerate(lib_size.keys()):
        lib_size_vect[i] = lib_size[key]

    return


 
def construct_glm_matrix(EmissionParameters, Sequences, Background, Paths, bg_type = None):

    #Determine shape of the matrix
    bg_type = EmissionParameters['BckType']
    nr_of_bck_rep = EmissionParameters['NrOfBckReplicates']
    nr_of_rep = EmissionParameters['NrOfReplicates']
    NrOfStates = EmissionParameters['NrOfStates']
    
    curr_mats = []
    curr_weights = []
    curr_ys = []
    curr_reps = []
    
    nr_of_genes = len(list(Sequences.keys()))

    np_proc = EmissionParameters['NbProc']
    number_of_processes = np_proc

    fg_state, bg_state = get_fg_and_bck_state(EmissionParameters)
    #Create a dictionary of unique rows for each gene
    #Check if parallel processing is activated
    replicates_fg = list(Sequences[list(Sequences.keys())[0]]['Coverage'].keys())
    if np_proc == 1:
        for gene_nr, gene in enumerate(Sequences):
            for rep, curr_rep in enumerate(replicates_fg):
                #Create the sparse matrix blocks for each replicate
                curr_data = (Paths[gene], Sequences[gene]['Coverage'][curr_rep][()], gene, gene_nr, rep, NrOfStates, nr_of_genes, bg_type, fg_state, bg_state, EmissionParameters['Verbosity'])
                gene_mat, weights, y, reps, new_pos = process_gene_for_glm_mat(curr_data)
                del curr_data
                curr_mats.append(gene_mat)
                curr_weights.append(weights)
                curr_ys.append(y)
                curr_reps.append(reps)
    else:
        #Create a function that groups together the values such that an iterator can be defined
        verb = EmissionParameters['Verbosity']
        f = lambda gene_nr, gene, rep,  Paths=Paths, Sequences=Sequences, NrOfStates=NrOfStates, nr_of_genes=nr_of_genes, bg_type=bg_type, fg_state=fg_state, bg_state=bg_state, verb=verb: (Paths[gene], Sequences[gene]['Coverage'][str(rep)][()], gene, gene_nr, rep, NrOfStates, nr_of_genes, bg_type, fg_state, bg_state, verb)
        #Create an iterator for the data
        list_gen = [(a, b, c) for (a, b) ,c  in itertools.product(zip(itertools.count(),list(Sequences.keys())), list(range(nr_of_rep)))]
        data = itertools.starmap(f, list_gen)
        pool = multiprocessing.get_context("spawn").Pool(number_of_processes, maxtasksperchild=100)
        results = pool.imap(process_gene_for_glm_mat, data, chunksize=1)
        pool.close()
        pool.join()
        results = [res for res in results]
        curr_mats += [res[0] for res in results]
        curr_weights += [res[1] for res in results]
        curr_ys += [res[2] for res in results]
        curr_reps += [res[3] for res in results]
        del results
    
    #Add row with pseudo counts

    if not bg_type == 'None':
        replicates_bck = list(Background[list(Background.keys())[0]]['Coverage'].keys())
        #Process the background
        if np_proc == 1:
            for gene_nr, gene in enumerate(Sequences):
                for rep, curr_rep in enumerate(replicates_bck):
                    if bg_type == 'Const':
                        gene_rep_background = Background[gene][curr_rep][()]
                    else:
                        gene_rep_background = Background[gene]['Coverage'][curr_rep][()]

                    curr_data = (Paths[gene], gene_rep_background, gene, gene_nr, rep, NrOfStates, nr_of_genes, bg_type, fg_state, bg_state)
                    gene_mat, weights, y, reps, new_pos = process_bck_gene_for_glm_mat(curr_data)
                    curr_mats.append(gene_mat)
                    curr_weights.append(weights)
                    curr_ys.append(y)
                    curr_reps.append(reps)
        else:
            #Create a function that groups together the values such that an iterator can be defined
            if bg_type == 'Const':
                f = lambda gene_nr, gene, rep,  Paths=Paths, Background=Background, NrOfStates=NrOfStates, nr_of_genes=nr_of_genes, bg_type=bg_type, fg_state=fg_state, bg_state=bg_state: (Paths[gene], Background[gene][str(rep)][()], gene, gene_nr, rep, NrOfStates, nr_of_genes, bg_type, fg_state, bg_state)
            else:
                f = lambda gene_nr, gene, rep,  Paths=Paths, Background=Background, NrOfStates=NrOfStates, nr_of_genes=nr_of_genes, bg_type=bg_type, fg_state=fg_state, bg_state=bg_state: (Paths[gene], Background[gene]['Coverage'][str(rep)][()], gene, gene_nr, rep, NrOfStates, nr_of_genes, bg_type, fg_state, bg_state)
            
            #Create an iterator for the data            
            list_gen = [(a, b, c) for (a, b) ,c  in itertools.product(zip(itertools.count(),list(Background.keys())), list(range(nr_of_bck_rep)))]
            data = itertools.starmap(f, list_gen)
            pool = multiprocessing.get_context("spawn").Pool(number_of_processes, maxtasksperchild=100)
            results = pool.imap(process_bck_gene_for_glm_mat, data, chunksize=1)
            pool.close()
            pool.join()
            results = [res for res in results]
            curr_mats += [res[0] for res in results]
            curr_weights += [res[1] for res in results]
            curr_ys += [res[2] for res in results]
            curr_reps += [res[3] for res in results]
            del results       

    #remove empty elements
    ix_nz = [curr_mats[i].nnz > 0 for i in range(len(curr_mats))]
    curr_mats = [curr_mats[i] for i in range(len(ix_nz)) if ix_nz[i]]
    curr_weights = [curr_weights[i] for i in range(len(ix_nz)) if ix_nz[i]]
    curr_ys = [curr_ys[i] for i in range(len(ix_nz)) if ix_nz[i]]
    curr_reps = [curr_reps[i] for i in range(len(ix_nz)) if ix_nz[i]]

    #Merge the componetens of all genes
    A = sp.sparse.vstack(curr_mats).tocsc()
    w = np.hstack(curr_weights)
    y = np.hstack(curr_ys)
    rep = np.hstack(curr_reps)

    return A, w, y, rep


 
def process_bck_gene_for_glm_mat(data):
    '''
    This function computes how much coverage is at each position in each state. 
    It returns a dictionary with the histogram with of the counts in each state.
    '''

    CurrGenePath, gene_rep_back, gene, gene_nr, rep, NrOfStates, nr_of_genes, bg_type, fg_state, bg_state = data

    #1) get the counts
    counts = {}
    if bg_type == 'Const':
        gene_length = len(CurrGenePath)
        counts[gene_rep_back/float(gene_length)] = gene_length
    elif bg_type == 'Coverage' or bg_type == 'Coverage_bck':
        for CurrState in range(NrOfStates):
            curr_counts = gene_rep_back[0, CurrGenePath == CurrState]
            curr_counts[curr_counts < 0] = 0
            counts[CurrState]= np.unique(curr_counts, return_counts=True)
    else: #'Count'   
        curr_counts = gene_rep_back[0, :]
        curr_counts[curr_counts < 0] = 0
        counts[CurrState]= np.unique(curr_counts, return_counts=True)
    
    #2) generate the matrix for this gene
    curr_y = []
    curr_data = []
    curr_rows = []
    curr_cols = []
    curr_weights = []
    curr_reps = []
    nr_of_rows = 0
    curr_pos = 0

    #Add rows for abundance and weights
    #Estimate the gene expression by a random count
    if bg_type == 'Coverage_bck':
        if np.sum(CurrGenePath == 3) == len(CurrGenePath):
            tmp_vals = gene_rep_back[0, :]
            tmp_nb_sample = np.sum(np.sum(tmp_vals > 0))
            if tmp_nb_sample > 0:
                curr_y.append(np.random.choice(tmp_vals, min(tmp_nb_sample, 10)) + 1)
                nr_of_sample_sub = min(tmp_nb_sample, 10)
            else:
                curr_y.append(np.random.choice(tmp_vals, 10) + 1)
                nr_of_sample_sub = 10
            
            curr_data.append(np.ones(nr_of_sample_sub))
            curr_rows.append(np.arange(curr_pos, curr_pos + nr_of_sample_sub))
            curr_cols.append(np.ones(nr_of_sample_sub) * gene_nr)
            curr_reps.append(-np.ones(nr_of_sample_sub) * (rep + 1))
            curr_weights.append(np.ones(nr_of_sample_sub))

            curr_pos += nr_of_sample_sub
            nr_of_rows += nr_of_sample_sub


    if bg_type == 'Coverage_bck' or bg_type == 'Coverage':
        for CurrState in range(NrOfStates):
            #Add rows for abundance and weights
            temp_y = counts[CurrState][0]
            temp_weights = counts[CurrState][1] 
            curr_weights.append(temp_weights)
            curr_y.append(temp_y)

            nr_of_obs = len(temp_weights)
            
            if bg_type == 'Coverage':
                curr_data.append(np.ones(nr_of_obs))
                curr_rows.append(np.arange(curr_pos, curr_pos + nr_of_obs))
                curr_cols.append(np.ones(nr_of_obs) * gene_nr)
                #add information for numper of replicates 
                curr_reps.append(-np.ones(nr_of_obs) * (rep + 1))
                
                if CurrState == fg_state:
                    curr_data.append(-np.ones(nr_of_obs))
                    curr_rows.append(np.arange(curr_pos, curr_pos + nr_of_obs))
                    curr_cols.append(np.ones(nr_of_obs) * (nr_of_genes ))
                
                elif CurrState == bg_state:
                    curr_data.append(np.ones(nr_of_obs))
                    curr_rows.append(np.arange(curr_pos, curr_pos + nr_of_obs))
                    curr_cols.append(np.ones(nr_of_obs) * (nr_of_genes ))
                else:
                    pass
            else: #bg_type == 'Coverage_bck':
                if CurrState < NrOfStates - 1:
                    curr_data.append(np.ones(nr_of_obs))
                    curr_rows.append(np.arange(curr_pos, curr_pos + nr_of_obs))
                    curr_cols.append(np.ones(nr_of_obs) * gene_nr)
                    #add information for numper of replicates 
                    curr_reps.append(-np.ones(nr_of_obs) * (rep + 1))
                    
                    if CurrState == fg_state:
                        curr_data.append(-np.ones(nr_of_obs))
                        curr_rows.append(np.arange(curr_pos, curr_pos + nr_of_obs))
                        curr_cols.append(np.ones(nr_of_obs) * (nr_of_genes ))
                    
                    elif CurrState == bg_state:
                        curr_data.append(np.ones(nr_of_obs))
                        curr_rows.append(np.arange(curr_pos, curr_pos + nr_of_obs))
                        curr_cols.append(np.ones(nr_of_obs) * (nr_of_genes ))
                    else:
                        pass
                else:
                    #add information for number of replicates 
                    curr_reps.append(-np.ones(nr_of_obs) * (rep + 1))
                    curr_data.append(np.ones(nr_of_obs))
                    curr_rows.append(np.arange(curr_pos, curr_pos + nr_of_obs))
                    curr_cols.append(np.ones(nr_of_obs) * (nr_of_genes + 1))

            curr_pos += nr_of_obs
            nr_of_rows += nr_of_obs

    else:
        temp_weights = np.array([counts[c] for c in list(counts.keys())])
        curr_weights.append(temp_weights)
        temp_y = np.array([c for c in list(counts.keys())])
        curr_y.append(temp_y)        
        nr_of_obs = len(temp_weights)

        #add rows for transcript abundance        
        curr_data.append(np.ones(nr_of_obs))
        curr_rows.append(np.arange(0, nr_of_obs))
        curr_cols.append(np.ones(nr_of_obs) * gene_nr)

        #add information for numper of replicates 
        curr_reps.append(-np.ones(nr_of_obs) * (rep + 1))
        nr_of_rows += nr_of_obs

    #create matrix
    if bg_type == 'Coverage_bck':
        gene_mat = coo_matrix((np.hstack(curr_data), (np.hstack(curr_rows), np.hstack(curr_cols))), shape = (nr_of_rows, nr_of_genes + 2))           
    elif bg_type == 'Coverage':
        gene_mat = coo_matrix((np.hstack(curr_data), (np.hstack(curr_rows), np.hstack(curr_cols))), shape = (nr_of_rows, nr_of_genes + 1))           
    else:
        gene_mat = coo_matrix((np.hstack(curr_data), (np.hstack(curr_rows), np.hstack(curr_cols))), shape = (nr_of_rows, nr_of_genes + NrOfStates))
        
    weights = np.hstack(curr_weights)
    y = np.hstack(curr_y)
    reps = np.hstack(curr_reps)
    new_pos = curr_pos
    return gene_mat, weights, y, reps, new_pos


 
def process_gene_for_glm_mat(data):
    '''
    This function computes how much coverage is at each position in each state. 
    It returns a dictionary with the histogram with of the counts in each state.
    '''

    CurrGenePath, rep_gene_seq, gene, gene_nr, rep, NrOfStates, nr_of_genes, bg_type, fg_state, bg_state, verbosity = data

    #1) get the counts
    counts = {}
    #CurrGenePath = Paths[gene]
    for CurrState in range(NrOfStates):
        curr_counts = rep_gene_seq[0, CurrGenePath == CurrState]
        curr_counts[curr_counts < 0] = 0
        counts[CurrState]= np.unique(curr_counts, return_counts=True)
    
    #2) generate the matrix for this gene
    curr_y = []
    curr_data = []
    curr_rows = []
    curr_cols = []
    curr_weights = []
    curr_reps = []
    nr_of_rows = 0
    curr_pos = 0
    #Check whether all positions are in background state.
    
    #Estimate the gene expression by a random count
    if bg_type == 'Coverage_bck':
        if np.sum(CurrGenePath == 3) == len(CurrGenePath):
            tmp_vals = rep_gene_seq[0, :]
            tmp_nb_sample = np.sum(np.sum(tmp_vals > 0))
            if tmp_nb_sample > 0:
                tmp_vals = tmp_vals[tmp_vals > 0]
                curr_y.append(np.random.choice(tmp_vals, min(tmp_nb_sample, 10)) + 1)
                nr_of_sample_sub = min(tmp_nb_sample, 10)
            else:
                curr_y.append(np.random.choice(tmp_vals, 10) + 1)
                nr_of_sample_sub = 10
            
            curr_data.append(np.ones(nr_of_sample_sub))
            curr_rows.append(np.arange(curr_pos, curr_pos + nr_of_sample_sub))
            curr_cols.append(np.ones(nr_of_sample_sub) * gene_nr)
            curr_reps.append(np.ones(nr_of_sample_sub) * (rep + 1))
            curr_weights.append(np.ones(nr_of_sample_sub))

            curr_pos += nr_of_sample_sub
            nr_of_rows += nr_of_sample_sub
            if verbosity:
                print('estimating gene expression on background for gene: ' + gene)
            #Assume that it is from the state with lower counts

    for CurrState in range(NrOfStates):
        #Add rows for abundance and weights
        temp_y = counts[CurrState][0]
        temp_weights = counts[CurrState][1]
        curr_weights.append(temp_weights)
        curr_y.append(temp_y)        

        nr_of_obs = len(temp_weights)
        
        if bg_type == 'None':
            #add information for numper of replicates 
            curr_reps.append(np.ones(nr_of_obs) * (rep + 1))
            #add rows for foreground and background
            curr_data.append(np.ones(nr_of_obs))
            curr_rows.append(np.arange(curr_pos, curr_pos + nr_of_obs))
            curr_cols.append(np.ones(nr_of_obs) * CurrState)
        elif bg_type == 'Coverage':
            curr_data.append(np.ones(nr_of_obs))
            curr_rows.append(np.arange(curr_pos, curr_pos + nr_of_obs))
            curr_cols.append(np.ones(nr_of_obs) * gene_nr)
            #add information for numper of replicates 
            curr_reps.append(np.ones(nr_of_obs) * (rep + 1))
            
            if CurrState == fg_state:
                curr_data.append(np.ones(nr_of_obs))
                curr_rows.append(np.arange(curr_pos, curr_pos + nr_of_obs))
                curr_cols.append(np.ones(nr_of_obs) * (nr_of_genes ))
            
            elif CurrState == bg_state:
                curr_data.append(-np.ones(nr_of_obs))
                curr_rows.append(np.arange(curr_pos, curr_pos + nr_of_obs))
                curr_cols.append(np.ones(nr_of_obs) * (nr_of_genes ))
            else:
                pass
        elif bg_type == 'Coverage_bck':
            if CurrState < NrOfStates - 1:
                curr_data.append(np.ones(nr_of_obs))
                curr_rows.append(np.arange(curr_pos, curr_pos + nr_of_obs))
                curr_cols.append(np.ones(nr_of_obs) * gene_nr)
                #add information for numper of replicates 
                curr_reps.append(np.ones(nr_of_obs) * (rep + 1))
                
                if CurrState == fg_state:
                    curr_data.append(np.ones(nr_of_obs))
                    curr_rows.append(np.arange(curr_pos, curr_pos + nr_of_obs))
                    curr_cols.append(np.ones(nr_of_obs) * (nr_of_genes ))
                
                elif CurrState == bg_state:
                    curr_data.append(-np.ones(nr_of_obs))
                    curr_rows.append(np.arange(curr_pos, curr_pos + nr_of_obs))
                    curr_cols.append(np.ones(nr_of_obs) * (nr_of_genes ))
                else:
                    pass
            else:
                #add information for numper of replicates 
                curr_reps.append(np.ones(nr_of_obs) * (rep + 1))
                curr_data.append(np.ones(nr_of_obs))
                curr_rows.append(np.arange(curr_pos, curr_pos + nr_of_obs))
                curr_cols.append(np.ones(nr_of_obs) * (nr_of_genes + 1))
        else:
            #add rows for transcript abundance
            curr_data.append(np.ones(nr_of_obs))
            curr_rows.append(np.arange(curr_pos, curr_pos + nr_of_obs))
            curr_cols.append(np.ones(nr_of_obs) * gene_nr)
            #add information for numper of replicates 
            curr_reps.append(np.ones(nr_of_obs) * (rep + 1))
            #add rows for foreground and background
            if not( (bg_type == 'Coverage_bck') and (CurrState == (NrOfStates - 1))):
                curr_data.append(np.ones(nr_of_obs))
                curr_rows.append(np.arange(curr_pos, curr_pos + nr_of_obs))
                curr_cols.append(np.ones(nr_of_obs) * (nr_of_genes + CurrState))
            else:#TODO: Code duplication
                curr_data.append(np.ones(nr_of_obs))
                curr_rows.append(np.arange(curr_pos, curr_pos + nr_of_obs))
                curr_cols.append(np.ones(nr_of_obs) * (nr_of_genes + CurrState))

        curr_pos += nr_of_obs
        nr_of_rows += nr_of_obs
  
    #create matrix
    if bg_type == 'None':
        gene_mat = coo_matrix((np.hstack(curr_data), (np.hstack(curr_rows), np.hstack(curr_cols))), shape = (nr_of_rows, NrOfStates))
    elif bg_type == 'Coverage_bck':
        gene_mat = coo_matrix((np.hstack(curr_data), (np.hstack(curr_rows), np.hstack(curr_cols))), shape = (nr_of_rows, nr_of_genes + 2))           
    elif bg_type == 'Coverage':
        gene_mat = coo_matrix((np.hstack(curr_data), (np.hstack(curr_rows), np.hstack(curr_cols))), shape = (nr_of_rows, nr_of_genes + 1))
    else: # 'Const' or 'Count'
        gene_mat = coo_matrix((np.hstack(curr_data), (np.hstack(curr_rows), np.hstack(curr_cols))), shape = (nr_of_rows, nr_of_genes + NrOfStates))

    weights = np.hstack(curr_weights)
    y = np.hstack(curr_y)
    reps = np.hstack(curr_reps)
    new_pos = curr_pos

    return gene_mat, weights, y, reps, new_pos


 
def process_pseudo_gene_for_glm_mat(Sequences, Background, rep, NrOfStates, curr_pos):
    '''
    This function computes a pseduo row entry for the matrix in order to have full rank
    '''

    curr_data = []
    curr_rows = []
    curr_cols = []
    nr_of_rows = 0
    nr_of_genes = len(list(Sequences.keys()))

    curr_data = np.ones(4)
    curr_rows = np.arange(curr_pos, (curr_pos + 4))
    curr_cols = np.hstack((np.zeros(1), np.arange(nr_of_genes, (nr_of_genes + 3))))

    curr_pos += 4
    nr_of_rows = 4

    #create matrix
    gene_mat = csc_matrix((curr_data, (curr_rows, curr_cols)), shape = (nr_of_rows, nr_of_genes + NrOfStates))
    
    weights = np.array([0.1,0.1,0.1, 0.1])*10.0
    y = np.array([0,0,0,0])
    reps = np.hstack([0,0,0,0])
    new_pos = curr_pos
    print ('warning:Using peseudocount during glm contruction')

    return gene_mat, weights, y, reps, new_pos


 
def fit_glm(A, w, Y, offset, sample_size, disp = None, start_params = None, norm_class=False, verbosity=1):
    '''
    This function fits the GLM
    '''
    if disp == None:
        disp = 1.0
    for i in range(3):
        #Fit GLM
        Y = Y.reshape((len(Y), 1)) 
        offset = offset.reshape((len(offset), 1))
        w = w.reshape((len(w), 1))
        if disp != None:
            disp = min(disp, 100)
        if isinstance(start_params, np.ndarray):
            first = 0
        else:
            first = 1

        tempw = np.float64(w)
        if norm_class:
            Ix_expr =(A[:,-1].toarray() == 1)
            tempw[Ix_expr] = tempw[Ix_expr] / np.sum(tempw[Ix_expr])

            Ix_expr =(A[:,-1].toarray() == 0) * (A[:,-2].toarray() == 0) 
            tempw[Ix_expr] = tempw[Ix_expr] / np.sum(tempw[Ix_expr])

            Ix_expr =(A[:,-1].toarray() == 0) * (A[:,-2].toarray() > 0)
            tempw[Ix_expr] = tempw[Ix_expr] / np.sum(tempw[Ix_expr])

            Ix_expr =(A[:,-1].toarray() == 0) * (A[:,-2].toarray() < 0)
            tempw[Ix_expr] = tempw[Ix_expr] / np.sum(tempw[Ix_expr])

        glm_binom = sparse_glm.sparse_glm(Y, A, offset = np.log(offset), family = sparse_glm.families.NegativeBinomial(alpha = disp))
        res = glm_binom.fit(method = "irls_sparse", data_weights = tempw, start_params = start_params)
        start_params = res[0] 
        mu = res[1]['mu'] 
        del res, glm_binom
        old_disp = disp
        
        disps = []
        if verbosity > 1:
            print(start_params[-3:])

        disp = sp.optimize.minimize_scalar(neg_NB_GLM_loglike, bracket=None, bounds=[0.1, 100], args=(Y, mu, tempw,), method='bounded', options={'xatol': 0.00001})['x'] #*10

        if verbosity > 1:
            print('Dispersion ' + str(disp))
        glm_binom = sparse_glm.sparse_glm(Y, A, offset = np.log(offset), family = sparse_glm.families.NegativeBinomial(alpha = disp))
        res = glm_binom.fit(method = "irls_sparse", data_weights = tempw, start_params = start_params)

        start_params = res[0]

        if verbosity > 1:
            print((np.sum(np.abs(mu - res[1]['mu']))))
        del glm_binom, res

    # 3) Process the output
    del A, w, Y, offset

    return start_params, disp


 
def neg_NB_GLM_loglike(alpha, endog, mu, weights):
    """
    This function computes the loglikelihood for a fitted NB GLM
    """

    lin_pred = sparse_glm.families.NegativeBinomial(alpha = alpha)._link(mu)
    constant = (special.gammaln(endog + 1 / alpha) -
                special.gammaln(endog+1)-special.gammaln(1/alpha))
    exp_lin_pred = np.exp(lin_pred)

    return -np.sum((endog * np.log(alpha * exp_lin_pred / (1 + alpha * exp_lin_pred)) - np.log(1 + alpha * exp_lin_pred) / alpha + constant) * weights)


 
def get_expected_mean_and_var(CurrStackSum, State, nr_of_genes, gene_nr, EmissionParameters, curr_type = 'fg', verbosity=1):
    '''
    This function computes the expected means and variances for each of the states
    '''

    fg_state, bg_state = get_fg_and_bck_state(EmissionParameters)
    start_params = EmissionParameters['ExpressionParameters'][0]
    disp = EmissionParameters['ExpressionParameters'][1]
    nr_of_states = EmissionParameters['NrOfStates']
    #Create the log mean matrix for each rep

    bg_type = EmissionParameters['BckType'] 
    if curr_type == 'fg':
        lib_size = EmissionParameters['LibrarySize']
        nr_of_rep = EmissionParameters['NrOfReplicates']
    else:
        lib_size = EmissionParameters['BckLibrarySize']
        nr_of_rep = EmissionParameters['NrOfBckReplicates']
    
    gene_len = CurrStackSum.shape[1]

    GLM_mat = np.zeros((nr_of_rep, 1))
    
    if bg_type == 'Coverage_bck':
        start_params = (start_params)
        if start_params[gene_nr, 0] - abs(start_params[nr_of_genes, 0]) < start_params[nr_of_genes + 1, 0]:
            start_params[gene_nr, 0] = abs(start_params[nr_of_genes, 0]) + start_params[nr_of_genes + 1, 0] + 0.01
            if EmissionParameters['Verbosity']:
                print('Adjusted start params')

    #check whether the current data is foreground data
    for rep in range(nr_of_rep):
        GLM_mat[rep, :] += np.ones((1)) * np.log(lib_size[str(rep)])
        if curr_type == 'fg':
            if bg_type == 'None':
                GLM_mat[rep, :] += np.ones((1)) * start_params[State, 0]
            elif bg_type == 'Coverage_bck':
                if State < nr_of_states - 1:
                    GLM_mat[rep, :] += np.ones((1)) * start_params[gene_nr, 0]
                    if State == fg_state:#Skip the last state
                        GLM_mat[rep, :] += np.ones((1)) * start_params[nr_of_genes, 0] 
                    elif State == bg_state:#TODO: Code duplication
                        GLM_mat[rep, :] -= np.ones((1)) * start_params[nr_of_genes, 0]
                    else:
                        continue
                else:
                    GLM_mat[rep, :] += np.ones((1)) * start_params[nr_of_genes + 1, 0]

            elif bg_type == 'Coverage':
                GLM_mat[rep, :] += np.ones((1)) * start_params[gene_nr, 0]
                if State == fg_state:#Skip the last state
                    GLM_mat[rep, :] += np.ones((1)) * start_params[nr_of_genes, 0] 
                elif State == bg_state:#TODO: Code duplication
                    GLM_mat[rep, :] -= np.ones((1)) * start_params[nr_of_genes, 0]
                else:
                    continue
            else:
                GLM_mat[rep, :] += np.ones((1)) * start_params[gene_nr, 0]
                GLM_mat[rep, :] += np.ones((1)) * start_params[nr_of_genes + State, 0] 
        else:
            if bg_type == 'None':
                GLM_mat[rep, :] += np.ones((1)) * start_params[State, 0]
            elif bg_type == 'Coverage_bck':
                if State < nr_of_states - 1:
                    GLM_mat[rep, :] += np.ones((1)) * start_params[gene_nr, 0]
                    if State == fg_state:#Skip the last state
                        GLM_mat[rep, :] -= np.ones((1)) * start_params[nr_of_genes, 0] 
                    elif State == bg_state:#TODO: Code duplication
                        GLM_mat[rep, :] += np.ones((1)) * start_params[nr_of_genes, 0]
                    else:
                        continue
                else:
                    GLM_mat[rep, :] += np.ones((1)) * start_params[nr_of_genes + 1, 0]
            elif bg_type == 'Coverage':
                GLM_mat[rep, :] += np.ones((1)) * start_params[gene_nr, 0]
                if State == fg_state:#Skip the last state
                    GLM_mat[rep, :] -= np.ones((1)) * start_params[nr_of_genes, 0] 
                elif State == bg_state:#TODO: Code duplication
                    GLM_mat[rep, :] += np.ones((1)) * start_params[nr_of_genes, 0]
                else:
                    continue
            else:
                GLM_mat[rep, :] += np.ones((1)) * start_params[gene_nr, 0]

    #Compute the mean of the positions in each state
    mean_mat = np.tile(np.exp(GLM_mat), (1, gene_len))

    #Compute the variance of the position
    var_mat = mean_mat + disp * (mean_mat ** 2)

    #Adjust variance for the background state
    if State == 3:
        if bg_type == 'Coverage_bck':
            GLM_mat_alt = np.zeros((nr_of_rep, 1))
            for rep in range(nr_of_rep):
                GLM_mat_alt[rep, :] += np.ones((1)) * np.log(lib_size[str(rep)])
            GLM_mat_alt +=  start_params[gene_nr, 0] - abs(start_params[nr_of_genes, 0])
            mean_mat_alt = np.tile(np.exp(GLM_mat_alt), (1, gene_len))
            disp_alt = mean_mat_alt * disp / mean_mat
            var_mat_alt = mean_mat * disp_alt * (mean_mat ** 2)
            ix = var_mat_alt > var_mat
            var_mat[ix] = var_mat_alt[ix]
        
    # make sure that var >mean
    ix = (mean_mat * 1.000001) > var_mat
    var_mat[ix] =  mean_mat[ix] * 1.000001

    return mean_mat, var_mat


 
def predict_expression_log_likelihood_for_gene(CurrStackSum, State, nr_of_genes, gene_nr, EmissionParameters, curr_type = 'fg'):
    '''
    This function predicts the likelihood of expression for a gene
    '''

    nr_of_rep = EmissionParameters['NrOfReplicates']
    mean_mat, var_mat = get_expected_mean_and_var(CurrStackSum, State, nr_of_genes, gene_nr, EmissionParameters, curr_type = curr_type)

    #reset the  variance if the empirical variance is larger than the estimated variance
    if EmissionParameters['emp_var'] and (mean_mat.shape[0] > 1):
        mu = np.zeros_like(mean_mat)
        emp_var = np.zeros_like(mean_mat)
        for rep in range(nr_of_rep):
            for i in range(nr_of_rep):
                mu[i, :] = CurrStackSum[i, :] / mean_mat[i, :] * mean_mat[rep, :]
            emp_var[rep, :] = np.var(mu, axis = 0)
            var_mat[rep, :] = np.max(np.vstack([var_mat[rep, :], emp_var[rep, :]]), axis = 0)

    #Get the parameters for the NB distributions
    p, n = NB_parameter_estimation(mean_mat, var_mat)

    #Compute the likelihood
    ix_nonzero = np.sum(CurrStackSum, axis=0) > 0
    zero_array = nbinom._logpmf(np.zeros((CurrStackSum.shape[0], 1)), np.expand_dims(n[:,0], axis=1), np.expand_dims(p[:,0], axis=1))

    loglike = np.tile(zero_array, (1, CurrStackSum.shape[1]))
    loglike[:, ix_nonzero] = nbinom._logpmf(CurrStackSum[:, ix_nonzero], n[:, ix_nonzero], p[:, ix_nonzero])

    #combine the replicates 
    loglike = loglike.sum(axis = 0)

    return loglike


 
def get_fg_and_bck_state(EmissionParameters, final_pred=False):
    '''
    This function determines the fg and background state
    '''

    #Ckeck whether the parameters have been already defined
    if final_pred:
        if EmissionParameters['ExpressionParameters'][0] is None:
            fg_state = 0
            bg_state = 1
        else:
            bg_type = EmissionParameters['BckType']
            if bg_type == 'Coverage_bck':
                if EmissionParameters['ExpressionParameters'][0][-2] > 0:
                    fg_state = 1
                    bg_state = 0
                else:
                    fg_state = 0
                    bg_state = 1
            else:
                if EmissionParameters['ExpressionParameters'][0][-1] > 0:
                    fg_state = 1
                    bg_state = 0
                else:
                    fg_state = 0
                    bg_state = 1
    else:
        fg_state = 1
        bg_state = 0

    return fg_state, bg_state
