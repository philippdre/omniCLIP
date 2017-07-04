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

import numpy as np
import os
import sys
sys.path.append('./data_parsing/')
sys.path.append('./stat/')
sys.path.append('./visualisation/')
import gffutils
from scipy.sparse import *
import mixture_tools
import time
import LoadReads
import emission
import trans
import argparse
import tools
import cPickle
import resource
import gc
from collections import defaultdict
from intervaltree import Interval, IntervalTree

def run_sclip(args):
    # Get the args
    args = parser.parse_args()
    print args

    #Check parameters
    if len(args.fg_libs) == 0:
        raise sys.exit('No CLIP-libraries given')

    if len(args.bg_libs) == 0:
        bg_type = 'None'
    else:
        bg_type = args.bg_type


    if args.out_dir == None:
        out_path = os.getcwdu()
    else:
        out_path = args.out_dir

    MaxIter  = args.max_it
    # process the parameters

    if not (bg_type == 'Coverage' or  bg_type == 'Coverage_bck'):
        print 'Bg-type: ' + bg_type + ' has not been implemented yet'
        return 

    #Load the gene annotation
    print 'Loading gene annotation'
    GeneAnnotation = gffutils.FeatureDB(args.gene_anno_file, keep_order=True)
    GenomeDir = args.genome_dir
    #Load the reads
    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print 'Loading reads'

    EmissionParameters = {}
    EmissionParameters['glm_weight'] = args.glm_weight
    #Check whether existing iteration parameters should be used
    restart_from_file = args.restart_from_file
    EmissionParameters['restart_from_file'] = restart_from_file

    EmissionParameters['mask_flank_variants'] = args.mask_flank_variants

    EmissionParameters['max_mm'] = args.max_mm
    t = time.time()
    DataOutFile = os.path.join(out_path, 'fg_reads.dat')
    Sequences = LoadReads.load_data(args.fg_libs, GenomeDir, GeneAnnotation, DataOutFile, load_from_file = ((not args.overwrite_fg) or restart_from_file), save_results = True, Collapse = args.fg_collapsed, mask_flank_variants=EmissionParameters['mask_flank_variants'], max_mm=EmissionParameters['max_mm'])
    
    DataOutFile = os.path.join(out_path, 'bg_reads.dat')
    Background = LoadReads.load_data(args.bg_libs, GenomeDir, GeneAnnotation, DataOutFile, load_from_file = ((not args.overwrite_bg) or restart_from_file), save_results = True, Collapse = args.bg_collapsed, OnlyCoverage = args.only_coverage,  mask_flank_variants=EmissionParameters['mask_flank_variants'], max_mm=EmissionParameters['max_mm'])
    #pdb.set_trace()
    #Mask the positions that overlap miRNA sites in the geneome
    EmissionParameters['mask_miRNA'] = args.mask_miRNA
    if args.mask_miRNA:
        print 'Removing miRNA-coverage'
        Sequences = mask_miRNA_positions(Sequences, GeneAnnotation)

    EmissionParameters['mask_ovrlp'] = args.mask_ovrlp

    if EmissionParameters['mask_ovrlp']:
        print 'Masking overlapping positions'
        Sequences = mark_overlapping_positions(Sequences, GeneAnnotation)

    
    EmissionParameters['BckLibrarySize'] =  tools.estimate_library_size(Background)
    EmissionParameters['LibrarySize'] = tools.estimate_library_size(Sequences)
    
    #Removing genes without any reads in the CLIP data
    print "Removing genes without CLIP coverage"

    genes_to_keep = []
    all_genes = Sequences.keys()
    for i, gene in enumerate(Sequences.keys()):
        curr_cov = np.sum(np.array([Sequences[gene]['Coverage'][rep].sum() for rep in Sequences[gene]['Coverage'].keys()]))
        curr_neg_vars = np.sum(np.array([np.sum(np.sum(Sequences[gene]['Variants'][rep].toarray() < 0 )) for rep in Sequences[gene]['Variants'].keys()]))

        if curr_cov <= 100 or curr_neg_vars > 0:
            continue

        genes_to_keep.append(gene)
        if i > args.gene_sample:
            break
    
    genes_to_del = list(set(all_genes).difference(set(genes_to_keep)))

    for gene in genes_to_del:
        del Sequences[gene]
        del Background[gene]

    del all_genes, genes_to_del, genes_to_keep 
    print 'Done: Elapsed time: ' + str(time.time() - t)
    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    #Initializing parameters
    print 'Initialising the parameters'
    if bg_type == 'Coverage_bck':
        NrOfStates = 4
    else:
        NrOfStates = 3

    #Remove the gene sequence from the Sequences and Background when not needed. Currently this is always the case:
    for gene in Sequences.keys():
        if Sequences[gene].has_key('GeneSeq'):
            del Sequences[gene]['GeneSeq']

    for gene in Background.keys():
        if Background[gene].has_key('GeneSeq'):
            del Background[gene]['GeneSeq']


    TransMat = np.ones((NrOfStates, NrOfStates)) + np.eye(NrOfStates)
    TransMat = TransMat / np.sum(np.sum(TransMat))
    TransitionParameters = [TransMat, []]



    NrOfReplicates = len(args.fg_libs)
    gene = Sequences.keys()[0]
    
    EmissionParameters['PriorMatrix'] = np.ones((NrOfStates, 1)) / float(NrOfStates)
    EmissionParameters['diag_bg'] = args.diag_bg
    EmissionParameters['emp_var'] = args.emp_var
    EmissionParameters['norm_class'] = args.norm_class
    EmissionParameters['Diag_event_params'] = {}
    EmissionParameters['Diag_event_params']['nr_mix_comp'] = args.nr_mix_comp
    EmissionParameters['Diag_event_params']['mix_comp'] = {}
    for state in range(NrOfStates):
        mixtures = np.random.uniform(0.0, 1.0, size=(args.nr_mix_comp))
        EmissionParameters['Diag_event_params']['mix_comp'][state] = mixtures / np.sum(mixtures)
    
    #initialise the parameter vector alpha
    alphashape = (Sequences[gene]['Variants'][0].shape[0] + Sequences[gene]['Coverage'][0].shape[0] + Sequences[gene]['Read-ends'][0].shape[0])
    alpha = {}
    for state in range(NrOfStates):        
            alpha[state] = np.random.uniform(0.9, 1.1, size=(alphashape, args.nr_mix_comp))

    EmissionParameters['Diag_event_params']['alpha'] = alpha
    EmissionParameters['Diag_event_type'] = args.diag_event_mod
    EmissionParameters['NrOfStates'] = NrOfStates
    EmissionParameters['NrOfReplicates'] = NrOfReplicates
    EmissionParameters['ExpressionParameters'] = [None, None]
    EmissionParameters['BckType'] = bg_type
    EmissionParameters['NrOfBckReplicates'] = len(args.bg_libs)
    EmissionParameters['TransitionType'] = args.tr_type
    EmissionParameters['Verbosity'] = args.verbosity
    EmissionParameters['NbProc'] = args.nb_proc
    EmissionParameters['Subsample'] = args.subs

    EmissionParameters['FilterSNPs'] = args.filter_snps
    EmissionParameters['SnpRatio'] = args.snps_thresh
    EmissionParameters['SnpAbs'] = args.snps_min_cov
    EmissionParameters['ign_diag'] = args.ign_diag
    EmissionParameters['ign_GLM'] = args.ign_GLM
    EmissionParameters['only_pred'] = args.only_pred
    
    # Transistion parameters
    IterParameters = [EmissionParameters, TransitionParameters]


    #Start computation

    #Iterativly fit the parameters of the model
    OldLogLikelihood = 0
    CurrLogLikelihood = -np.inf
    CurrIter = 0
    LoglikelihodList = []
    First = 1
    IterSaveFile = os.path.join(out_path, 'IterSaveFile.dat')
    IterSaveFileHist = os.path.join(out_path, 'IterSaveFileHist.dat')
    IterHist = []
    Paths = {}
    iter_cond = True
    #Check whether to preload the iteration file
    if EmissionParameters['only_pred']:
        IterParameters, args_old = cPickle.load(open(IterSaveFile,'r'))
        EmissionParameters['mask_miRNA'] = args.mask_miRNA
        EmissionParameters['glm_weight'] = args.glm_weight
        EmissionParameters['restart_from_file'] = restart_from_file
        EmissionParameters =  IterParameters[0]
        EmissionParameters['ign_diag'] = args.ign_diag
        EmissionParameters['ign_GLM'] = args.ign_GLM
        TransitionParameters = IterParameters[1]
        TransitionType = EmissionParameters['TransitionType']
        OldLogLikelihood = -np.inf
        fg_state, bg_state = emission.get_fg_and_bck_state(EmissionParameters, final_pred=True)

        Paths, CurrLogLikelihood = tools.ParallelGetMostLikelyPath(Paths, Sequences, Background, EmissionParameters, TransitionParameters, 'nonhomo')
        First = 0
        iter_cond = False


    if restart_from_file:
        IterParameters, args_old = cPickle.load(open(IterSaveFile,'r'))
        EmissionParameters =  IterParameters[0]
        EmissionParameters['mask_miRNA'] = args.mask_miRNA

        EmissionParameters['glm_weight'] = args.glm_weight
        EmissionParameters['restart_from_file'] = restart_from_file
        EmissionParameters['ign_diag'] = args.ign_diag
        EmissionParameters['ign_GLM'] = args.ign_GLM
        TransitionParameters = IterParameters[1]
        TransitionType = EmissionParameters['TransitionType']
        OldLogLikelihood = -np.inf
        Paths, CurrLogLikelihood = tools.ParallelGetMostLikelyPath(Paths, Sequences, Background, EmissionParameters, TransitionParameters, 'nonhomo')
        First = 1
        iter_cond = True

    while iter_cond:
        print "Iteration: " + str(CurrIter)
        if EmissionParameters['Verbosity'] > 0:
            print IterParameters[0]

        OldLogLikelihood  = CurrLogLikelihood
        
        CurrLogLikelihood, IterParameters, First, Paths = PerformIteration(Sequences, Background, IterParameters, NrOfStates, First, Paths)
        gc.collect()
        
        if args.safe_tmp: 
            cPickle.dump([IterParameters, args], open(IterSaveFile,'w'))
        if args.safe_tmp:
            if CurrIter > 0:
                IterHist = cPickle.load(open(IterSaveFileHist,'r'))
            IterHist.append([IterParameters, CurrLogLikelihood])
            cPickle.dump(IterHist, open(IterSaveFileHist,'w'))
            del IterHist
        
        print "Log-likelihood: " + str(CurrLogLikelihood) 
        LoglikelihodList.append(CurrLogLikelihood)
        
        print LoglikelihodList
        CurrIter += 1
        
        if CurrIter >= MaxIter:
            print 'Maximal number of iterations reached'

        if not restart_from_file:
            if CurrIter < max(3, MaxIter):
                iter_cond = True
            else:
                iter_cond = (CurrIter < MaxIter) and ((abs(CurrLogLikelihood - OldLogLikelihood)/max(abs(CurrLogLikelihood), abs(OldLogLikelihood))) > 0.01) and (abs(CurrLogLikelihood - OldLogLikelihood) > args.tol_lg_lik)

        else:
            if np.isinf(OldLogLikelihood):
                iter_cond = (CurrIter < MaxIter) and (abs(CurrLogLikelihood - OldLogLikelihood) > args.tol_lg_lik)    
            else:
                iter_cond = (CurrIter < MaxIter) and ((abs(CurrLogLikelihood - OldLogLikelihood)/max(abs(CurrLogLikelihood), abs(OldLogLikelihood))) > 0.01) and (abs(CurrLogLikelihood - OldLogLikelihood) > args.tol_lg_lik)
    
    #Return the fitted parameters
    print 'Finished fitting of parameters'


    EmissionParameters, TransitionParameters = IterParameters
    if not isinstance(EmissionParameters['ExpressionParameters'][0], np.ndarray):
        print 'Emmision parameters have not been fit yet'
        return
    out_file_base = 'pred'
    if EmissionParameters['ign_GLM']:
       out_file_base += '_no_glm'
    if EmissionParameters['ign_diag']:
       out_file_base += '_no_diag'
    OutFile = os.path.join(out_path, out_file_base + '.txt')
    #determine which state has higher weight in fg.
    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    fg_state, bg_state = emission.get_fg_and_bck_state(EmissionParameters, final_pred=True)
    
    tools.GeneratePred(Paths, Sequences, Background, IterParameters, GeneAnnotation, OutFile, fg_state, bg_state)

    print 'Done'

    return

def mask_miRNA_positions(Sequences, GeneAnnotation):
    '''
    This function takes the sequences and 
    the gene annotation and sets all counts in Sequences to zero that overlap
    miRNAs in the gene annotaion
    '''
    keys = ['Coverage', 'Read-ends', 'Variants']
    #Create a dictionary that stores the genes in the Gene annnotation
    gene_dict = {}

    for gene in  GeneAnnotation.features_of_type('gene'):
        gene_dict[gene.id.split('.')[0]] = gene

    #Get Chromosomes:
    genes_chr_dict = defaultdict(list)
    for gene in gene_dict.values():
        genes_chr_dict[gene.chrom].append(gene)

    #Create an interval tree for the genes:
    interval_chr_dict = {}
    for chr in genes_chr_dict.keys():
        interval_chr_dict[chr] = IntervalTree()
        for gene in genes_chr_dict[chr]:
            interval_chr_dict[chr][gene.start : gene.stop] = gene
    
    miRNAs = [miRNA for miRNA in GeneAnnotation.features_of_type('gene') if miRNA.attributes.get('gene_type').count('miRNA') > 0]
    #Iterate over the genes in the Sequences:
    
    for miRNA in miRNAs:
        curr_chr = miRNA.chrom
        curr_genes = sorted(interval_chr_dict[curr_chr][miRNA.start : miRNA.stop])
        curr_genes = [gene[2] for gene in curr_genes]
        #Get the miRNAs that overalp:
        for curr_gene_obj in curr_genes:
            curr_gene = curr_gene_obj.id.split('.')[0]

            #Get position relative to the host gene 
            curr_start = max(0, miRNA.start - gene_dict[curr_gene].start)
            curr_stop = max(gene_dict[curr_gene].stop - gene_dict[curr_gene].start, miRNA.stop - gene_dict[curr_gene].start)

            #Set for each field the sequences to zeros
            for curr_key in keys:
                if Sequences[curr_gene].has_key(curr_key):
                    for rep in Sequences[curr_gene][curr_key].keys():
                        curr_seq = Sequences[curr_gene][curr_key][rep][:,].tolil()
                        curr_seq[:, curr_start: curr_stop] = 0
                        Sequences[curr_gene][curr_key][rep] = curr_seq.tocsr()

    return Sequences


def mark_overlapping_positions(Sequences, GeneAnnotation):
    '''
    This function takes the sequences and 
    the gene annotation and adds to Sequences a track that indicates the overlaping regions
    '''

    #add fields to Sequence structure:
    for gene in Sequences.keys():
        Sequences[gene]['mask'] = {}
        rep = Sequences[gene]['Coverage'].keys()[0]
        Sequences[gene]['mask'][rep] = np.zeros(Sequences[gene]['Coverage'][rep].shape, dtype=bool)

    #Create a dictionary that stores the genes in the Gene annnotation
    gene_dict = {}

    for gene in  GeneAnnotation.features_of_type('gene'):
        gene_dict[gene.id.split('.')[0]] = gene

    #Get Chromosomes:
    genes_chr_dict = defaultdict(list)
    for gene in gene_dict.values():
        genes_chr_dict[gene.chrom].append(gene)

    #Create an interval tree for the genes:
    interval_chr_dict = {}
    for chr in genes_chr_dict.keys():
        interval_chr_dict[chr] = IntervalTree()
        for gene in genes_chr_dict[chr]:
            interval_chr_dict[chr][gene.start : gene.stop] = gene
    
    genes = [gene for gene in GeneAnnotation.features_of_type('gene')]
    #Iterate over the genes in the Sequences:
    for gene in genes:
        if not Sequences.has_key(gene.id.split('.')[0]):
            continue
        curr_chr = gene.chrom
        curr_genes = sorted(interval_chr_dict[curr_chr][gene.start : gene.stop])
        curr_genes = [curr_gene[2] for curr_gene in curr_genes]
        curr_genes.remove(gene)
        #Get the genes that overalp:
        for curr_gene_obj in curr_genes:
            curr_gene = curr_gene_obj.id.split('.')[0]

            #Get position of overlapping gene relative to the host gene 
            ovrlp_start = max(0, gene_dict[curr_gene].start - gene.start)
            ovrlp_stop = min(gene.stop - gene.start, gene_dict[curr_gene].stop - gene.start)

            #Set for each field the sequences to zeros
            rep = Sequences[gene.id.split('.')[0]]['Coverage'].keys()[0]
            Sequences[gene.id.split('.')[0]]['mask'][rep][0, ovrlp_start : ovrlp_stop] = True

    return Sequences



def pred_sites(args):
    # Get the args

    args = parser.parse_args()
    print args

    #Check parameters
    if len(args.fg_libs) == 0:
        raise sys.exit('No CLIP-libraries given')

    if len(args.bg_libs) == 0:
        bg_type = 'None'
    else:
        bg_type = args.bg_type


    if args.out_dir == None:
        out_path = os.getcwdu()
    else:
        out_path = args.out_dir

    MaxIter  = args.max_it
    # process the parameters

    if not (bg_type == 'Coverage' or  bg_type == 'Coverage_bck'):
        print 'Bg-type: ' + bg_type + ' has not been implemented yet'
        return 



    #Load the gene annotation
    print 'Loading gene annotation'
    GeneAnnotation = gffutils.FeatureDB(args.gene_anno_file, keep_order=True)
    GenomeDir = args.genome_dir
    #Load the reads
    t = time.time()
    print 'Loading reads'
    DataOutFile = os.path.join(out_path, 'fg_reads.dat')
    Sequences = LoadReads.load_data(args.fg_libs, GenomeDir, GeneAnnotation, DataOutFile, load_from_file = True, save_results = False, Collapse = args.fg_collapsed)
    
    DataOutFile = os.path.join(out_path, 'bg_reads.dat')
    Background = LoadReads.load_data(args.bg_libs, GenomeDir, GeneAnnotation, DataOutFile, load_from_file = True, save_results = False, Collapse = args.bg_collapsed, OnlyCoverage = True)

    
    #Removing genes without any reads in the CLIP data
    genes_to_keep = []
    all_genes = Sequences.keys()
    for i, gene in enumerate(Sequences.keys()):
        curr_cov = np.sum(np.array([np.sum(Sequences[gene]['Coverage'][rep].toarray()) for rep in Sequences[gene]['Coverage'].keys()]))
        curr_neg_vars = np.sum(np.array([np.sum(np.sum(Sequences[gene]['Variants'][rep].toarray() < 0 )) for rep in Sequences[gene]['Variants'].keys()]))

        if curr_cov < 100 or curr_neg_vars > 0:
            continue

        genes_to_keep.append(gene)
        if i > args.gene_sample:
            break
    
    genes_to_del = list(set(all_genes).difference(set(genes_to_keep)))

    for gene in genes_to_del:
        del Sequences[gene]
        del Background[gene]

    del all_genes, genes_to_del, genes_to_keep 
    print 'Done: Elapsed time: ' + str(time.time() - t)
    
    #Load data
    tmp_file = cPickle.load(open(os.path.join(out_path, 'IterSaveFile.dat'), 'r'))
    IterParameters = tmp_file[0]
    args = tmp_file[1]
    EmissionParameters = IterParameters[0]
    fg_state, bg_state = emission.get_fg_and_bck_state(EmissionParameters, final_pred=True)
    tools.GeneratePred(Sequences, Background, IterParameters, GeneAnnotation, OutFile, fg_state, bg_state)

    print 'Done'



def PerformIteration(Sequences, Background, IterParameters, NrOfStates, First, NewPaths={}):
    '''
    This function performs an iteration of the HMM algorithm 
    '''
    #unpack the Iteration parameters
    EmissionParameters = IterParameters[0]
    TransitionParameters = IterParameters[1]
    TransitionType = EmissionParameters['TransitionType']
    #Get new most likely path
    if (not EmissionParameters['restart_from_file']) and First:
        NewPaths, LogLike = tools.ParallelGetMostLikelyPath(NewPaths, Sequences, Background, EmissionParameters, TransitionParameters, 'homo', RandomNoise = True)
        #NewPaths, LogLike = tools.GetMostLikelyPath(Sequences, Background, EmissionParameters, TransitionParameters, 'homo', RandomNoise = True)
        #pdb.set_trace()
        
        print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    #Perform EM to compute the new emission parameters
    print 'Fitting emission parameters'
    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    NewEmissionParameters = FitEmissionParameters(Sequences, Background, NewPaths, EmissionParameters, First)
    if First:
        First = 0
    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    #Fit the transition matrix parameters
    #print 'Fitting transition parameters'
    NewTransitionParameters = TransitionParameters
    C = 1
    print 'Fitting transistion parameters'
    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    
    TransistionPredictors = trans.FitTransistionParameters(Sequences, Background, TransitionParameters, NewPaths, C, TransitionType)
    NewTransitionParameters[1] = TransistionPredictors
    #NewTransitionParameters = FitTransistionParameters(Sequences, Background, NewPaths)
    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    NewIterParameters = [NewEmissionParameters, NewTransitionParameters]

    #Compute the log likelihood of the model
    
    print 'Computing most likely path'
    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    
    gc.collect()
    NewPaths, LogLike = tools.ParallelGetMostLikelyPath(NewPaths, Sequences, Background, EmissionParameters, TransitionParameters, 'nonhomo')
    CurrLogLikelihood = LogLike
    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print 'LogLik:'
    print CurrLogLikelihood
    return CurrLogLikelihood, NewIterParameters, First, NewPaths


def FitEmissionParameters(Sequences, Background, NewPaths, OldEmissionParameters, First):
    print 'Fitting emission parameters'
    t = time.time() 
    #Unpack the arguments
    OldAlpha = OldEmissionParameters['Diag_event_params']
    NrOfStates = OldEmissionParameters['NrOfStates']
    OldPriorMatrix = OldEmissionParameters['PriorMatrix']
    NewEmissionParameters = OldEmissionParameters

    #Compute new prior matrix    
    PriorMatrix = np.zeros_like(OldPriorMatrix)
    for State in range(NrOfStates):        
        for path in NewPaths:
            PriorMatrix[State] += np.sum(NewPaths[path] == State)
        
    CorrectedPriorMatrix = np.copy(PriorMatrix)
    
    CorrectedPriorMatrix[CorrectedPriorMatrix == 0] = np.min(CorrectedPriorMatrix[CorrectedPriorMatrix > 0])/10 
    CorrectedPriorMatrix /= np.sum(CorrectedPriorMatrix)
    #Keep a copy to check which states are not used
    NewEmissionParameters['PriorMatrix'] = CorrectedPriorMatrix

    #Add Pseudo gene to Sequences, Background and Paths
    if NewEmissionParameters['ExpressionParameters'][0] != None:
        Sequences, Background, NewPaths, pseudo_gene_names = add_pseudo_gene(Sequences, Background, NewPaths, PriorMatrix)

    #Compute parameters for the expression
    sample_size = 10000

    if NewEmissionParameters['BckType'] != 'None':
        if Sequences.has_key('Pseudo'):        
            nr_of_genes = len(Sequences.keys())
            new_pars = NewEmissionParameters['ExpressionParameters'][0]
            new_pars = np.vstack((new_pars[:(nr_of_genes), :], np.mean(new_pars[:(nr_of_genes), :]), new_pars[(nr_of_genes):, :]))
            NewEmissionParameters['ExpressionParameters'][0] = new_pars
    print 'Estimating expression parameters'
    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    bg_type = NewEmissionParameters['BckType']
    expr_data = (NewEmissionParameters, Sequences, Background, NewPaths, sample_size, bg_type)
    NewEmissionParameters = emission.estimate_expression_param(expr_data)

    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    
    if NewEmissionParameters['BckType'] != 'None':
        if Sequences.has_key('Pseudo'):        
            nr_of_genes = len(Sequences.keys())
            new_pars = NewEmissionParameters['ExpressionParameters'][0]
            new_pars = np.vstack((new_pars[:(nr_of_genes-1), :], new_pars[(nr_of_genes):, :]))
            NewEmissionParameters['ExpressionParameters'][0] = new_pars

    
    #Compute parameters for the ratios
    print 'computing sufficient statitics for fitting md'
    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    SuffStat = tools.GetSuffStat(Sequences, Background, NewPaths, NrOfStates, Type='Conv', EmissionParameters=NewEmissionParameters)
    
    #Vectorize SuffStat
    Counts, NrOfCounts = tools.ConvertSuffStatToArrays(SuffStat)

    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if NewEmissionParameters['Subsample']:
        Counts, NrOfCounts = tools.subsample_suff_stat(Counts, NrOfCounts)


    print 'fitting md distribution'
    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    if NewEmissionParameters['diag_bg']:
        print "Adjusting background"
        SuffStatBck = tools.GetSuffStatBck(Sequences, Background, NewPaths, NrOfStates, Type='Conv', EmissionParameters=NewEmissionParameters)
        #Vectorize SuffStat
        CountsBck, NrOfCountsBck = tools.ConvertSuffStatToArrays(SuffStatBck)

        if NewEmissionParameters['Subsample']:
            CountsBck, NrOfCountsBck = tools.subsample_suff_stat(CountsBck, NrOfCountsBck)
        #Overwrite counts in other bins
        fg_state, bg_state = emission.get_fg_and_bck_state(NewEmissionParameters, final_pred=True)
        for curr_state in Counts.keys():
            if curr_state != fg_state:
                Counts[curr_state] = CountsBck[fg_state]
                NrOfCounts[curr_state] = NrOfCountsBck[fg_state]
        #pdb.set_trace()

    cPickle.dump((Counts, NrOfCounts, CountsBck, NrOfCountsBck, NewPaths), open('/data/ohler/Philipp/tmp/diag_dump.dat','w'))

    NewEmissionParameters = mixture_tools.em(Counts, NrOfCounts, NewEmissionParameters, x_0=OldAlpha, First=First)

    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    if Sequences.has_key('Pseudo'):
        del Sequences['Pseudo']
        del Background['Pseudo']
        del NewPaths['Pseudo']

    print 'Done: Elapsed time: ' + str(time.time() - t)
    del Counts, NrOfCounts, SuffStat
    return NewEmissionParameters

def add_pseudo_gene(Sequences, Background, NewPaths, PriorMatrix):
    pseudo_gene_names = ['Pseudo']
    nr_of_genes_to_gen = np.sum(PriorMatrix == 0)

    #if no pseudo gen has tp be generated, continue
    if nr_of_genes_to_gen == 0:
        return Sequences, Background, NewPaths, pseudo_gene_names

    #Generate pseudo genes

    #Get the gene lengths
    gen_lens = [Sequences[gene]['Coverage'][0].shape[1] for gene in Sequences]

    random_gen  = np.random.choice(np.arange(len(gen_lens)), 1, p = np.array(gen_lens)/np.float(np.sum(np.array(gen_lens))))  
    gene_name = Sequences.keys()[random_gen]

    Sequences['Pseudo'] = Sequences[gene_name]
    Background['Pseudo'] = Background[gene_name]
    
    zero_states = [i for i in range(len(PriorMatrix))]

    new_path = np.random.choice(zero_states, size=NewPaths[gene_name].shape, replace=True)
    NewPaths['Pseudo'] = new_path

    pseudo_gene_names = ['Pseudo']
    return Sequences, Background, NewPaths, pseudo_gene_names



def ComputeLikelihood(Sequences, IterParameters):
    '''
    This function computes the log-likelihood of the FitModel
    '''
    LogLikelihood = 0


    return LogLikelihood



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='sclip')


    # Gene annotation
    parser.add_argument('--annot', action='store', dest='gene_anno_file', help='File where gene annotation is stored')
    parser.add_argument('--genome-dir', action='store', dest='genome_dir', help='Directory where fasta files are stored')

    # FG files
    parser.add_argument('--clip-files', action='append', dest='fg_libs', default=[], help='Bam-files for CLIP-libraries')

    # BG collapsed
    parser.add_argument('--restart-from-iter', action='store_true', default=False, dest='restart_from_file', help='restart from existing run')

    # Overwrite existing FG .dat files
    parser.add_argument('--overwrite-CLIP-data', action='store_false', default=True, dest='overwrite_fg', help='Overwrite the existing CLIP data')

    # FG collapsed
    parser.add_argument('--collapsed-CLIP', action='store_true', default=False, dest='fg_collapsed', help='CLIP-reads are collapsed')

    # BG files
    parser.add_argument('--bg-files', action='append', dest='bg_libs', default=[], help='Bam-files for bg-libraries or files with counts per gene')

    # Overwrite existing BG .dat files
    parser.add_argument('--overwrite-bg-data', action='store_false', default=True, dest='overwrite_bg', help='Overwrite the existing CLIP data')

    # BG collapsed
    parser.add_argument('--collapsed-bg', action='store_true', default=False, dest='bg_collapsed', help='bg-reads are collapsed')

    # Also load variants for background 
    parser.add_argument('--bck-var', action='store_false', default=True, dest='only_coverage', help='Parse variants for background reads')

    # BG type
    parser.add_argument('--bg-type', action='store', dest='bg_type', help='Background type', choices=['None', 'Coverage', 'Coverage_bck'], default='Coverage_bck')

    # Transistion type
    parser.add_argument('--trans-model', action='store', dest='tr_type', help='Transition type', choices=['binary', 'binary_bck', 'multi'], default='binary')

    # verbosity
    parser.add_argument('--verbosity', action='store', dest='verbosity', help='Verbosity', type=int, default=0)

    # save temporary results
    parser.add_argument('--save-tmp', action='store_false', default=True, dest='safe_tmp', help='Safe temporary results')

    # only predict sites using existing mode
    parser.add_argument('--pred-sites', action='store_true', default=False, dest='pred_sites', help='Only predict sites')

    # Likelihood treshold
    parser.add_argument('--thresh', action='store', dest='thresh', help='Likelihood threshold after which to stop the iterations', type=float)

    # max number of iterations 
    parser.add_argument('--tol-log-like', action='store', dest='tol_lg_lik', help='tolerance for lok-likelihood', type=float, default = 10000.0)

    # max number of iterations 
    parser.add_argument('--max-it', action='store', dest='max_it', help='Maximal number of iterations', type=int, default = 20)

    # max iterations for GLM
    parser.add_argument('--max-it-glm', action='store', dest='max_it_glm', help='Maximal number of iterations in GLM', type=int, default = 10)

    # Pseudo count for null state
    parser.add_argument('--pseudo', action='store', dest='pseudo_count', help='Pseudo count for null state', type=int)

    # Output directory
    parser.add_argument('--out-dir', action='store', dest='out_dir', help='Output directory for results')

    # Number of genes to sample
    parser.add_argument('--gene-sample', action='store', dest='gene_sample', help='Nr of genes to sample', type=int, default = 100000)
    # Disable subsampling for parameter estimation
    parser.add_argument('--no-subsample', action='store_false', default=True, dest='subs', help='Disabaple subsampling for parameter estimations (Warning: Leads to slow estimation)')

    #Do not fit diagnostic events at SNP-Positions
    parser.add_argument('--filter-snps', action='store_false', default=True, dest='filter_snps', help='Do not fit diagnostic events at SNP-positions')

    #Criterion for definition of SNPs
    parser.add_argument('--snp-ratio', action='store', dest='snps_thresh', help='Ratio of reads showing the SNP', type=float, default = 0.2)
    parser.add_argument('--snp-abs-cov', action='store', dest='snps_min_cov', help='Absolute number of reads covering the SNP position', type=float, default = 10)

    #Number of mixture components for the diagnostic event model
    parser.add_argument('--nr_mix_comp', action='store', dest='nr_mix_comp', help='Number of diagnostic events mixture components', type=int, default = 1)

    #Diagnostic event model
    parser.add_argument('--diag_event_mod', action='store', dest='diag_event_mod', help='Diagnostic event model', choices=['DirchMult', 'DirchMultK'], default='DirchMultK')

    # Number of cores to use
    parser.add_argument('--nb-cores', action='store', dest='nb_proc', help='Number of cores o use', type=int, default = 1)

    #Mask miRNA positions
    parser.add_argument('--mask-miRNA', action='store_false', default=True, dest='mask_miRNA', help='Mask miRNA positions')
    
    #Ignore overlping gene regions for diagnostic event model
    parser.add_argument('--mask-ovrlp', action='store_false', default=True, dest='mask_ovrlp', help='Ignore overlping gene regions for diagnostic event model fitting')

    #Ignore diagnostic event model for scoring
    parser.add_argument('--ign-diag', action='store_true', default=False, dest='ign_diag', help='Ignore diagnostic event model for scoring')

    #Ignore GLM model for scoring
    parser.add_argument('--ign-GLM', action='store_true', default=False, dest='ign_GLM', help='Ignore GLM model for scoring')

    #Only predict, do no model fiting
    parser.add_argument('--only-pred', action='store_true', default=False, dest='only_pred', help='only predict the sites. No model fitting')
    
    #Estimate diagnostic events on background
    parser.add_argument('--diag-bg', action='store_false', default=True, dest='diag_bg', help='estimate diagnostic events for the background states on the background')

    #Estimate diagnostic events on background
    parser.add_argument('--emp-var', action='store_true', default=False, dest='emp_var', help='use the empirical variance if it larger than the expected variance')

    #Normalize class weights during glm fit
    parser.add_argument('--norm_class', action='store_false', default=True, dest='norm_class', help='Normalize class weights during glm fit')

    # reweight glm and diagnostic events
    parser.add_argument('--glm_weight', action='store', dest='glm_weight', help='weight of the glm score with respect to the diagnostic event score', type=float, default = -1.0)

    # max mismatches per read
    parser.add_argument('--max-mismatch', action='store', dest='max_mm', help='Maximal number of mismatches that is allowed per read (default: 2)', type=int, default = 2)

    # mask read ends fo diag event counting
    parser.add_argument('--mask_flank_mm', action='store', dest='mask_flank_variants', help='Do not consider mismatches in the N bp at the ends of reads for diagnostic event modelling (default: 3)', type=int, default = 3)

    

    #Check if only sites should be predicted
    args = parser.parse_args()
    if args.pred_sites:
        pred_sites()
    else:
        #if not run sclip
        run_sclip(args)

