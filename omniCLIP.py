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


import sys
sys.path.append('./data_parsing/')
sys.path.append('./stat/')
sys.path.append('./visualisation/')
from collections import defaultdict
from intervaltree import Interval, IntervalTree
from scipy.sparse import *
import argparse
import pickle
import emission_prob
import gc
import gffutils
import h5py
import LoadReads
import mixture_tools
import numpy as np
import os
import random
import resource
import shutil
import tempfile
import time
import tools
import trans

##@profile
#@profile 
def run_omniCLIP(args):
    # Get the args
    args = parser.parse_args()

    verbosity = args.verbosity

    if verbosity > 1:
        print(args)

    #Check parameters
    if len(args.fg_libs) == 0:
        raise sys.exit('No CLIP-libraries given')

    if len(args.bg_libs) == 0:
        bg_type = 'None'
    else:
        bg_type = args.bg_type


    if args.out_dir == None:
        out_path = os.getcwd()
    else:
        out_path = args.out_dir

    MaxIter  = args.max_it
    # process the parameters

    if not (bg_type == 'Coverage' or  bg_type == 'Coverage_bck'):
        print('Bg-type: ' + bg_type + ' has not been implemented yet')
        return 
    
    #Set seed for the random number generators
    if args.rnd_seed is not None:
        random.seed(args.rnd_seed)
        print('setting seed')

    #Set the p-value cutoff for the bed-file creation
    pv_cutoff = args.pv_cutoff

    #Load the gene annotation
    print('Loading gene annotation')
    if args.gene_anno_file.split('.')[-1] == 'db':
        GeneAnnotation = gffutils.FeatureDB(args.gene_anno_file, keep_order=True)
    else:
        if os.path.isfile(args.gene_anno_file + '.db'):
            print('Using existing gene annotation database: ' + args.gene_anno_file + '.db')
            GeneAnnotation = gffutils.FeatureDB(args.gene_anno_file + '.db', keep_order=True)
        else:
            print('Creating gene annotation database')
            db = gffutils.create_db(args.gene_anno_file, dbfn=(args.gene_anno_file + '.db'), force=True, keep_order=True, merge_strategy='merge', sort_attribute_values=True, disable_infer_transcripts=True, disable_infer_genes=True)
            GeneAnnotation = gffutils.FeatureDB(args.gene_anno_file + '.db', keep_order=True)
            del db

    GenomeDir = args.genome_dir
    
    import warnings
    warnings.filterwarnings('error')


    #Load the reads
    if verbosity > 0:
        print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print('Loading reads')

    EmissionParameters = {}
 
    #Check whether existing iteration parameters should be used
    restart_from_file = args.restart_from_file
    EmissionParameters['restart_from_file'] = restart_from_file

    EmissionParameters['glm_weight'] = args.glm_weight

    EmissionParameters['mask_flank_variants'] = args.mask_flank_variants

    EmissionParameters['max_mm'] = args.max_mm

    EmissionParameters['rev_strand'] = args.rev_strand

    EmissionParameters['skip_diag_event_mdl'] = args.skip_diag_event_mdl

    EmissionParameters['ign_out_rds'] = args.ign_out_rds

    EmissionParameters['DataOutFile_seq'] = os.path.join(out_path, 'fg_reads.dat')
    EmissionParameters['DataOutFile_bck'] = os.path.join(out_path, 'bg_reads.dat')
    EmissionParameters['tmp_dir'] = args.tmp_dir
    t = time.time()

    Sequences = LoadReads.load_data(args.fg_libs, GenomeDir, GeneAnnotation, EmissionParameters['DataOutFile_seq'], load_from_file = ((not args.overwrite_fg) or restart_from_file), save_results = True, Collapse = args.fg_collapsed, mask_flank_variants=EmissionParameters['mask_flank_variants'], max_mm=EmissionParameters['max_mm'], ign_out_rds=EmissionParameters['ign_out_rds'], rev_strand=EmissionParameters['rev_strand'])
    Background = LoadReads.load_data(args.bg_libs, GenomeDir, GeneAnnotation, EmissionParameters['DataOutFile_bck'], load_from_file = ((not args.overwrite_bg) or restart_from_file), save_results = True, Collapse = args.bg_collapsed, OnlyCoverage = args.only_coverage,  mask_flank_variants=EmissionParameters['mask_flank_variants'], max_mm=EmissionParameters['max_mm'], ign_out_rds=EmissionParameters['ign_out_rds'], rev_strand=EmissionParameters['rev_strand'])
    #pdb.set_trace()
    #Mask the positions that overlap miRNA sites in the geneome
    
    Sequences.close()
    Background.close()

    f_name_read_fg = EmissionParameters['DataOutFile_seq']
    f_name_read_bg = EmissionParameters['DataOutFile_bck']

    #Create temporary read-files that can be modified by the masking operations
    if EmissionParameters['tmp_dir'] is None:
        f_name_read_fg_tmp = EmissionParameters['DataOutFile_seq'].replace('fg_reads.dat', 'fg_reads.tmp.dat')
        f_name_read_bg_tmp = EmissionParameters['DataOutFile_bck'].replace('bg_reads.dat', 'bg_reads.tmp.dat')
    else:
        f_name_read_fg_tmp = os.path.join(EmissionParameters['tmp_dir'], next(tempfile._get_candidate_names()) + '.dat') 
        f_name_read_bg_tmp = os.path.join(EmissionParameters['tmp_dir'], next(tempfile._get_candidate_names()) + '.dat') 
        
    shutil.copy(f_name_read_fg, f_name_read_fg_tmp)
    shutil.copy(f_name_read_bg, f_name_read_bg_tmp)

    #open the temporary read files
    Sequences = h5py.File(f_name_read_fg_tmp, 'r+')
    Background = h5py.File(f_name_read_bg_tmp, 'r+')

    EmissionParameters['DataOutFile_seq'] = f_name_read_fg_tmp
    EmissionParameters['DataOutFile_bck'] = f_name_read_bg_tmp
    

    #Set coverage for regions that overlapp annotated miRNAs to zero
    EmissionParameters['mask_miRNA'] = args.mask_miRNA
    if args.mask_miRNA: 
        print('Removing miRNA-coverage')
        Sequences = mask_miRNA_positions(Sequences, GeneAnnotation)

    #Mask regions where genes overlap
    EmissionParameters['mask_ovrlp'] = args.mask_ovrlp

    if EmissionParameters['mask_ovrlp']:
        print('Masking overlapping positions')
        Sequences = mark_overlapping_positions(Sequences, GeneAnnotation)

    #Estimate the library size
    EmissionParameters['BckLibrarySize'] =  tools.estimate_library_size(Background)
    EmissionParameters['LibrarySize'] = tools.estimate_library_size(Sequences)
    
    #Removing genes without any reads in the CLIP data
    print("Removing genes without CLIP coverage")

    genes_to_keep = []
    all_genes = list(Sequences.keys())
    for i, gene in enumerate(Sequences.keys()):
        curr_cov = sum([Sequences[gene]['Coverage'][rep][()].sum() for rep in list(Sequences[gene]['Coverage'].keys())])

        if curr_cov <= 100:
            continue

        genes_to_keep.append(gene)
        if i > args.gene_sample:
            break
    
    genes_to_del = list(set(all_genes).difference(set(genes_to_keep)))

    for gene in genes_to_del:
        del Sequences[gene]
        del Background[gene]

    del all_genes, genes_to_del, genes_to_keep 
    if verbosity > 0:
        print('Done: Elapsed time: ' + str(time.time() - t))
        print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    #Initializing parameters
    print('Initialising the parameters')
    if bg_type == 'Coverage_bck':
        NrOfStates = 4
    else:
        NrOfStates = 3

    #Remove the gene sequence from the Sequences and Background when not needed. Currently this is always the case:
    for gene in list(Sequences.keys()):
        if 'GeneSeq' in Sequences[gene]:
            del Sequences[gene]['GeneSeq']

    for gene in list(Background.keys()):
        if 'GeneSeq' in Background[gene]:
            del Background[gene]['GeneSeq']

    #pdb.set_trace()
    TransMat = np.ones((NrOfStates, NrOfStates)) + np.eye(NrOfStates)
    TransMat = TransMat / np.sum(np.sum(TransMat))
    TransitionParameters = [TransMat, []]

    NrOfReplicates = len(args.fg_libs)
    gene = list(Sequences.keys())[0]
    
    EmissionParameters['PriorMatrix'] = np.ones((NrOfStates, 1)) / float(NrOfStates)
    EmissionParameters['diag_bg'] = args.diag_bg
    EmissionParameters['emp_var'] = args.emp_var
    EmissionParameters['norm_class'] = args.norm_class

    #Define flag for penalized path prediction
    EmissionParameters['LastIter'] = False    
    EmissionParameters['fg_pen'] = args.fg_pen

    EmissionParameters['Diag_event_params'] = {}
    EmissionParameters['Diag_event_params']['nr_mix_comp'] = args.nr_mix_comp
    EmissionParameters['Diag_event_params']['mix_comp'] = {}
    for state in range(NrOfStates):
        mixtures = np.random.uniform(0.0, 1.0, size=(args.nr_mix_comp))
        EmissionParameters['Diag_event_params']['mix_comp'][state] = mixtures / np.sum(mixtures)
    
    #initialise the parameter vector alpha
    alphashape = (Sequences[gene]['Variants']['0']['shape'][0] + Sequences[gene]['Coverage']['0'][()].shape[0] + Sequences[gene]['Read-ends']['0'][()].shape[0])
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
    EmissionParameters['Verbosity'] = args.verbosity
    EmissionParameters['NbProc'] = args.nb_proc
    EmissionParameters['Subsample'] = args.subs

    EmissionParameters['FilterSNPs'] = args.filter_snps
    EmissionParameters['SnpRatio'] = args.snps_thresh
    EmissionParameters['SnpAbs'] = args.snps_min_cov
    EmissionParameters['ign_diag'] = args.ign_diag
    if EmissionParameters['ign_out_rds']:
        EmissionParameters['ign_diag'] = EmissionParameters['ign_out_rds']
    EmissionParameters['ign_GLM'] = args.ign_GLM
    EmissionParameters['only_pred'] = args.only_pred

    EmissionParameters['use_precomp_diagmod'] = args.use_precomp_diagmod

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
        IterParameters, args_old = pickle.load(open(IterSaveFile,'rb'))
        EmissionParameters['mask_miRNA'] = args.mask_miRNA
        EmissionParameters['glm_weight'] = args.glm_weight
        EmissionParameters['restart_from_file'] = restart_from_file
        EmissionParameters =  IterParameters[0]
        EmissionParameters['ign_diag'] = args.ign_diag
        if EmissionParameters['ign_out_rds']:
            EmissionParameters['ign_diag'] = EmissionParameters['ign_out_rds']
        EmissionParameters['ign_GLM'] = args.ign_GLM
        TransitionParameters = IterParameters[1]
        OldLogLikelihood = -np.inf
        fg_state, bg_state = emission_prob.get_fg_and_bck_state(EmissionParameters, final_pred=True)
        
        Paths, CurrLogLikelihood = tools.ParallelGetMostLikelyPath(Paths, Sequences, Background, EmissionParameters, TransitionParameters, 'nonhomo')
        Sequences = h5py.File(EmissionParameters['DataOutFile_seq'], 'r')
        Background = h5py.File(EmissionParameters['DataOutFile_bck'], 'r')

        First = 0
        iter_cond = False

    if restart_from_file:
        IterParameters, args_old = pickle.load(open(IterSaveFile,'rb'))
        EmissionParameters =  IterParameters[0]
        EmissionParameters['mask_miRNA'] = args.mask_miRNA
        EmissionParameters['glm_weight'] = args.glm_weight
        EmissionParameters['restart_from_file'] = restart_from_file
        EmissionParameters['ign_diag'] = args.ign_diag
        EmissionParameters['ign_GLM'] = args.ign_GLM
        TransitionParameters = IterParameters[1]
        OldLogLikelihood = -np.inf
        Paths, CurrLogLikelihood = tools.ParallelGetMostLikelyPath(Paths, Sequences, Background, EmissionParameters, TransitionParameters, 'nonhomo')
        Sequences = h5py.File(EmissionParameters['DataOutFile_seq'], 'r')
        Background = h5py.File(EmissionParameters['DataOutFile_bck'], 'r')
        First = 1
        iter_cond = True


    #import warnings
    #warnings.filterwarnings('error')



    if not EmissionParameters['use_precomp_diagmod'] is None:
        IterParametersPreComp, args_old = pickle.load(open(EmissionParameters['use_precomp_diagmod'],'r'))
        IterParameters[0]['Diag_event_params'] = IterParametersPreComp[0]['Diag_event_params']

    while iter_cond:
        print("\n")
        print("Iteration: " + str(CurrIter))
        if EmissionParameters['Verbosity'] > 1:
            print(IterParameters[0])

        OldLogLikelihood  = CurrLogLikelihood
        
        CurrLogLikelihood, IterParameters, First, Paths = PerformIteration(Sequences, Background, IterParameters, NrOfStates, First, Paths, verbosity=EmissionParameters['Verbosity'])
        gc.collect()
        
        if True:
            pickle.dump([IterParameters, args], open(IterSaveFile,'wb'))
        if args.safe_tmp:
            if CurrIter > 0:
                IterHist = pickle.load(open(IterSaveFileHist,'rb'))
            IterHist.append([IterParameters, CurrLogLikelihood])
            pickle.dump(IterHist, open(IterSaveFileHist,'wb'))
            del IterHist
        
        if verbosity > 1:
            print("Log-likelihood: " + str(CurrLogLikelihood)) 
        LoglikelihodList.append(CurrLogLikelihood)
        
        if verbosity > 1:
            print(LoglikelihodList)
        CurrIter += 1
        
        if CurrIter >= MaxIter:
            print('Maximal number of iterations reached')

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
    print('Finished parameter fitting')

    EmissionParameters, TransitionParameters = IterParameters
    if not isinstance(EmissionParameters['ExpressionParameters'][0], np.ndarray):
        print('Emmision parameters have not been fit yet')
        return
    out_file_base = 'pred'
    if EmissionParameters['ign_GLM']:
       out_file_base += '_no_glm'
    if EmissionParameters['ign_diag']:
       out_file_base += '_no_diag'
    OutFile = os.path.join(out_path, out_file_base + '.txt')
    #determine which state has higher weight in fg.
    if verbosity > 0:
        print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    fg_state, bg_state = emission_prob.get_fg_and_bck_state(EmissionParameters, final_pred=True)
    if EmissionParameters['fg_pen'] > 0.0:
        print('Recomputing paths')
        EmissionParameters['LastIter'] = True        
        Sequences = h5py.File(EmissionParameters['DataOutFile_seq'], 'r')
        Background = h5py.File(EmissionParameters['DataOutFile_bck'], 'r')
        Paths, LogLike = tools.ParallelGetMostLikelyPath(Paths, Sequences, Background, EmissionParameters, TransitionParameters, 'nonhomo', verbosity=EmissionParameters['Verbosity'])
        Sequences = h5py.File(EmissionParameters['DataOutFile_seq'], 'r')
        Background = h5py.File(EmissionParameters['DataOutFile_bck'], 'r')

    tools.GeneratePred(Paths, Sequences, Background, IterParameters, GeneAnnotation, OutFile, fg_state, bg_state, seq_file=EmissionParameters['DataOutFile_seq'], bck_file=EmissionParameters['DataOutFile_bck'], pv_cutoff=pv_cutoff, verbosity=EmissionParameters['Verbosity'])
    print('Done')

    #Remove the temporary files
    if not (EmissionParameters['tmp_dir'] is None):
        print('removing temporary files')
        os.remove(EmissionParameters['DataOutFile_seq'])
        os.remove(EmissionParameters['DataOutFile_bck'])

    return

##@profile
#@profile 
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
    for gene in list(gene_dict.values()):
        genes_chr_dict[gene.chrom].append(gene)

    #Create an interval tree for the genes:
    interval_chr_dict = {}
    for chr in list(genes_chr_dict.keys()):
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
                if curr_key in Sequences:
                    if curr_key in Sequences[curr_gene]:
                        for rep in list(Sequences[curr_gene][curr_key].keys()):
                            if curr_key == 'Variants':
                               #Convert the Variants to array 
                                curr_seq = csr_matrix((Sequences[curr_gene]['Variants'][rep]['data'][:],Sequences[curr_gene]['Variants'][rep]['indices'][:], 
                                    Sequences[curr_gene]['Variants'][rep]['indptr'][:]), shape=Sequences[curr_gene]['Variants'][rep]['shape'][:])
                                
                                ix_slice =  np.logical_and(curr_start <= curr_seq.indices, curr_seq.indices < curr_stop)
                                Sequences[curr_gene]['Variants'][rep]['data'][ix_slice] = 0
                            else:
                                curr_seq = Sequences[curr_gene][curr_key][rep][:, :]
                                curr_seq[:, curr_start: curr_stop] = 0
                                Sequences[curr_gene][curr_key][rep][:, :] = curr_seq

    return Sequences


#@profile 
def mark_overlapping_positions(Sequences, GeneAnnotation):
    '''
    This function takes the sequences and 
    the gene annotation and adds to Sequences a track that indicates the overlaping regions
    '''

    #add fields to Sequence structure:
    for gene in list(Sequences.keys()):
        Sequences[gene].create_group('mask')
        rep = list(Sequences[gene]['Coverage'].keys())[0]
        if rep == '0':
            Sequences[gene]['mask'].create_dataset(rep, data=np.zeros(Sequences[gene]['Coverage'][rep][()].shape), compression="gzip", compression_opts=9, chunks=Sequences[gene]['Coverage'][rep][()].shape, dtype='i8')

    #Create a dictionary that stores the genes in the Gene annnotation
    genes = [
        gene for gene in GeneAnnotation.features_of_type('gene')
        if '_PAR_Y' not in gene.id]

    gene_dict = {gene.id.split('.')[0]: gene for gene in genes}

    #Get Chromosomes:
    genes_chr_dict = defaultdict(list)
    for gene in list(gene_dict.values()):
        genes_chr_dict[gene.chrom].append(gene)

    #Create an interval tree for the genes:
    interval_chr_dict = {}
    for chr in list(genes_chr_dict.keys()):
        interval_chr_dict[chr] = IntervalTree()
        for gene in genes_chr_dict[chr]:
            interval_chr_dict[chr][gene.start : gene.stop] = gene
    

    #Iterate over the genes in the Sequences:
    for gene in genes:
        if not (gene.id.split('.')[0] in Sequences):
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
            rep = list(Sequences[gene.id.split('.')[0]]['Coverage'].keys())[0]
            Sequences[gene.id.split('.')[0]]['mask'][rep][0, ovrlp_start : ovrlp_stop] = True

    return Sequences



#@profile 
def pred_sites(args, verbosity=1):
    # Get the args

    args = parser.parse_args()
    print(args)

    #Check parameters
    if len(args.fg_libs) == 0:
        raise sys.exit('No CLIP-libraries given')

    if len(args.bg_libs) == 0:
        bg_type = 'None'
    else:
        bg_type = args.bg_type


    if args.out_dir == None:
        out_path = os.getcwd()
    else:
        out_path = args.out_dir

    MaxIter  = args.max_it
    # process the parameters

    if not (bg_type == 'Coverage' or  bg_type == 'Coverage_bck'):
        print('Bg-type: ' + bg_type + ' has not been implemented yet')
        return 

    #Load the gene annotation
    print('Loading gene annotation')
    GeneAnnotation = gffutils.FeatureDB(args.gene_anno_file, keep_order=True)
    GenomeDir = args.genome_dir

    #Load the reads
    t = time.time()
    print('Loading reads')
    DataOutFile = os.path.join(out_path, 'fg_reads.dat')
    Sequences = LoadReads.load_data(args.fg_libs, GenomeDir, GeneAnnotation, DataOutFile, load_from_file = True, save_results = False, Collapse = args.fg_collapsed, ign_out_rds=EmissionParameters['ign_out_rds'], rev_strand=EmissionParameters['rev_strand'])
    
    DataOutFile = os.path.join(out_path, 'bg_reads.dat')
    Background = LoadReads.load_data(args.bg_libs, GenomeDir, GeneAnnotation, DataOutFile, load_from_file = True, save_results = False, Collapse = args.bg_collapsed, OnlyCoverage = True, ign_out_rds=EmissionParameters['ign_out_rds'], rev_strand=EmissionParameters['rev_strand'])

    
    #Removing genes without any reads in the CLIP data
    genes_to_keep = []
    all_genes = list(Sequences.keys())
    for i, gene in enumerate(Sequences.keys()):
        curr_cov = np.sum(np.array([np.sum(Sequences[gene]['Coverage'][rep].toarray()) for rep in list(Sequences[gene]['Coverage'].keys())]))

        if curr_cov < 100:
            continue

        genes_to_keep.append(gene)
        if i > args.gene_sample:
            break
    
    genes_to_del = list(set(all_genes).difference(set(genes_to_keep)))

    for gene in genes_to_del:
        del Sequences[gene]
        del Background[gene]

    del all_genes, genes_to_del, genes_to_keep 
    if verbosity > 0:
        print('Done: Elapsed time: ' + str(time.time() - t))
    
    #Load data
    tmp_file = pickle.load(open(os.path.join(out_path, 'IterSaveFile.dat'), 'rb'))
    IterParameters = tmp_file[0]
    args = tmp_file[1]
    EmissionParameters = IterParameters[0]
    fg_state, bg_state = emission_prob.get_fg_and_bck_state(EmissionParameters, final_pred=True)
    if EmissionParameters['fg_pen'] > 0.0:
        print('Recomputing paths')
        EmissionParameters['LastIter'] = True        
        Sequences = h5py.File(EmissionParameters['DataOutFile_seq'], 'r')
        Background = h5py.File(EmissionParameters['DataOutFile_bck'], 'r')
        Paths, LogLike = tools.ParallelGetMostLikelyPath(Paths, Sequences, Background, EmissionParameters, TransitionParameters, 'nonhomo', verbosity=EmissionParameters['Verbosity'])
        Sequences = h5py.File(EmissionParameters['DataOutFile_seq'], 'r')
        Background = h5py.File(EmissionParameters['DataOutFile_bck'], 'r')

    tools.GeneratePred(Sequences, Background, IterParameters, GeneAnnotation, OutFile, fg_state, bg_state, verbosity=EmissionParameters['Verbosity'])

    print('Done')



##@profile
#@profile 
def PerformIteration(Sequences, Background, IterParameters, NrOfStates, First, NewPaths={}, verbosity=1):
    '''
    This function performs an iteration of the HMM algorithm 
    '''
    #unpack the Iteration parameters
    EmissionParameters = IterParameters[0]
    TransitionParameters = IterParameters[1]

    #Get new most likely path
    if (not EmissionParameters['restart_from_file']) and First:
        NewPaths, LogLike = tools.ParallelGetMostLikelyPath(NewPaths, Sequences, Background, EmissionParameters, TransitionParameters, 'homo', RandomNoise = True, verbosity=verbosity)
        Sequences = h5py.File(EmissionParameters['DataOutFile_seq'], 'r')
        Background = h5py.File(EmissionParameters['DataOutFile_bck'], 'r')

        if verbosity > 0:
            print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    
    #Perform EM to compute the new emission parameters
    print('Fitting emission parameters')
    if verbosity > 0:
        print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    NewEmissionParameters = FitEmissionParameters(Sequences, Background, NewPaths, EmissionParameters, First, verbosity=verbosity)
    if First:
        First = 0
    if verbosity > 0:
        print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    #Fit the transition matrix parameters
    NewTransitionParameters = TransitionParameters
    C = 1
    print('Fitting transition parameters')
    if verbosity > 0:
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

    TransistionPredictors = trans.FitTransistionParameters(Sequences, Background, TransitionParameters, NewPaths, C, verbosity=verbosity)
    NewTransitionParameters[1] = TransistionPredictors
    if verbosity > 0:
        print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    NewIterParameters = [NewEmissionParameters, NewTransitionParameters]
    
    print('Computing most likely path')
    if verbosity > 0:
        print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    
    gc.collect()
    NewPaths, LogLike = tools.ParallelGetMostLikelyPath(NewPaths, Sequences, Background, EmissionParameters, TransitionParameters, 'nonhomo', verbosity=verbosity)
    Sequences = h5py.File(EmissionParameters['DataOutFile_seq'], 'r')
    Background = h5py.File(EmissionParameters['DataOutFile_bck'], 'r')

    CurrLogLikelihood = LogLike
    if verbosity > 0:
            print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if verbosity > 1:
        print('LogLik:')
        print(CurrLogLikelihood)
    return CurrLogLikelihood, NewIterParameters, First, NewPaths

##@profile
#@profile 
def FitEmissionParameters(Sequences, Background, NewPaths, OldEmissionParameters, First, verbosity=1):
    print('Fitting emission parameters')
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

    #Check if one of the states is not used and add pseudo gene to prevent singularities during distribution fitting 
    if np.sum(PriorMatrix == 0) > 0:
        Sequences.close()
        Background.close()
        Sequences = h5py.File(NewEmissionParameters['DataOutFile_seq'], 'r+')
        Background = h5py.File(NewEmissionParameters['DataOutFile_bck'], 'r+')
        Sequences, Background, NewPaths, pseudo_gene_names = add_pseudo_gene(Sequences, Background, NewPaths, PriorMatrix)
        Sequences.close()
        Background.close()
        print('Addes pseudo gene to prevent singular matrix during GLM fitting')

    CorrectedPriorMatrix = np.copy(PriorMatrix)
    
    CorrectedPriorMatrix[CorrectedPriorMatrix == 0] = np.min(CorrectedPriorMatrix[CorrectedPriorMatrix > 0])/10 
    CorrectedPriorMatrix /= np.sum(CorrectedPriorMatrix)
    #Keep a copy to check which states are not used
    NewEmissionParameters['PriorMatrix'] = CorrectedPriorMatrix

    #Add Pseudo gene to Sequences, Background and Paths
    if NewEmissionParameters['ExpressionParameters'][0] is not None:
        Sequences, Background, NewPaths, pseudo_gene_names = add_pseudo_gene(Sequences, Background, NewPaths, PriorMatrix)

    #Compute parameters for the expression
    sample_size = 10000

    if NewEmissionParameters['BckType'] != 'None':
        if 'Pseudo' in Sequences:        
            nr_of_genes = len(list(Sequences.keys()))
            new_pars = NewEmissionParameters['ExpressionParameters'][0]
            new_pars = np.vstack((new_pars[:(nr_of_genes), :], np.mean(new_pars[:(nr_of_genes), :]), new_pars[(nr_of_genes):, :]))
            NewEmissionParameters['ExpressionParameters'][0] = new_pars
    print('Estimating expression parameters')
    if verbosity > 0:
        print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    bg_type = NewEmissionParameters['BckType']
    expr_data = (NewEmissionParameters, Sequences, Background, NewPaths, sample_size, bg_type)
    NewEmissionParameters = emission_prob.estimate_expression_param(expr_data, verbosity=verbosity)
    
    if verbosity > 0:
        print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    
    if NewEmissionParameters['BckType'] != 'None':
        if 'Pseudo' in Sequences:        
            nr_of_genes = len(list(Sequences.keys()))
            new_pars = NewEmissionParameters['ExpressionParameters'][0]
            new_pars = np.vstack((new_pars[:(nr_of_genes-1), :], new_pars[(nr_of_genes):, :]))
            NewEmissionParameters['ExpressionParameters'][0] = new_pars
    
    if (NewEmissionParameters['skip_diag_event_mdl'] == False) or (not (EmissionParameters['use_precomp_diagmod'] is None)):
        #Compute parameters for the ratios
        print('Computing sufficient statistic for fitting md')
        if verbosity > 0:
            print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        SuffStat = tools.GetSuffStat(Sequences, Background, NewPaths, NrOfStates, Type='Conv', EmissionParameters=NewEmissionParameters, verbosity=verbosity)
        
        #Vectorize SuffStat
        Counts, NrOfCounts = tools.ConvertSuffStatToArrays(SuffStat)

        del SuffStat
        if verbosity > 0:
            print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if NewEmissionParameters['Subsample']:
            Counts, NrOfCounts = tools.subsample_suff_stat(Counts, NrOfCounts)


        print('Fitting md distribution')
        if verbosity > 0:
            print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        if NewEmissionParameters['diag_bg']:
            print("Adjusting background")
            SuffStatBck = tools.GetSuffStatBck(Sequences, Background, NewPaths, NrOfStates, Type='Conv', EmissionParameters=NewEmissionParameters, verbosity=verbosity)
            #Vectorize SuffStat
            CountsBck, NrOfCountsBck = tools.ConvertSuffStatToArrays(SuffStatBck)

            if NewEmissionParameters['Subsample']:
                CountsBck, NrOfCountsBck = tools.subsample_suff_stat(CountsBck, NrOfCountsBck)
            
            #Overwrite counts in other bins
            fg_state, bg_state = emission_prob.get_fg_and_bck_state(NewEmissionParameters, final_pred=True)
            for curr_state in list(Counts.keys()):
                if curr_state != fg_state:
                    Counts[curr_state] = CountsBck[fg_state]
                    NrOfCounts[curr_state] = NrOfCountsBck[fg_state]

            del SuffStatBck
            
        NewEmissionParameters = mixture_tools.em(Counts, NrOfCounts, NewEmissionParameters, x_0=OldAlpha, First=First, verbosity=verbosity)
        if verbosity > 0:
            print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        del Counts, NrOfCounts
    
    if 'Pseudo' in Sequences:
        del Sequences['Pseudo']
        del Background['Pseudo']
        del NewPaths['Pseudo']

    if verbosity > 0:
        print('Done: Elapsed time: ' + str(time.time() - t))
    return NewEmissionParameters

#@profile 
def add_pseudo_gene(Sequences, Background, NewPaths, PriorMatrix):
    pseudo_gene_names = ['Pseudo']
    nr_of_genes_to_gen = np.sum(PriorMatrix == 0)

    #if no pseudo gen has tp be generated, continue
    if nr_of_genes_to_gen == 0:
        return Sequences, Background, NewPaths, pseudo_gene_names

    #Generate pseudo genes
    #Get the gene lengths
    gen_lens = [Sequences[gene]['Coverage']['0'][()].shape[1] for gene in Sequences]

    random_gen  = np.random.choice(np.arange(len(gen_lens)), 1, p = np.array(gen_lens)/np.float(sum(gen_lens)))  
    
    if len(random_gen) == 1:
        gene_name = list(Sequences.keys())[random_gen[0]]
    else:
        gene_name = list(Sequences.keys())[random_gen]

    Sequences['Pseudo'] = Sequences[gene_name]
    Background['Pseudo'] = Background[gene_name]
    
    zero_states = [i for i in range(len(PriorMatrix))]

    new_path = np.random.choice(zero_states, size=NewPaths[gene_name].shape, replace=True)
    NewPaths['Pseudo'] = new_path

    pseudo_gene_names = ['Pseudo']
    return Sequences, Background, NewPaths, pseudo_gene_names

#@profile 
def ComputeLikelihood(Sequences, IterParameters):
    '''
    This function computes the log-likelihood of the FitModel
    '''
    LogLikelihood = 0

    return LogLikelihood

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='omniCLIP')


    # Gene annotation
    parser.add_argument('--annot', action='store', dest='gene_anno_file', help='File where gene annotation is stored')
    parser.add_argument('--genome-dir', action='store', dest='genome_dir', help='Directory where fasta files are stored')

    # FG files
    parser.add_argument('--clip-files', action='append', dest='fg_libs', default=[], help='Bam-files for CLIP-libraries')

    # BG collapsed
    parser.add_argument('--restart-from-iter', action='store_true', default=False, dest='restart_from_file', help='restart from existing run')

    # Overwrite existing FG .dat files
    parser.add_argument('--use-precomp-CLIP-data', action='store_false', default=True, dest='overwrite_fg', help='Use existing fg_data.dat file')

    # FG collapsed
    parser.add_argument('--collapsed-CLIP', action='store_true', default=False, dest='fg_collapsed', help='CLIP-reads are collapsed')

    # BG files
    parser.add_argument('--bg-files', action='append', dest='bg_libs', default=[], help='Bam-files for bg-libraries or files with counts per gene')

    # Overwrite existing BG .dat files
    parser.add_argument('--use-precomp-bg-data', action='store_false', default=True, dest='overwrite_bg', help='Use existing bg_data.dat data')

    # BG collapsed
    parser.add_argument('--collapsed-bg', action='store_true', default=False, dest='bg_collapsed', help='bg-reads are collapsed')

    #Also load variants for background 
    parser.add_argument('--bck-var', action='store_false', default=True, dest='only_coverage', help='Parse variants for background reads')

    # BG type
    parser.add_argument('--bg-type', action='store', dest='bg_type', help='Background type', choices=['None', 'Coverage', 'Coverage_bck'], default='Coverage_bck')

    # verbosity
    parser.add_argument('--verbosity', action='store', dest='verbosity', help='Verbosity: 0 (default) gives information of current state of site prediction, 1 gives aditional output on runtime and meomry consupmtiona and 2 shows selected internal variables', type=int, default=0)

    # save temporary results
    parser.add_argument('--save-tmp', action='store_true', default=False, dest='safe_tmp', help='Safe temporary results')

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

    # Output directory
    parser.add_argument('--tmp-dir', action='store', default=None, dest='tmp_dir', help='Output directory for temporary results')

    # Number of genes to sample
    parser.add_argument('--gene-sample', action='store', dest='gene_sample', help='Nr of genes to sample', type=int, default = 100000)
    # Disable subsampling for parameter estimation
    parser.add_argument('--no-subsample', action='store_false', default=True, dest='subs', help='Disabaple subsampling for parameter estimations (Warning: Leads to slow estimation)')

    #Do not fit diagnostic events at SNP-Positions
    parser.add_argument('--filter-snps', action='store_true', default=False, dest='filter_snps', help='Do not fit diagnostic events at SNP-positions')

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
    parser.add_argument('--mask-miRNA', action='store_true', default=False, dest='mask_miRNA', help='Mask miRNA positions')
    
    #Ignore overlping gene regions for diagnostic event model
    parser.add_argument('--mask-ovrlp', action='store_true', default=True, dest='mask_ovrlp', help='Ignore overlping gene regions for diagnostic event model fitting')

    #Ignore diagnostic event model for scoring
    parser.add_argument('--ign-diag', action='store_true', default=False, dest='ign_diag', help='Ignore diagnostic event model for scoring')

    #Ignore GLM model for scoring
    parser.add_argument('--ign-GLM', action='store_true', default=False, dest='ign_GLM', help='Ignore GLM model for scoring')

    #Only predict, do no model fiting
    parser.add_argument('--only-pred', action='store_true', default=False, dest='only_pred', help='only predict the sites. No model fitting')
    
    #Estimate diagnostic events on background
    parser.add_argument('--diag-bg', action='store_true', default=False, dest='diag_bg', help='estimate diagnostic events for the background states on the background')

    #Estimate diagnostic events on background
    parser.add_argument('--emp-var', action='store_true', default=False, dest='emp_var', help='use the empirical variance if it larger than the expected variance')

    #Normalize class weights during glm fit
    parser.add_argument('--norm_class', action='store_true', default=False, dest='norm_class', help='Normalize class weights during glm fit')

    # reweight glm and diagnostic events
    parser.add_argument('--glm_weight', action='store', dest='glm_weight', help='weight of the glm score with respect to the diagnostic event score', type=float, default = -1.0)

    # max mismatches per read
    parser.add_argument('--max-mismatch', action='store', dest='max_mm', help='Maximal number of mismatches that is allowed per read (default: 2)', type=int, default = 2)

    # mask read ends fo diag event counting
    parser.add_argument('--mask_flank_mm', action='store', dest='mask_flank_variants', help='Do not consider mismatches in the N bp at the ends of reads for diagnostic event modelling (default: 3)', type=int, default = 3)

    # ignore reads where the ends map ouside of the genes
    parser.add_argument('--ign_out_rds', action='store_true', dest='ign_out_rds', help='ignore reads where the ends map ouside of the genes', default = False)

    # Do not model the diagnostic events
    parser.add_argument('--skip_diag_event_mdl', action='store_true', dest='skip_diag_event_mdl', help='Do not model the diagnostic events', default = False)

    # reweight glm and diagnostic events
    parser.add_argument('--fg_pen', action='store', dest='fg_pen', help='Penalty for fg during scoring', type=float, default = 0.0)
    
    # only consider reads on a given strand
    parser.add_argument('--rev_strand', action='store', dest='rev_strand', help='Only consider reads on the forward (0) or reverse strand (1) relative to the gene orientation', type=int, default = None)
    
    # only consider reads on a given strand
    parser.add_argument('--use_precomp_diagmod', action='store', dest='use_precomp_diagmod', help='Use a precomputed diagnostic event model (Stored in ', type=str, default = None)
    
    # mask read ends fo diag event counting
    parser.add_argument('--seed', action='store', default=None, dest='rnd_seed', help='Set a seed for the random number generators')

    # bonferroni cutoff
    parser.add_argument('--pv', action='store', dest='pv_cutoff', help='bonferroni corrected p-value cutoff for peaks in bed-file', type=float, default = 0.05)


    #Check if only sites should be predicted
    args = parser.parse_args()
    if args.pred_sites:
        pred_sites()
    else:
        #if not run omniCLIP
        run_omniCLIP(args)

