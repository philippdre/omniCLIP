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

import argparse
import gc
import gffutils
import h5py
import numpy as np
import os
import time

from omniCLIP.data_parsing import tools
from omniCLIP.data_parsing import CreateGeneAnnotDB
from omniCLIP.data_parsing import LoadReads
from omniCLIP.data_parsing import ParsingArgs
from omniCLIP.data_parsing import ParsingPositions
from omniCLIP.omni_stat import emission_prob
from omniCLIP.omni_stat import mixture_tools
from omniCLIP.omni_stat import trans
from omniCLIP.omni_stat.utils import get_mem_usage


def run_omniCLIP(args):
    """Run omniCLIP function."""
    # Parsing the arguments
    EmissionParameters = ParsingArgs.parsing_argparse(args)

    # Creating temp copies of Sequence and Background
    ParsingArgs.dup_seqfiles(EmissionParameters)

    # Parsing arguments dependents of Sequence and Background
    EmissionParameters = ParsingArgs.parsing_files(args, EmissionParameters)

    # Load the gene annotation
    print('Loading gene annotation')
    if args.gene_anno_file.split('.')[-1] == 'db':
        GeneAnnotation = gffutils.FeatureDB(args.gene_anno_file, keep_order=True)

    import warnings
    warnings.filterwarnings('error')

    # Load the reads
    get_mem_usage(EmissionParameters['verbosity'])
    t = time.time()
    Sequences = h5py.File(EmissionParameters['dat_file_clip'], 'r+')
    Background = h5py.File(EmissionParameters['dat_file_bg'], 'r+')

    msg = 'Done: Elapsed time: ' + str(time.time() - t)
    get_mem_usage(EmissionParameters['verbosity'], t=t, msg=msg)

    # Initializing parameters
    print('Initialising the parameters')
    TransitionParameters = [EmissionParameters['TransMat'], []]

    # Transistion parameters
    IterParameters = [EmissionParameters, TransitionParameters]

    # Start computation

    # Iteratively fit the parameters of the model
    OldLogLikelihood = 0
    CurrLogLikelihood = -np.inf
    CurrIter = 0
    LoglikelihodList = []
    First = 1
    Paths = {}
    iter_cond = True

    while iter_cond:
        print("\n")
        print("Iteration: " + str(CurrIter))
        if EmissionParameters['verbosity'] > 1:
            print(IterParameters[0])

        OldLogLikelihood = CurrLogLikelihood

        CurrLogLikelihood, IterParameters, First, Paths = PerformIteration(
            Sequences, Background, IterParameters,
            EmissionParameters['NrOfStates'], First, Paths,
            verbosity=EmissionParameters['verbosity'])

        gc.collect()

        if EmissionParameters['verbosity'] > 1:
            print("Log-likelihood: " + str(CurrLogLikelihood))
        LoglikelihodList.append(CurrLogLikelihood)

        if EmissionParameters['verbosity'] > 1:
            print(LoglikelihodList)
        CurrIter += 1

        if CurrIter >= EmissionParameters['max_it']:
            print('Maximal number of iterations reached')

        if CurrIter < max(3, EmissionParameters['max_it']):
            iter_cond = True
        else:
            iter_cond = (
                (CurrIter < EmissionParameters['max_it'])
                and ((abs(CurrLogLikelihood - OldLogLikelihood)/max(abs(CurrLogLikelihood), abs(OldLogLikelihood))) > 0.01)
                and (abs(CurrLogLikelihood - OldLogLikelihood) > args.tol_lg_lik))

    # Return the fitted parameters
    print('Finished parameter fitting')

    EmissionParameters, TransitionParameters = IterParameters
    if not isinstance(EmissionParameters['ExpressionParameters'][0], np.ndarray):
        print('Emmision parameters have not been fit yet')
        return

    OutFile = os.path.join(EmissionParameters['out_dir'],
                           EmissionParameters['out_file_base'] + '.txt')

    # Determine which state has higher weight in fg.
    get_mem_usage(EmissionParameters['verbosity'])

    fg_state, bg_state = emission_prob.get_fg_and_bck_state(EmissionParameters,
                                                            final_pred=True)
    if EmissionParameters['fg_pen'] > 0.0:
        print('Recomputing paths')
        EmissionParameters['LastIter'] = True
        Sequences = h5py.File(EmissionParameters['dat_file_clip'], 'r')
        Background = h5py.File(EmissionParameters['dat_file_bg'], 'r')

        Paths, LogLike = tools.ParallelGetMostLikelyPath(
            Paths, Sequences, Background, EmissionParameters,
            TransitionParameters, 'nonhomo',
            verbosity=EmissionParameters['verbosity'])

        Sequences = h5py.File(EmissionParameters['dat_file_clip'], 'r')
        Background = h5py.File(EmissionParameters['dat_file_bg'], 'r')

    tools.GeneratePred(
        Paths, Sequences, Background, IterParameters, GeneAnnotation, OutFile,
        fg_state, bg_state, pv_cutoff=EmissionParameters['pv_cutoff'],
        verbosity=EmissionParameters['verbosity'])
    print('Done')

    # Remove the temporary files
    print('Removing temporary files')
    os.remove(EmissionParameters['dat_file_clip'])
    os.remove(EmissionParameters['dat_file_bg'])

    return


def PerformIteration(Sequences, Background, IterParameters, NrOfStates, First,
                     NewPaths={}, verbosity=1):
    """
    This function performs an iteration of the HMM algorithm
    """
    # Unpack the Iteration parameters
    EmissionParameters = IterParameters[0]
    TransitionParameters = IterParameters[1]
    TransitionType = EmissionParameters['TransitionType']

    # Get new most likely path
    if First:
        NewPaths, LogLike = tools.ParallelGetMostLikelyPath(
            NewPaths, Sequences, Background, EmissionParameters,
            TransitionParameters, 'homo', RandomNoise=True,
            verbosity=verbosity)

        Sequences = h5py.File(EmissionParameters['dat_file_clip'], 'r')
        Background = h5py.File(EmissionParameters['dat_file_bg'], 'r')

        get_mem_usage(verbosity)

    # Perform EM to compute the new emission parameters
    print('Fitting emission parameters')
    get_mem_usage(verbosity)

    NewEmissionParameters = FitEmissionParameters(
        Sequences, Background, NewPaths, EmissionParameters, First,
        verbosity=verbosity)

    if First:
        First = 0

    get_mem_usage(verbosity)

    # Fit the transition matrix parameters
    NewTransitionParameters = TransitionParameters
    print('Fitting transition parameters')
    get_mem_usage(verbosity)

    LoadReads.close_data_handles(handles=[Sequences, Background])
    Sequences = h5py.File(EmissionParameters['dat_file_clip'], 'r')
    Background = h5py.File(EmissionParameters['dat_file_bg'], 'r')

    TransistionPredictors = trans.FitTransistionParameters(
        Sequences, Background, TransitionParameters, NewPaths,
        TransitionType, verbosity=verbosity)

    NewTransitionParameters[1] = TransistionPredictors
    get_mem_usage(verbosity)

    NewIterParameters = [NewEmissionParameters, NewTransitionParameters]

    print('Computing most likely path')
    get_mem_usage(verbosity)

    gc.collect()
    NewPaths, LogLike = tools.ParallelGetMostLikelyPath(
        NewPaths, Sequences, Background, EmissionParameters,
        TransitionParameters, 'nonhomo', verbosity=verbosity)

    Sequences = h5py.File(EmissionParameters['dat_file_clip'], 'r')
    Background = h5py.File(EmissionParameters['dat_file_bg'], 'r')

    CurrLogLikelihood = LogLike
    get_mem_usage(verbosity)
    if verbosity > 1:
        print('LogLik:')
        print(CurrLogLikelihood)
    return CurrLogLikelihood, NewIterParameters, First, NewPaths


def FitEmissionParameters(Sequences, Background, NewPaths,
                          OldEmissionParameters, First, verbosity=1):
    """Fit EmissionParameters."""
    print('Fitting emission parameters')
    t = time.time()
    # Unpack the arguments
    OldAlpha = OldEmissionParameters['Diag_event_params']
    NrOfStates = OldEmissionParameters['NrOfStates']
    OldPriorMatrix = OldEmissionParameters['PriorMatrix']
    NewEmissionParameters = OldEmissionParameters

    # Compute new prior matrix
    PriorMatrix = np.zeros_like(OldPriorMatrix)
    for State in range(NrOfStates):
        for path in NewPaths:
            PriorMatrix[State] += np.sum(NewPaths[path] == State)

    # Check if one of the states is not used and add pseudo gene to prevent
    # singularities during distribution fitting
    if np.sum(PriorMatrix == 0) > 0:
        LoadReads.close_data_handles(handles=[Sequences, Background])
        Sequences = h5py.File(NewEmissionParameters['dat_file_clip'], 'r+')
        Background = h5py.File(NewEmissionParameters['dat_file_bg'], 'r+')
        Sequences, Background, NewPaths = add_pseudo_gene(
            Sequences, Background, NewPaths, PriorMatrix)
        print('Adds pseudo gene to prevent singular matrix during GLM fitting')

    CorrPriorMatrix = np.copy(PriorMatrix)
    CorrPriorMatrix[CorrPriorMatrix == 0] = np.min(
        CorrPriorMatrix[CorrPriorMatrix > 0])/10
    CorrPriorMatrix /= np.sum(CorrPriorMatrix)
    # Keep a copy to check which states are not used
    NewEmissionParameters['PriorMatrix'] = CorrPriorMatrix

    # Add Pseudo gene to Sequences, Background and Paths
    if NewEmissionParameters['ExpressionParameters'][0] is not None:
        Sequences, Background, NewPaths = add_pseudo_gene(
            Sequences, Background, NewPaths, PriorMatrix)

    # Compute parameters for the expression
    Sequences = h5py.File(NewEmissionParameters['dat_file_clip'], 'r')
    if (NewEmissionParameters['bg_type'] != 'None') and not First:
        if 'Pseudo' in list(Sequences.keys()):
            nr_of_genes = len(list(Sequences.keys()))
            new_pars = NewEmissionParameters['ExpressionParameters'][0]
            new_pars = np.vstack(
                (new_pars[:(nr_of_genes), :],
                 np.mean(new_pars[:(nr_of_genes), :]),
                 new_pars[(nr_of_genes):, :]))
            NewEmissionParameters['ExpressionParameters'][0] = new_pars
    print('Estimating expression parameters')
    get_mem_usage(verbosity)

    NewEmissionParameters = emission_prob.estimate_expression_param(
        (NewEmissionParameters, NewPaths), verbosity=verbosity)

    get_mem_usage(verbosity)

    Sequences = h5py.File(NewEmissionParameters['dat_file_clip'], 'r')
    Background = h5py.File(NewEmissionParameters['dat_file_bg'], 'r')

    if NewEmissionParameters['bg_type'] != 'None':
        if 'Pseudo' in list(Sequences.keys()):
            nr_of_genes = len(list(Sequences.keys()))
            new_pars = NewEmissionParameters['ExpressionParameters'][0]
            new_pars = np.vstack((new_pars[:(nr_of_genes-1), :],
                                  new_pars[(nr_of_genes):, :]))
            NewEmissionParameters['ExpressionParameters'][0] = new_pars

    if NewEmissionParameters['skip_diag_event_mdl'] is False:
        # Compute parameters for the ratios
        print('Computing sufficient statistic for fitting md')
        get_mem_usage(verbosity)

        SuffStat = tools.GetSuffStat(
            NewPaths, NrOfStates, Type='Conv',
            EmissionParameters=NewEmissionParameters, verbosity=verbosity)

        # Vectorize SuffStat
        Counts, NrOfCounts = tools.ConvertSuffStatToArrays(SuffStat)

        del SuffStat
        get_mem_usage(verbosity)
        if NewEmissionParameters['subs']:
            Counts, NrOfCounts = tools.subsample_suff_stat(Counts, NrOfCounts)

        print('Fitting md distribution')
        get_mem_usage(verbosity)
        if NewEmissionParameters['diag_bg']:
            print("Adjusting background")
            SuffStatBck = tools.GetSuffStatBck(
                NewPaths, NrOfStates, Type='Conv',
                EmissionParameters=NewEmissionParameters, verbosity=verbosity)

            # Vectorize SuffStat
            CountsBck, NrOfCountsBck = tools.ConvertSuffStatToArrays(
                SuffStatBck)

            if NewEmissionParameters['subs']:
                CountsBck, NrOfCountsBck = tools.subsample_suff_stat(
                    CountsBck, NrOfCountsBck)

            # Overwrite counts in other bins
            fg_state, bg_state = emission_prob.get_fg_and_bck_state(
                NewEmissionParameters, final_pred=True)

            for curr_state in list(Counts.keys()):
                if curr_state != fg_state:
                    Counts[curr_state] = CountsBck[fg_state]
                    NrOfCounts[curr_state] = NrOfCountsBck[fg_state]

            del SuffStatBck

        NewEmissionParameters = mixture_tools.em(
            Counts, NrOfCounts, NewEmissionParameters, x_0=OldAlpha,
            First=First, verbosity=verbosity)

        get_mem_usage(verbosity)
        del Counts, NrOfCounts

    if 'Pseudo' in list(Sequences.keys()):
        del Sequences['Pseudo']
        del Background['Pseudo']
        del NewPaths['Pseudo']

    if verbosity > 0:
        print('Done: Elapsed time: ' + str(time.time() - t))
    return NewEmissionParameters


def add_pseudo_gene(Sequences, Background, NewPaths, PriorMatrix):
    """Add pseudo gene to Sequences and Background."""
    nr_of_genes_to_gen = np.sum(PriorMatrix == 0)

    # If no pseudo gen has tp be generated, continue
    if nr_of_genes_to_gen == 0:
        return Sequences, Background, NewPaths

    # If pseudo is already in Sequences, skip
    if 'Pseudo' in list(Sequences.keys()):
        return Sequences, Background, NewPaths

    # Generate pseudo genes
    # Get the gene lengths
    gen_lens = [Sequences[gene]['Coverage']['0'][()].shape[1]
                for gene in Sequences]

    random_gen = np.random.choice(np.arange(len(gen_lens)), 1,
                                  p=np.array(gen_lens)/np.float(sum(gen_lens)))

    if len(random_gen) == 1:
        gene_name = list(Sequences.keys())[random_gen[0]]
    else:
        gene_name = list(Sequences.keys())[random_gen]

    Sequences['Pseudo'] = Sequences[gene_name]
    Background['Pseudo'] = Background[gene_name]

    zero_states = [i for i in range(len(PriorMatrix))]

    new_path = np.random.choice(zero_states, size=NewPaths[gene_name].shape,
                                replace=True)
    NewPaths['Pseudo'] = new_path

    return Sequences, Background, NewPaths


def generateDB(args):
    """Given a GFF file, launches to CreateDB function.

    By default, omniCLIP only considers entries with a 'gene' feature. To
    include other features, as needed when using an Ensembl GFF3 for example,
    one needs to specify them at the command line.

    Example : --gene-features=gene,ncRNA
    """
    if args.gene_features is None:
        gene_features = ['gene']
    else:
        gene_features = args.gene_features[0].split(',')

    # Creating the DB
    CreateGeneAnnotDB.CreateDB(args.gff_file, args.db_file, gene_features)


def parsingBG(args):
    """Parse the background (BG) BAM files."""
    GeneAnnotation = gffutils.FeatureDB(args.db_file, keep_order=True)
    LoadReads.load_data(
        bam_files=args.bg_libs,
        genome_dir=args.genome_dir,
        gene_annotation=GeneAnnotation,
        out_file=args.out_file,
        Collapse=args.collapsed,
        OnlyCoverage=args.only_coverage,
        mask_flank_variants=args.mask_flank_variants,
        max_mm=args.max_mm,
        ign_out_rds=args.ign_out_rds,
        rev_strand=args.rev_strand
    )


def parsingCLIP(args):
    """Parse the CLIP (BG) BAM files."""
    GeneAnnotation = gffutils.FeatureDB(args.db_file, keep_order=True)
    LoadReads.load_data(
        bam_files=args.clip_libs,
        genome_dir=args.genome_dir,
        gene_annotation=GeneAnnotation,
        out_file=args.out_file,
        Collapse=args.collapsed,
        mask_flank_variants=args.mask_flank_variants,
        max_mm=args.max_mm,
        ign_out_rds=args.ign_out_rds,
        rev_strand=args.rev_strand,
        CLIP_exp=True,
        min_coverage=int(args.min_coverage),
        min_peak=int(args.min_peak)
    )

    Sequences = LoadReads.get_data_handle(args.out_file, write=True)

    if args.mask_miRNA:
        print('Removing miRNA-coverage')
        ParsingPositions.mask_miRNA_positions(Sequences, GeneAnnotation)

    if args.mask_ovrlp:
        print('Masking overlapping positions')
        ParsingPositions.mask_overlapping_positions(Sequences, GeneAnnotation)


# if __name__ == '__main__':
def main():

    parser = argparse.ArgumentParser(prog='PROG', description='omniCLIP - probabilistic identification of protein-RNA interactions from CLIP-seq data')
    subparsers = parser.add_subparsers(title='subcommands', help='sub-command help', dest='command')

    # Create the parser for the generateDB command
    parser_generateDB = subparsers.add_parser('generateDB', help='generateDB help', description="Preprocessing of a GFF annotation file into an SQL database.")
    parser_generateDB_reqNamed = parser_generateDB.add_argument_group('required arguments')
    parser_generateDB_reqNamed.add_argument('--gff-file', dest='gff_file', help='Path to the .GFF annotation file', required=True)
    parser_generateDB_reqNamed.add_argument('--db-file', dest='db_file', help='Path to the output .GFF.DB file', required=True)
    # Optional args for the generateDB command
    parser_generateDB.add_argument('--gene_features', nargs='+', dest='gene_features', help='Features in the annotation file to be considered as genes', required=False)

    # Shared optional arguments from parsingBG and parsingCLIP command
    parent_parsing = argparse.ArgumentParser(add_help=False)
    parent_parsing.add_argument('--rev_strand', action='store', dest='rev_strand', choices=[0, 1], help='Only consider reads on the forward (0) or reverse strand (1) relative to the gene orientation', type=int, default=None)
    parent_parsing.add_argument('--collapsed', action='store_true', default=False, dest='collapsed', help='Reads are collapsed')
    parent_parsing.add_argument('--ign_out_rds', action='store_true', dest='ign_out_rds', help='ignore reads where the ends map ouside of the genes', default=False)
    parent_parsing.add_argument('--max-mismatch', action='store', dest='max_mm', help='Maximal number of mismatches that is allowed per read (default: 2)', type=int, default=2)
    parent_parsing.add_argument('--mask_flank_mm', action='store', dest='mask_flank_variants', help='Do not consider mismatches in the N bp at the ends of reads for diagnostic event modelling (default: 3)', type=int, default=3)

    # Create the parser for the parsingBG command
    parser_parsingBG = subparsers.add_parser('parsingBG', help='parsingBG help', description="Parsing the background files.", parents=[parent_parsing])
    parser_parsingBG_reqNamed = parser_parsingBG.add_argument_group('required arguments')
    parser_parsingBG_reqNamed.add_argument('--bg-files', action='append', dest='bg_libs', help='BAM files for background libraries', required=True)
    parser_parsingBG_reqNamed.add_argument('--db-file', action='store', dest='db_file', help='Path to the .GFF.DB file', required=True)
    parser_parsingBG_reqNamed.add_argument('--out-file', action='store', dest='out_file', help='Output path for .dat file', required=True)
    parser_parsingBG_reqNamed.add_argument('--genome-dir', action='store', dest='genome_dir', help='Directory where fasta files are stored')
    # Optional args for the parsingBG command
    parser_parsingBG.add_argument('--bck-var', action='store_false', default=True, dest='only_coverage', help='Parse variants for background reads')

    # Create the parser for the parsingCLIP command
    parser_parsingCLIP = subparsers.add_parser('parsingCLIP', help='parsingCLIP help', description="Parsing the CLIP files.", parents=[parent_parsing])
    parser_parsingCLIP_reqNamed = parser_parsingCLIP.add_argument_group('required arguments')
    parser_parsingCLIP_reqNamed.add_argument('--clip-files', action='append', dest='clip_libs', help='BAM files for CLIP libraries', required=True)
    parser_parsingCLIP_reqNamed.add_argument('--db-file', action='store', dest='db_file', help='Path to the .GFF.DB file', required=True)
    parser_parsingCLIP_reqNamed.add_argument('--out-file', action='store', dest='out_file', help='Output path for .dat file', required=True)
    parser_parsingCLIP_reqNamed.add_argument('--genome-dir', action='store', dest='genome_dir', help='Directory where fasta files are stored')
    # Optional args for the parsingCLIP command
    parser_parsingCLIP.add_argument('--mask-miRNA', action='store_true', dest='mask_miRNA', help='Mask miRNA positions', default=False)
    parser_parsingCLIP.add_argument('--mask-ovrlp', action='store_true', dest='mask_ovrlp', help='Ignore overlapping gene regions for diagnostic event model fitting', default=False)
    parser_parsingCLIP.add_argument('--min-coverage', action='store', dest='min_coverage', help='Minimum summed coverage on a whole gene to be considered in the analysis', default=100)
    parser_parsingCLIP.add_argument('--min-peak', action='store', dest='min_peak', help='Minimum maximum peak heigth on a gene to be considered in the analysis', default=5)

    # Create the parser for the run_omniCLIP command
    parser_run_omniCLIP = subparsers.add_parser('run_omniCLIP', help='run_omniCLIP help', description="running the main omniCLIP program.")
    parser_run_omniCLIP_reqNamed = parser_run_omniCLIP.add_argument_group('required arguments')
    parser_run_omniCLIP_reqNamed.add_argument('--bg-dat', action='store', dest='dat_file_bg', help='Path to the parsed background .dat file', required=True)
    parser_run_omniCLIP_reqNamed.add_argument('--clip-dat', action='store', dest='dat_file_clip', help='Path to the parsed CLIP .dat file', required=True)
    parser_run_omniCLIP_reqNamed.add_argument('--out-dir', action='store', dest='out_dir', help='Output directory for results')
    parser_run_omniCLIP_reqNamed.add_argument('--db-file', action='store', dest='gene_anno_file', help='File where gene annotation is stored')
    # Optional args for the run_omniCLIP command
    parser_run_omniCLIP.add_argument('--bg-type', action='store', dest='bg_type', help='Background type', choices=['None', 'Coverage', 'Coverage_bck'], default='Coverage_bck')
    parser_run_omniCLIP.add_argument('--max-it', action='store', dest='max_it', help='Maximal number of iterations', type=int, default=20)
    parser_run_omniCLIP.add_argument('--tol-log-like', action='store', dest='tol_lg_lik', help='tolerance for lok-likelihood', type=float, default=10000.0)
    parser_run_omniCLIP.add_argument('--nr_mix_comp', action='store', dest='nr_mix_comp', help='Number of diagnostic events mixture components', type=int, default=1)
    parser_run_omniCLIP.add_argument('--ign-diag', action='store_true', dest='ign_diag', help='Ignore diagnostic event model for scoring', default=False)
    parser_run_omniCLIP.add_argument('--ign-GLM', action='store_true', dest='ign_GLM', help='Ignore GLM model for scoring', default=False)
    parser_run_omniCLIP.add_argument('--snp-ratio', action='store', dest='snps_thresh', help='Ratio of reads showing the SNP', type=float, default=0.2)
    parser_run_omniCLIP.add_argument('--snp-abs-cov', action='store', dest='snps_min_cov', help='Absolute number of reads covering the SNP position', type=float, default=10)
    parser_run_omniCLIP.add_argument('--norm_class', action='store_true', dest='norm_class', help='Normalize class weights during glm fit', default=False)
    parser_run_omniCLIP.add_argument('--seed', action='store', dest='rnd_seed', help='Set a seed for the random number generators', default=None)
    parser_run_omniCLIP.add_argument('--diag_event_mod', action='store', dest='diag_event_mod', help='Diagnostic event model', choices=['DirchMult', 'DirchMultK'], default='DirchMultK')
    parser_run_omniCLIP.add_argument('--glm_weight', action='store', dest='glm_weight', help='weight of the glm score with respect to the diagnostic event score', type=float, default=-1.0)
    parser_run_omniCLIP.add_argument('--skip_diag_event_mdl', action='store_true', dest='skip_diag_event_mdl', help='Do not model the diagnostic events', default=False)
    parser_run_omniCLIP.add_argument('--pv', action='store', dest='pv_cutoff', help='bonferroni corrected p-value cutoff for peaks in bed-file', type=float, default=0.05)
    parser_run_omniCLIP.add_argument('--emp-var', action='store_true', dest='emp_var', help='use the empirical variance if it larger than the expected variance', default=False)
    parser_run_omniCLIP.add_argument('--diag-bg', action='store_true', dest='diag_bg', help='estimate diagnostic events for the background states on the background', default=False)
    parser_run_omniCLIP.add_argument('--fg_pen', action='store', dest='fg_pen', help='Penalty for fg during scoring', type=float, default=0.0)
    parser_run_omniCLIP.add_argument('--filter-snps', action='store_true', dest='filter_snps', help='Do not fit diagnostic events at SNP-positions', default=False)
    parser_run_omniCLIP.add_argument('--no-subsample', action='store_false', default=True, dest='subs', help='Disable subsampling for parameter estimations (Warning: Leads to slow estimation)')
    parser_run_omniCLIP.add_argument('--ign_out_rds', action='store_true', dest='ign_out_rds', help='ignore reads where the ends map ouside of the genes', default=False)
    # Runtime args for the run_omniCLIP command (using a different subparser?)
    parser_run_omniCLIP.add_argument('--nb-cores', action='store', dest='nb_proc', help='Number of cores o use', type=int, default=1)
    parser_run_omniCLIP.add_argument('--save-tmp', action='store_true', dest='safe_tmp', help='Safe temporary results', default=False)
    parser_run_omniCLIP.add_argument('--tmp-dir', action='store', dest='tmp_dir', help='Output directory for temporary results', default=None)
    parser_run_omniCLIP.add_argument('--verbosity', action='store', dest='verbosity', help='verbosity: 0 (default) gives information of current state of site prediction, 1 gives aditional output on runtime and meomry consupmtiona and 2 shows selected internal variables', type=int, default=0)
    # TODO : need to implement this in the Sequence file
    parser_run_omniCLIP.add_argument('--mask-ovrlp', action='store_true', dest='mask_ovrlp', help='Ignore overlapping gene regions for diagnostic event model fitting', default=False)

    # Parsing the arguments if only sites should be predicted
    args = parser.parse_args()

    if args.command == 'generateDB':
        generateDB(args)
    elif args.command == 'parsingBG':
        parsingBG(args)
    elif args.command == 'parsingCLIP':
        parsingCLIP(args)
    elif args.command == 'run_omniCLIP':
        run_omniCLIP(args)
