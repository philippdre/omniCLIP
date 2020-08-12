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
import os
import random
import shutil

from omniCLIP.data_parsing import LoadReads
from omniCLIP.data_parsing import tools


def parsing_argparse(args):
    """Parse the argparse dict."""
    # Listing args to extract from the argparse parser
    named_args = [
        'bg_type', 'dat_file_bg', 'dat_file_clip', 'diag_bg', 'diag_event_mod',
        'emp_var', 'fg_pen', 'filter_snps', 'glm_weight', 'ign_diag',
        'ign_GLM', 'ign_out_rds', 'mask_ovrlp', 'max_it', 'nb_proc',
        'norm_class', 'out_dir', 'pv_cutoff', 'skip_diag_event_mdl',
        'snps_min_cov', 'snps_thresh', 'subs', 'tmp_dir', 'verbosity'
    ]

    # Creating the params dictionary
    params = {arg: getattr(args, arg) for arg in named_args}

    # Verifying the validity of the args
    verifying_args(params)

    # Printing args if verbosity
    if params['verbosity'] > 1:
        print(args)

    # Defining fixed-value params
    params['ExpressionParameters'] = [None, None]
    params['LastIter'] = False
    params['TransitionType'] = 'binary'

    # Conditional params - Number of States
    if params['bg_type'] == 'Coverage_bck':
        params['NrOfStates'] = 4
    else:
        params['NrOfStates'] = 3

    # Conditional params - Ignore reads outside gene and diag model
    if params['ign_out_rds']:
        params['ign_diag'] = params['ign_out_rds']

    # Conditional params - Prior Matrix
    params['PriorMatrix'] = (np.ones((params['NrOfStates'], 1))
                             / float(params['NrOfStates']))

    # Conditional  params - Transition Matrix
    TransMat = (np.ones((params['NrOfStates'], params['NrOfStates']))
                + np.eye(params['NrOfStates']))
    params['TransMat'] = TransMat / np.sum(np.sum(TransMat))

    # Conditional params - Diag Event Params, flag for penalized path pred
    params['Diag_event_params'] = {}
    params['Diag_event_params']['nr_mix_comp'] = args.nr_mix_comp
    params['Diag_event_params']['alpha'] = {}
    params['Diag_event_params']['mix_comp'] = {}
    for state in range(params['NrOfStates']):
        mixtures = np.random.uniform(0.0, 1.0, size=(args.nr_mix_comp))
        params['Diag_event_params']['mix_comp'][state] = (mixtures
                                                          / np.sum(mixtures))

    # Conditional params - out file name
    params['out_file_base'] = 'pred'
    if params['ign_GLM']:
        params['out_file_base'] += '_no_glm'
    if params['ign_diag']:
        params['out_file_base'] += '_no_diag'

    # Optional params - Random seeding
    if args.rnd_seed is not None:
        random.seed(args.rnd_seed)
        print('setting random seed')

    return params


def dup_seqfiles(params):
    """Create temporary Seqfiles that will be modified by omniCLIP.

    If no tmp_dir has been specified, both tmp files will be created in the
    same location as the CLIP dat file.
    """
    if params['tmp_dir'] is None:
        dir_path = os.path.dirname(os.path.realpath(params['dat_file_clip']))
        fg_tmp = params['dat_file_clip'] + '.tmp'
        bg_tmp = os.path.join(dir_path, 'bg_data.dat.tmp')
    else:
        fg_tmp = os.path.join(params['tmp_dir'], 'clip_data.dat.tmp')
        bg_tmp = os.path.join(params['tmp_dir'], 'bg_data.dat.tmp')

    shutil.copy(params['dat_file_clip'], fg_tmp)
    shutil.copy(params['dat_file_bg'], bg_tmp)

    # Open the temporary read files
    params['dat_file_clip'] = fg_tmp
    params['dat_file_bg'] = bg_tmp

    return params


def parsing_files(args, params):
    """Parse arguments that are function of the data files."""
    # Loading Sequence and Background
    Sequences = LoadReads.get_data_handle(params['dat_file_clip'], write=True)
    Background = LoadReads.get_data_handle(params['dat_file_bg'], write=True)

    # Estimate the library size
    params['LibrarySize'] = tools.estimate_library_size(Sequences)
    params['BckLibrarySize'] = tools.estimate_library_size(Background)

    # Estimate to number of replicates
    gene = list(Sequences.keys())[0]
    params['NrOfReplicates'] = len(Sequences[gene]['Coverage'])
    params['NrOfBckReplicates'] = len(Background[gene]['Coverage'])

    # Initialise the parameter vector alpha
    alphashape = (Sequences[gene]['Variants']['0']['shape'][0]
                  + Sequences[gene]['Coverage']['0'][()].shape[0]
                  + Sequences[gene]['Read-ends']['0'][()].shape[0])
    for state in range(params['NrOfStates']):
        params['Diag_event_params']['alpha'][state] = np.random.uniform(
            0.9, 1.1, size=(alphashape, args.nr_mix_comp))

    return params


def verifying_args(params):
    """Testing the arguments.

    TODO: Implement more testing."""
    pass
