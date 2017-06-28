import os
import numpy as np
from collections import defaultdict
import pdb


def load_ground_truth(file_name_motif):
	'''
	This file load the ground truth from the FIMO motif file
	'''
	fid = open(file_name_motif, 'r')
	motifs = {}
	for line in fid.readlines():
		line = line.strip().split('\t')
		transcript_name = line[1].split('|')[0]
		start = int(line[2])
		stop = int(line[3])
		motifs[transcript_name] = [start, stop]
		#TODO covert to genomic coordinates
	return motifs


def filter_expressed_motifs(motifs):
	'''
	This function returns only the motifs that are in expressed genes
	'''
	return motifs


def evaluate_predictions(motifs, predictions, measure = 'overlap'):
	'''
	This function evaluates the Motif Predictions
	'''
	nr_motifs = 0
	nr_pred = 0
	nr_of_overlap = 0
	nr_pos_motifs = 0
	nr_of_pos_pred = 0
	nr_of_overlap_pred = 0

	#Measure the overlap
	if measure == 'overlap':
		genes = list(set(motifs.keys() + predictions.keys()))
		for gene in genes:
			curr_motifs = motifs[gene]
			curr_pred = predictions[gene]
			nr_motifs += len(curr_motifs)
			nr_pred += len(curr_pred)
			nr_pos_motifs += sum([site[1] - site[0] for site in curr_motifs])
			nr_of_pos_pred += sum([site[1] - site[0] for site in curr_pred])
			if min(len(curr_motifs), len(curr_pred)) == 0:
				continue
				
			#check for each motif whether it overlaps a prediction
			for pred in curr_pred:
				overlaping_mot = [mot for mot in curr_motifs if (pred[0] <= mot[1] and pred[1] >= mot[0])] 
				nr_of_overlap += min(1, len(overlaping_mot))
				nr_of_overlap_pred += sum([max(pred[1] - mot[0], mot[1] - pred[0]) for mot in overlaping_mot])
	else:
		pass 

	return [nr_pred, nr_of_overlap, nr_pos_motifs, nr_of_pos_pred, nr_of_overlap_pred]


def convert_paths_to_sites(Paths, fg_state, merge_neighbouring_sites, minimal_site_length):
	'''
	This function takes the paths and computes the site predictions (defined by the state fg_state)
	'''
	sites = defaultdict(list)

	#Iterate over the paths
	for gene in Paths:
		#first_pos = -1
		#end_pos = -1
		#curr_dist_counter = merge_neighbouring_sites
		curr_path = Paths[gene] == fg_state
		Starts = np.where(np.concatenate(([curr_path[0]], curr_path[:-1] != curr_path[1:], [curr_path[-1]])))[0][::2]
		Stops = np.where(np.concatenate(([curr_path[0]], curr_path[:-1] != curr_path[1:], [curr_path[-1]])))[0][1::2]
		nr_sites = Starts.shape[0]
		sites[gene] = [[Starts[i], Stops[i]] for i in xrange(nr_sites) if (Stops[i] - Starts[i] >= minimal_site_length)]

	return sites


	def run_evaluation():
		'''
		This function takes the paths and writes the predictions
		'''
		

