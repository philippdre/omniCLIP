import numpy as np
from Bio import SeqIO
import os
import re
import sys
sys.path.append('./CFG/')
sys.path.append('./DataParsing/')
sys.path.append('./Utils/')
sys.path.append('./Analysis/')
sys.path.append('./Model/')
from scipy.sparse import *
import cPickle
import gzip
from collections import defaultdict
import pysam
import GetCoverageFromBam
import pdb

def load_data(bam_files, genome_dir, gene_annotation, out_file, load_from_file = False, save_results = True, Collapse = False, OnlyCoverage = False, select_chrom = None, store_gene_seq=False, mask_flank_variants=3, max_mm=2):
	'''
	This function reads the data from the bam-files
	'''
	if load_from_file:
		with gzip.GzipFile(out_file, 'r') as f:
			GeneConversionEvents = cPickle.load(f)
		return GeneConversionEvents
	else:
		GeneConversionEvents = defaultdict(dict)
		if OnlyCoverage:
			print('Loading coverage only')

		print "Parsing the gene annotation"
		Iter = gene_annotation.features_of_type('gene')
		Genes = []
		for gene in Iter:
			Genes.append(gene)

		#Create a dictionary of genes by chromosome
		genes_chr_dict = defaultdict(list)
		for gene in Genes:
			genes_chr_dict[gene.chrom].append(gene)

		#Create a list of cromosomes
		Chrs = genes_chr_dict.keys()

		Chrs = [chrom for chrom in Chrs if chrom.lower() != 'chrm']

		for bam_file in bam_files:
			pass

		#3. Iterate over the CurrChromosomes 
		for CurrChr in Chrs:
			if select_chrom != None:
				if select_chrom != CurrChr:
					continue
			#Make sure that the CurrChr is indeed a CurrChromosome and not header information
			print 'Processing ' + CurrChr
			#Get the genome
			CurrChrFile  = os.path.join(genome_dir, CurrChr + '.fa.gz')
			if not os.path.isfile(CurrChrFile):
				print 'Warning, chromosome not found: ' + CurrChrFile
				continue
			handle = gzip.open(CurrChrFile, "rU")
			record_dict = SeqIO.to_dict(SeqIO.parse(handle, "fasta"))
			CurrChrSeq = record_dict[CurrChr]
			handle.close()

			for i, bam_file in enumerate(bam_files):
				#Iterate over the Genes
				SamReader = pysam.Samfile(bam_file,'rb')
				for gene in Genes:
					Variants = {}
					ReadEnds = {}
					Coverage = {}
					if gene.chrom == CurrChr:
						Start = gene.start
						Stop = gene.stop 
						Strand =  1 if gene.strand == '+' else -1
						#get the data
						Coverage = GetCoverageFromBam.GetRawCoverageFromRegion(SamReader, CurrChr, Start, Stop, Collapse = Collapse, CovType = 'coverage', Genome = '', legacy = False, mask_flank_variants=mask_flank_variants, max_mm=max_mm)
						if not OnlyCoverage:
							Variants = GetCoverageFromBam.GetRawCoverageFromRegion(SamReader, CurrChr, Start, Stop, Collapse = Collapse, CovType = 'variants', Genome = '', legacy = False, mask_flank_variants=mask_flank_variants, max_mm=max_mm)
							ReadEnds = GetCoverageFromBam.GetRawCoverageFromRegion(SamReader, CurrChr, Start, Stop, Collapse = Collapse, CovType = 'read-ends', Genome = '', legacy = False, mask_flank_variants=mask_flank_variants, max_mm=max_mm)
							Coverage = Coverage - np.sum(Variants, axis = 0) - ReadEnds

							#Get the TC conversions
							CurrSeq = str(CurrChrSeq.seq[Start : Stop])
							GeneSeq = np.zeros((1, len(CurrSeq)), dtype=np.uint8)
							
							Ix = [m.start() for m in re.finditer('C', CurrSeq.upper())]
							GeneSeq[0, Ix] = 1
							del Ix
							Ix = [m.start() for m in re.finditer('G', CurrSeq.upper())]
							GeneSeq[0, Ix] = 2
							del Ix
							Ix = [m.start() for m in re.finditer('T', CurrSeq.upper())]
							GeneSeq[0, Ix] = 3

							del Ix

							if not GeneConversionEvents[gene.id.split('.')[0]].has_key('Coverage'):
								if store_gene_seq:
									GeneConversionEvents[gene.id.split('.')[0]]['GeneSeq'] = {}
								GeneConversionEvents[gene.id.split('.')[0]]['Variants'] = {}
								GeneConversionEvents[gene.id.split('.')[0]]['Coverage'] = {}
								GeneConversionEvents[gene.id.split('.')[0]]['Read-ends'] = {}

							non_zer_var = np.where(Variants)
							ij = np.vstack((GeneSeq[0, non_zer_var[1]] * 5 + non_zer_var[0], non_zer_var[1]))
							SparseVarMat = csr_matrix((Variants[np.where(Variants)], ij), shape=(20,len(CurrSeq)))
							if store_gene_seq:
								GeneConversionEvents[gene.id.split('.')[0]]['GeneSeq'][i] = GeneSeq
							GeneConversionEvents[gene.id.split('.')[0]]['Variants'][i] = SparseVarMat
							#Only save the positions where a read in one of the replicates occured
							GeneConversionEvents[gene.id.split('.')[0]]['Read-ends'][i] = csr_matrix(ReadEnds)
							del Variants, ReadEnds,  non_zer_var, ij, SparseVarMat, CurrSeq, GeneSeq
						else:
							if not GeneConversionEvents[gene.id.split('.')[0]].has_key('Coverage'):
								GeneConversionEvents[gene.id.split('.')[0]]['Coverage'] = {}
						GeneConversionEvents[gene.id.split('.')[0]]['strand'] = Strand
						GeneConversionEvents[gene.id.split('.')[0]]['Coverage'][i] = csr_matrix(Coverage)
						del Coverage

			del CurrChrSeq
			del record_dict
			
		print 'Saving results'
		del Genes
		if save_results:
			with gzip.GzipFile(out_file, 'w') as f:
					cPickle.dump(GeneConversionEvents, f, protocol=-1)
	return GeneConversionEvents



