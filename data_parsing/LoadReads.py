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
sys.path.append('./Analysis/')
sys.path.append('./CFG/')
sys.path.append('./DataParsing/')
sys.path.append('./Model/')
sys.path.append('./Utils/')
from Bio import SeqIO
from collections import defaultdict
from scipy.sparse import *
import GetCoverageFromBam
import gzip
import h5py
import numpy as np
import os
import pysam


##@profile
#@profile 
def load_data(bam_files, genome_dir, gene_annotation, out_file, load_from_file = False, save_results = True, Collapse = False, OnlyCoverage = False, select_chrom = None, store_gene_seq=False, mask_flank_variants=3, max_mm=2, ign_out_rds=False, rev_strand=None):
	'''
	This function reads the data from the bam-files
	'''
	
	if load_from_file:
		GeneConversionEvents = h5py.File(out_file, 'r+')
		return GeneConversionEvents
	else:
		if save_results:
			GeneConversionEvents = h5py.File(out_file, 'w')

		if OnlyCoverage:
			print ('Loading coverage only')

		print("Parsing the gene annotation")
		Iter = gene_annotation.features_of_type('gene')
		Genes = []
		for gene in Iter:
			Genes.append(gene)

		#Create a dictionary of genes by chromosome
		genes_chr_dict = defaultdict(list)
		for gene in Genes:
			genes_chr_dict[gene.chrom].append(gene)

		#Create a list of cromosomes
		Chrs = list(genes_chr_dict.keys())

		Chrs = [chrom for chrom in Chrs if chrom.lower() != 'chrm']

		for bam_file in bam_files:
			pass

		#3. Iterate over the CurrChromosomes 
		for CurrChr in Chrs:
			if select_chrom != None:
				if select_chrom != CurrChr:
					continue
			#Make sure that the CurrChr is indeed a CurrChromosome and not header information
			print('Processing ' + CurrChr)
			#Get the genome
			CurrChrFile  = os.path.join(genome_dir, CurrChr + '.fa.gz')
			if not os.path.isfile(CurrChrFile):
				print('Warning, chromosome not found: ' + CurrChrFile)
				continue
			handle = gzip.open(CurrChrFile, "rt" )
			record_dict = SeqIO.to_dict(SeqIO.parse(handle, "fasta"))
			CurrChrSeq = record_dict[CurrChr]
			handle.close()

			for i, bam_file in enumerate(bam_files):
				#Iterate over the Genes
				SamReader = pysam.Samfile(bam_file,'rb')
				for gene in Genes:
					if gene.id.count('_PAR_Y') > 0:
						continue
					Variants = {}
					ReadEnds = {}
					Coverage = {}
					if gene.chrom == CurrChr:
						Start = gene.start
						Stop = gene.stop 
						new_gene_name = gene.id.split('.')[0]
						Strand =  1 if gene.strand == '+' else -1
						#get the data
						CovType = ['coverage']
						if not OnlyCoverage:
							CovType += ['variants', 'read-ends']

						ret_arrays = GetCoverageFromBam.GetRawCoverageFromRegion(SamReader, CurrChr, Start, Stop, Collapse = Collapse, CovType = CovType, Genome = '', legacy = False, mask_flank_variants=mask_flank_variants, max_mm=max_mm, ign_out_rds=ign_out_rds, gene_strand=Strand, rev_strand=rev_strand)
						Coverage = ret_arrays['coverage']
						#Coverage = GetCoverageFromBam.GetRawCoverageFromRegion(SamReader, CurrChr, Start, Stop, Collapse = Collapse, CovType = 'coverage', Genome = '', legacy = False, mask_flank_variants=mask_flank_variants, max_mm=max_mm, ign_out_rds=ign_out_rds, gene_strand=Strand, rev_strand=rev_strand)
						if not new_gene_name in GeneConversionEvents:
							GeneConversionEvents.create_group(new_gene_name)
						if not 'SummedCoverage' in GeneConversionEvents[new_gene_name]:
							GeneConversionEvents[new_gene_name].create_group('SummedCoverage')

						#Save the total coverage
						GeneConversionEvents[new_gene_name]['SummedCoverage'].create_dataset(str(i), data=Coverage, compression="gzip", compression_opts=9, chunks=Coverage.shape)
						
						if not OnlyCoverage:
							Variants = ret_arrays['variants']
							ReadEnds = ret_arrays['read-ends']

							Coverage = Coverage - np.sum(Variants, axis=0) - np.sum(ReadEnds, axis=0)

							#Get the TC conversions
							CurrSeq = str(CurrChrSeq.seq[Start : Stop]).upper()
							GeneSeq = np.zeros((1, len(CurrSeq)), dtype=np.uint8)
							
							trdict = {"A":0,"C":1,"G":2,"T":3, "N":4}
							GeneSeq[0, :] = np.array([trdict[e] for e in list(CurrSeq)])

							#Check that group already exists, otherwise create it
							if not new_gene_name in GeneConversionEvents:
								GeneConversionEvents.create_group(new_gene_name)

							if not 'Coverage' in GeneConversionEvents[new_gene_name]:
								if store_gene_seq and (str(i) == '0'):
									GeneConversionEvents[new_gene_name].create_group('GeneSeq')
								GeneConversionEvents[new_gene_name].create_group('Variants')
								GeneConversionEvents[new_gene_name].create_group('Coverage')
								GeneConversionEvents[new_gene_name].create_group('Read-ends')
							
							#Ignore Variants at N positions
							Variants[:, GeneSeq[0, :] == 4] = 0 
							Variants[Variants < 0] = 0

							non_zer_var = np.where(Variants)
							ij = np.vstack((GeneSeq[0, non_zer_var[1]] * 5 + non_zer_var[0], non_zer_var[1]))
							Variants_sparse = csr_matrix((Variants[np.where(Variants)], ij), shape=(20,len(CurrSeq)))
							Variants = Variants_sparse.toarray()
							#pdb.set_trace()
							del non_zer_var, ij

							if store_gene_seq:
								GeneConversionEvents[new_gene_name]['GeneSeq'].create_dataset(str(i), data=GeneSeq, compression="gzip", compression_opts=9, )#, chunks=GeneSeq.shape)
							#GeneConversionEvents[new_gene_name]['Variants'].create_dataset(str(i), data=Variants, compression="gzip", compression_opts=9)#, chunks=Variants.shape)
							GeneConversionEvents[new_gene_name]['Variants'].create_group(str(i))
							GeneConversionEvents[new_gene_name]['Variants'][str(i)].create_dataset('data', data=Variants_sparse.data)
							GeneConversionEvents[new_gene_name]['Variants'][str(i)].create_dataset('indptr', data=Variants_sparse.indptr)
							GeneConversionEvents[new_gene_name]['Variants'][str(i)].create_dataset('indices', data=Variants_sparse.indices)
							GeneConversionEvents[new_gene_name]['Variants'][str(i)].create_dataset('shape', data=Variants.shape)

							#Only save the positions where a read in one of the replicates occured
							GeneConversionEvents[new_gene_name]['Read-ends'].create_dataset(str(i), data=ReadEnds, compression="gzip", compression_opts=9)#, chunks=ReadEnds.shape)


							del Variants, ReadEnds, CurrSeq, GeneSeq, Variants_sparse

						else:
							if not 'Coverage' in GeneConversionEvents[new_gene_name]:								
								GeneConversionEvents[new_gene_name].create_group('Coverage')
						GeneConversionEvents[new_gene_name]['Coverage'].create_dataset(str(i), data=Coverage, compression="gzip", compression_opts=9)#, chunks=Coverage.shape)
						
						if str(i)=='0':
							GeneConversionEvents[new_gene_name].create_dataset('strand', data=Strand)
						del Coverage

			del CurrChrSeq
			del record_dict
			
		print('Saving results')
		GeneConversionEvents.close()
		GeneConversionEvents = h5py.File(out_file, 'r+')
	return GeneConversionEvents



