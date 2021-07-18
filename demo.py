import pandas as pd
import numpy as np
import scipy as sp


# download from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE120861
# reference: Gasperini M, Hill AJ, McFaline-Figueroa JL, Martin B et al. A Genome-wide Framework for Mapping Gene Regulation via Cellular Genetic Screens. Cell 2019 Jan 10;176(1-2):377-390.e19. PMID: 30612741


# load

base = '/home/jjpeng/data/yxlong/CRISPR/source/30849375/GEO/'

## load cell phenotypes

phenoData = pd.read_csv(base + 'GSE120861_at_scale_screen.phenoData.txt.gz', sep=' ', header=None, usecols=[0,1,2,3,5,7,8,9,10,14,17])
phenoData.columns = ['sample','cell','total_umis','Size_Factor','gRNA_groups','read_count','umi_count','proportion','guide_count','prep_batch','percent.mito']
phenoData.set_index('cell', inplace=True)

phenoData['guide_count'].fillna(0, inplace=True)
phenoData['prep_batch'] = phenoData.eval('prep_batch != "prep_batch_1"').astype(int)

## parse & save 10X expression matrix

cells = pd.read_csv(base + 'GSE120861_at_scale_screen.cells.txt.gz', sep = '\t', header=None)[0]
genes = pd.read_csv(base + 'GSE120861_at_scale_screen.genes.txt.gz', sep = '\t', header=None)[0]

# mtx = pd.read_csv(base + 'GSE120861_at_scale_screen.exprs.mtx.gz', sep = ' ', header=None, skiprows=2, compression=None)
# coo = sp.sparse.coo_matrix((mtx[2], (mtx[0] - 1, mtx[1] - 1)), shape=(genes.shape[0], cells.shape[0]), dtype=np.int16)
# np.save(base + 'at_scale.exprs.npy', coo)

## load expression matrix

coo = np.load(base + 'at_scale.exprs.npy', allow_pickle=True).item()
exprs = pd.DataFrame.sparse.from_spmatrix(coo.T, index=cells, columns=genes)


# prepare

## x

# select 4 gRNA groups targeting non-coding regions, 2 positiv control gRNA groups and a non-targeting control gRNA
gRNA_groups = ['chr1.8492_top_two','chr1.8492_second_two','chrX.952_top_two','chrX.952_second_two','CERS2_TSS','GATA2_TSS','bassik_mch']

X, xnames = list(), list()

for gRNA_group in gRNA_groups:
  
  geno = phenoData['gRNA_groups'].str.contains(gRNA_group, na=False) # get genotypes
  
  x = geno.astype(int).values.reshape((-1, 1)) # use genotype as x
  cov = phenoData[['guide_count','percent.mito','prep_batch']].values # covariants
  c = np.ones((x.shape[0], 1)) # constraint

  X.append(np.concatenate([x, cov, c], axis=1))
  xnames.append((gRNA_group, ['gRNA_group', 'guide_count', 'percent.mito', 'prep_batch', 'intercept']))
  
## y

size_factor = phenoData['Size_Factor'].values # load size factor calculate by monocle

loader = lambda i: np.round(exprs[exprs.columns[i]].values.T / size_factor)


# export

args = {
  'X': X,
  'Y': {'data': loader, 'length': exprs.shape[1]},
  'family': 'negative_binomial',
  'xnames': xnames,
  'yname': exprs.columns,
  'kwargs': {}
}


if __name__ == '__main__':
  
    import scpy
    
    args['Y'] = scpy.dataloader(**args['Y'])
    
    summaries = scpy.fit_batch(device='cpu', disp=True, **args)
    
    print(summaries)
  
