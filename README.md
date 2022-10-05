# Tumoroscope model in PyMC

In this repo, I try to reproduce the Tumoroscope model presented by Shafighi *et al.* (2022).

![Tumorosope-overview](tumoroscope-overview.png)

[bioR$\chi$iv preprint](https://www.biorxiv.org/content/10.1101/2022.09.22.508914)

Citation for the preprint:

```
@article {Shafighi2022.09.22.508914,
	author = {Shafighi, Shadi Darvish and Geras, Agnieszka and Jurzysta, Barbara and Naeini, Alireza Sahaf and Filipiuk, Igor and R{\k a}czkowski, {\L}ukasz and Toosi, Hosein and Koperski, {\L}ukasz and Thrane, Kim and Engblom, Camilla and Mold, Jeff and Chen, Xinsong and Hartman, Johan and Nowis, Dominika and Carbone, Alessandra and Lagergren, Jens and Szczurek, Ewa},
	title = {Tumoroscope: a probabilistic model for mapping cancer clones in tumor tissues},
	elocation-id = {2022.09.22.508914},
	year = {2022},
	doi = {10.1101/2022.09.22.508914},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Spatial and genomic heterogeneity of tumors is the key for cancer progression, treatment, and survival. However, a technology for direct mapping the clones in the tumor tissue based on point mutations is lacking. Here, we propose Tumoroscope, the first probabilistic model that accurately infers cancer clones and their high-resolution localization by integrating pathological images, whole exome sequencing, and spatial transcriptomics data. In contrast to previous methods, Tumoroscope explicitly addresses the problem of deconvoluting the proportions of clones in spatial transcriptomics spots. Applied to a reference prostate cancer dataset and a newly generated breast cancer dataset, Tumoroscope reveals spatial patterns of clone colocalization and mutual exclusion in sub-areas of the tumor tissue. We further infer clone-specific gene expression levels and the most highly expressed genes for each clone. In summary, Tumoroscope enables an integrated study of the spatial, genomic, and phenotypic organization of tumors.Competing Interest StatementProjects in Szczurek lab are co-funded by Merck Healthcare. C.E., K.T., and J.M. are scientific consultants for 10x Genomics Inc. Other authors declare that they have no competing interests.},
	URL = {https://www.biorxiv.org/content/early/2022/09/23/2022.09.22.508914},
	eprint = {https://www.biorxiv.org/content/early/2022/09/23/2022.09.22.508914.full.pdf},
	journal = {bioRxiv}
}

```
