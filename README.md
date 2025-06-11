
# NexusGene

This project is a pytorch geometric learning framework for pan-cancer gene identifications. It applies to both homogeneous networks (PPI network)and heterogeneous networks. TREE can provide generalizable robust and accurate cancer gene identifications with multi-omics-level and network structure-level interpretability.


## Installation

The code is written in Python 3 and was mainly tested on Python 3.12

NexusGene has the following dependencies:

-h5py 3.1.0
-numpy 1.19.5
-scikit-learn 0.24.2
-shap 0.41.0
-numpy 1.19.5
-torch_geometric
    
## Datasets
 All datasets used in this study are publicly available. The public data used 
for producing the results in this study can be downloaded from Zenodo 
at https://doi.org/10.5281/zenodo.11648891 (ref. 80). The cancer-specific 
homogeneous networks are available in Zenodo at https://doi.
 org/10.5281/zenodo.11648365 (ref. 81). The cancer-specific hetero
geneous networks are available in Zenodo at https://doi.org/10.5281/
 zenodo.11648733 (ref. 82). Source data are provided with this paper.
## Reproducibility
you can train NexusGene by:
```bash
  python main.py
```