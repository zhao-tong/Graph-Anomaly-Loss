## Error-Bounded Graph Anomaly Loss for GNNs

This repository contains the code package for the CIKM'20 paper:

**[Error-Bounded Graph Anomaly Loss for GNNs](https://dl.acm.org/doi/pdf/10.1145/3340531.3411979).**

#### Authors: Tong Zhao (tzhao2@nd.edu), Chuchen Deng, Kaifeng Yu, Tianwen Jiang, Daheng Wang and Meng Jiang.

## Usage
### 1. Dependencies
This code package was developed and tested with Python 3.6.8 and [PyTorch 1.0.1](https://pytorch.org/).
A detailed dependencies list can be found in `requirements.txt` and can be installed by:
```
pip install -r requirements.txt
```

### 2. Data
Data files are located at `/data/[dataset]/`, a simple example of loading the data can be found [here](https://github.com/zhao-tong/Graph-Anomaly-Loss/blob/master/src/dataCenter.py#L220). Specifically, `[dataset]_graph_u2p.pkl` is the pickled sparse adjacency matrix ([csr_matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html)) and `[dataset]_labels_u.pkl` is the pickled user labels. 

### 3. Run
To train the model, run
```
python -m src.main
```
list of arguments can be found at `/src/main.py`.


## Cite
If you find this repository useful in your research, please cite our paper:

```bibtex
@inproceedings{zhao2020error,
  title={Error-Bounded Graph Anomaly Loss for GNNs},
  author={Zhao, Tong and Deng, Chuchen and Yu, Kaifeng and Jiang, Tianwen and Wang, Daheng and Jiang, Meng},
  booktitle={Proceedings of the 29th ACM International Conference on Information \& Knowledge Management},
  pages={1873--1882},
  year={2020}
}
```

