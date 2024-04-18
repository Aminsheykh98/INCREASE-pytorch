# INCREASE-pytorch

Unofficial code of paper 'INCREASE: Inductive Graph Representation Learning for Spatio-Temporal Kriging'[^1] in Pytorch.

---

**Abstract**:
> Spatio-temporal kriging is an important problem in web and social applications, such as Web or Internet of Things, where things (e.g., sensors) connected into a web often come with spatial and temporal properties. It aims to infer knowledge for (the things at) unobserved locations using the data from (the things at) observed locations during a given time period of interest. This problem essentially requires inductive learning. Once trained, the model should be able to perform kriging for different locations including newly given ones, without retraining. However, it is challenging to perform accurate kriging results because of the heterogeneous spatial relations and diverse temporal patterns. In this paper, we propose a novel inductive graph representation learning model for spatio-temporal kriging. We first encode heterogeneous spatial relations between the unobserved and observed locations by their spatial proximity, functional similarity, and transition probability. Based on each relation, we accurately aggregate the information of most correlated observed locations to produce inductive representations for the unobserved locations, by jointly modeling their similarities and differences. Then, we design relation-aware gated recurrent unit (GRU) networks to adaptively capture the temporal correlations in the generated sequence representations for each relation. Finally, we propose a multi-relation attention mechanism to dynamically fuse the complex spatio-temporal information at different time steps from multiple relations to compute the kriging output. Experimental results on three real-world datasets show that our proposed model outperforms state-of-the-art methods consistently, and the advantage is more significant when there are fewer observed locations. 

---

## Requirements:
+ torch
+ numpy
+ pandas
+ h5py
+ argparse

---

## Datasets:
METR-LA: The traffic data files for Los Angeles (METR-LA), i.e., metr-la.h5, are available at [Google Drive](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX), and should be put into the dataset/metr-la/ folder.

BEIJING: Right now this dataset is not supported.

---

## Files

- `load_metrla.py`: Loads and preprocesses the metr-la dataset.
- `parser.py`: Argparser file which contains necessary parameters and hyperparameters.
- `model.py`: File that contains model architecture.
- `utils.py`: Contains mse_loss and metric functions.
- `run.py`: Trains and tests the INCREASE on metr-la dataset.

---

## Acknowledgment:
The code is the pytorch version of the official code [INCREASE](https://github.com/zhengchuanpan/INCREASE/tree/main).

---


> [!NOTE]
> This repository is the translation of the source code (which is in Tensorflow) to Pytorch. This code is not the stable version, and will collapse after 5 epochs as it is mentioned [here](https://github.com/zhengchuanpan/INCREASE/issues/2).


[^1]: Zheng, Chuanpan, et al. "Increase: Inductive graph representation learning for spatio-temporal kriging." Proceedings of the ACM Web Conference 2023. 2023.

