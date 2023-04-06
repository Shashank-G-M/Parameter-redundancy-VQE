# Parameter-redundancy-VQE

The repository contains codes and data used in the article [Parameter Redundancy in the Unitary Coupled-Cluster Ansatze for Hybrid Variational Quantum Computing](https://arxiv.org/pdf/2301.09825.pdf). If you are using the code available in this repository, kindly site the corresponding article mentioned here.

The directory "Data" contains all the data needed to genarate the figures given in the article. It has been divided into subdirectories "ML Data" and "Pre ML Data" which contain the data required to generate figures corresponding to ML algorithm and Pre ML (or non ML) algorithm respectively. These subdirectories have been further divided according to the figure indices, and they inturn contain the data required to generate those figures. All the raw data is in the format of .pkl which when loaded into python will give list type objects. To understand the exact structure of these data files, please refer to the README files present in those subdirectories.

The file "small_amplitude_filtration.ipynb" is the code used in the article to perform small amplitude filtration and spin adaption. The file contains further directions regarding its usage.
