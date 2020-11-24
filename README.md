### Privacy-of-Distributed-Fair-Learning-Framework
Matlab implementation of "Inference Attack and Defense on the Distributed Private Fair Machine Learning Framework" (PPAI 2019) [[paper]](https://www2.isye.gatech.edu/~fferdinando3/cfp/PPAI20/papers/paper_26.pdf).

### Abstract
Fairness and privacy are both significant social norms in machine learning. In (Hu et al 2019), we propose a distributed framework to learn fair prediction models while protecting
the privacy of user demographics. However, we did not assume an adversary who tries to infer the hidden demographics, e.g., with a good intention of building fairer models.
In this paper, we examine vulnerability of the above framework under inference attack and two defense strategies. Under mild assumptions on the attack model, we first propose an
inference strategy and formulate it as an integer programming(IP) task. We show it achieves high inference accuracy when sufficient information is exchanged across the distributed parties. Then, we present two defense strategies at one party, one perturbing its evaluation of model fairness and the other randomizing its process of selecting fair models. We show they effectively defend the inference, by preventing the IP solver from returning feasible solutions, without sacrificing a significant amount of model fairness. Theoretical properties of the proposed attack and defenses strategies are briefly discussed.

### Dataset

Datasets for Fair Machine Learning Research: https://github.com/HuiHu1/Datasets-for-Fair-Machine-Learning-Research.
