# L65 project: Revisiting Time Encoding Functions in Temporal Graph Neural Networks

## Abstract

Learning useful representations of time is an essential aspect of temporal graph
learning, the importance of which is often overlooked. We identify two limitations
in the standard representation of time using a vector of sinusoidal signals: 1)
TGNNs often struggle with learning appropriate frequency values during training,
and 2) they fail to distinguish useful properties of the representation from noise.
This leads to an excessive high frequency oscillation of link predictions with time,
which we quantify using the novel total variation metric, and significantly mitigate
through the design of several new time encoding functions. We verify our methods
on a number of TGNN models and real-world dynamic link prediction datasets.

## Codebase and commit log

The code for this project is organised in the following way: 
1. [QianyiLiu309/TGB](https://github.com/QianyiLiu309/TGB.git): this submodule contains every change we made on top of TGB, including evaluation scripts and metric calculation pipelines for TGN and DyRep. 
2. [zzbuzzard/TGB_Baselines](https://github.com/zzbuzzard/TGB_Baselines.git): this submodule contains our changes on top of TGB_Baselines, including evaluation scripts and metric calculation pipelines for GraphMixer.
3. This top-level repository, which contains implementations of time encoding functions, plotting scripts, and code for calculating the smoothness metric. 

For detailed commit logs, see inside the two submodules. 

