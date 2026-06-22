---
title: "UTrack.jl: A Julia Package for Highly Parallel Optimization of Dynamical Decoupling Sequences using Optimal Tracking"
tags:
  - Julia
  - quantum control
  - dynamical decoupling
  - quantum computing
  - GPU
authors:
  - name: Emanuel Malvetti
    orcid: 0000-0002-0736-1613
    affiliation: "1, 2"
  - name: Léo Van Damme
    orcid: 
    affiliation: "1, 2"
  - name: Amit Devra
    orcid: 
    affiliation: "1, 2"\
  - name: Steffen J. Glaser
    orcid: 
    affiliation: "1, 2"
affiliations:
  - name: Technical University Munich, Germany
    index: 1
  - name: MQV and MCQST
    index: 2
date: ...
bibliography: paper.bib
---

# Summary

In quantum information science, in particular quantum computing, the fundamental unit of information is the qubit.
These qubits are very delicate physical systems strongly affected by noise stemming from various sources, depending on the physical implementation.
Idling qubits will decohere over time, leading to the loss of information.
A popular way to mitigate this effect is dynamical decoupling. [@Hahn]
In this method, a pulse sequence is applied to idling qubits such that the information is protected and decoherence is significantly slowed down.
The `UTrack.jl` software package is an efficient and highly parallel optimizer for dynamical decoupling sequences.
It employs a combination of the GRAPE algorithm [@GRAPE] and a genetic algorithm to globally optimize dynamical decoupling sequences which are robust to noise and repeatedly refocus.
This software was used for the paper [@DD] where the improvement over previously known dynamical decoupling sequences is demonstrated.

# Statement of need
 
UTrack.jl generates dynamical decoupling sequences that improve on existing sequences, and it does
so using quantum optimal control. Since these optimization can be quite expensive, a highly parallel implementation is necessary.

Existing implementation of the GRAPE algorithm include QuTiP [@qutip24] (Python), GRAPE.jl (part of JuliaQuantumControl) [@juliaqc] (Julia), and Spinach [@spinach] (MATLAB).
The GRAPE implementation provided in UTrack.jl is highly optimized for the specific task, precomputing some of the propagator elements to minimize the number of operations needed to perform the forward and backward propagations.
Moreover, the gradient computation is performed on a GPU using CUDA.jl, and several optimizations can be run in parallel to make use of multiple CPUs.
Finally, the optimal tracking approach which is essential to the optimization of dynamical decoupling sequences requires an unusual cost function evaluated at regular intervals along the pulse, necessitating a custom implementation of the GRAPE algorithm.
Genetic algorithms have also been applied to the search for dynamical decoupling sequences[@genetic], using simulated annealing instead of gradient based optimization, and without optimal tracking.

# Acknowledgements

We would like to thank Abhishek Agarwal, and Niklas Glaser for their valuable feedback on this software package.

# References
