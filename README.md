
# UTrack.jl

Implementation of an optimization algorithm for generating U-Track dynamical decoupling sequences.

## Installation

Since this package is not yet registered, you can install it directly from GitHub by typing:

```julia
using Pkg
Pkg.add(url="https://github.com/emanuel-malvetti/UTrack.jl.git")
```

in the Julia REPL.

## Quick Start

To perform an optimization, simply construct an `OptimizationParameters` object and call `new_optimization`.

```julia
using UTrack

folder = "results/opt-10x5/"
p = OptimizationParameters(NSteps=10, NCycles=5, NPulses=12, NGens=10, NGensXUR=5)
new_optimization(folder, p)
```

This will optimize a dynamical decoupling sequence composed of 50 $\pi$-pulses which refocusses every 10 pulses.
The genetic algorithm will use a population size of 12 pulses over 5 and 10 generations.
All results, including the final and intermediate pulses as well as various plots will be stored in the given folder.

If, for any reason, the optimization is interrupted, it can be restarted from the latest state using: 

```julia
continue_optimization(folder)
```

## Documentation

The documentation of `UTrack.jl` is available at <https://emanuel-malvetti.github.io/UTrack.jl/dev/>.

## Citing UTrack.jl

If you use UTrack.jl in your work, please cite the following:

A. Devra, E. Malvetti, N. J. Glaser, A. Agarwal, I. Rungger, S. Lujan, M. Werninghaus, S. Filipp, L. Van Damme, and S. J. Glaser (2026): *Dynamical Decoupling using Universal Optimal Tracking* DOI: https://doi.org/10.48550/arXiv.2606.21762

```bibtex
@misc{UTrack26,
  title        = {Dynamical Decoupling using Optimal Tracking Approach},
  author       = {Devra, A. and Malvetti, E. and Glaser, N. J. and Agarwal, A. and Rungger, I. and Lujan, S. and Werninghaus, M. and Filipp, S. and Van Damme, L. and Glaser, S. J.},
  year         = {2026},
  eprint       = {2606.21762},
  archivePrefix = {arXiv},
  doi          = {10.48550/arXiv.2606.21762},
  url          = {https://arxiv.org/abs/2606.21762}
}
```
## License

The source code of this project is licensed under the [EUPL](LICENSE).
