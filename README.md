
# UTrack.jl

Implementation of U-Track

## Installation

[As usual for a registered Julia package](https://docs.julialang.org/en/v1/stdlib/Pkg/), `UTrack` can be installed by typing

```
] add UTrack
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

This will optimize a dynamical decoupling sequence composed of 50 ``\pi``-pulses which refocusses every 10 pulses.
The genetic algorithm will use a population size of 12 pulses over 5 and 10 generations.
All results, including the final and intermediate pulses as well as various plots will be stored in the given folder.

If, for any reason, the optimization is interrupted, it can be restarted from the latest state using: 

```julia
continue_optimization(folder)
```

## Documentation

The documentation of `UTrack.jl` is available at <https://>.

## Citing UTrack.jl

If you use UTrack.jl in your work, please cite the following

E. Malvetti, L. Van Damme, A. Devra, and S. J. Glaser (2026): *UTrack.jl: A Julia Package for Highly Parallel Optimization of Dynamical Decoupling Sequences using Optimal Tracking.* DOI: ...

```bibtex
@Article{DD-julia,
  title = {UTrackDD.jl: A Julia Package for Highly Parallel Optimization of Dynamical Decoupling Sequences using Optimal Tracking},
  author = {Malvetti, E. and Van Damme, L., A. Devra, S. J. Glaser},
  journal = {JOSS},
  year = {2026},
  doi = {}
}
```

and

A. Devra, E. Malvetti, N. J. Glaser, L. Van Damme, and S. J. Glaser (2026): *Dynamical Decoupling using Optimal Tracking Approach.* DOI:

```bibtex
@Article{DD-paper,
  title = {Dynamical Decoupling using Optimal Tracking Approach},
  author = {A. Devra, E. Malvetti, N. J. Glaser, L. Van Damme, S. J. Glaser},
  journal = {},
  year = {2026},
  doi = {}
}
```
## License

The source code of this project is licensed under the [EUPL](LICENSE).