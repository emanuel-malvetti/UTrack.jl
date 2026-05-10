# Developer Documentation

This documentation describes the internal functioning of the library. 
For most users, the [User Manual](@ref) is sufficient.

## Full GRAPE

The basic GRAPE implementation works on the full pulse and uses tracking to refocus the propagator
at regular time intervals. Some mathematical details are given in [Background Details](@ref).

```@docs
UTrack.FullPulse
```

The state of the optimization is stored in an `OptimizationState`, which automatically detects if CUDA is available and falls back to a CPU implementation otherwise.

```@docs
UTrack.OptimizationState
UTrack.OptimizationState(::OptimizationParameters)

UTrack.set_pulse(s::UTrack.OptimizationState{T,S}, ϕ::UTrack.FullPulse) where {T<: CuArray, S <: CuArray}
UTrack.set_pulse(s::UTrack.OptimizationState{T,S}, ϕ::UTrack.FullPulse) where {T<: Array, S <: Array}

UTrack.get_gradient(s::UTrack.OptimizationState, δϕ) 
```

Some information can be computed in advance once the `OptimizationParameters` are known.

```@docs
UTrack.OptimizationPrecomputed
UTrack.OptimizationPrecomputed(p::OptimizationParameters)
```

The GRAPE algorithm is performed by the following two functions which execute CUDA kernels (if available).

```@docs
UTrack.propagate_state_compute_cost
UTrack.backpropagate_compute_gradient
```

## Cycle GRAPE (XUR)

```@docs
UTrack.XUR_Pulse
```

The state of the optimization is stored in an `OptimizationStateXUR` object.

```@docs
UTrack.OptimizationStateXUR
UTrack.OptimizationStateXUR(::OptimizationParameters)

UTrack.set_pulse(::UTrack.OptimizationStateXUR{T,S,R}, ::UTrack.XUR_Pulse) where {T<: CuArray, S <: CuArray, R <: CuArray}
UTrack.set_pulse(::UTrack.OptimizationStateXUR{T,S,R}, ::UTrack.XUR_Pulse) where {T<: Array, S <: Array, R <: Array}

UTrack.set_types

UTrack.get_gradient(s::UTrack.OptimizationStateXUR, δϕ) 
```

As in the full optimization, the precomputed data is stored in an `OptimizationPrecomputedXUR` object.

```@docs
UTrack.OptimizationPrecomputedXUR
UTrack.OptimizationPrecomputedXUR(p::OptimizationParameters)
```

The cycle GRAPE algorithm is performed by the following two functions which execute CUDA kernels if available.

```@docs
UTrack.propagate_state_compute_cost_xur
UTrack.backpropagate_compute_gradient_xur
```



## Gradient Optimization

For the gradient based optimization of the pulse, we combine the GRAPE algorithm to compute the gradient efficiently with the general purpose optimization package `Optim`. In particular we rely on the optimization algorithms BFGS and L-BFGS.

A list of pulses with the same parameters can be optimized simultaneously.

```@docs
UTrack.optimize_pulses!
```

A single optimization is implemented using the following functions.

```@docs
UTrack.fg_full!
UTrack.fg_xur!
UTrack.optimize_pulse
```



## Genetic Algorithm

To avoid getting stuck in local optima, the gradient based optimization is complemented with a genetic algorithm to explore larger regions of the optimization space.

```@docs
UTrack.evolve_generation
```



## Complete Optimization

As described in [User Manual](@ref), in practice it is recommended to use `new_optimization` and `continue_optimization` in practice.

```@docs
UTrack.perform_optimization_steps!
UTrack.new_optimization
UTrack.new_optimization_no_xur
UTrack.continue_optimization
```


## Persistence Functions

```@docs
UTrack.setup_folder(folder::AbstractString, p::OptimizationParameters)

UTrack.save_parameters(folder::String, p::OptimizationParameters)
UTrack.load_parameters(folder::String)

UTrack.save_generation(folder::AbstractString, gen::Integer, p::OptimizationParameters, pulses::AbstractArray{UTrack.FullPulse})
UTrack.save_generation(folder::AbstractString, gen::Integer, p::OptimizationParameters, pulses::AbstractArray{UTrack.XUR_Pulse})
UTrack.load_generation(folder::AbstractString, gen::Integer, p::OptimizationParameters, type)

UTrack.latest_gen(folder::AbstractString, type)
UTrack.load_latest_generation(folder::AbstractString)
UTrack.load_best_pulse(folder::AbstractString)

UTrack.plot_optimization
UTrack.plot_grape_convergence

UTrack.plot_and_write_final
```

## Helper Functions

### Printing

```@docs
UTrack.trace_printing(x)
UTrack.generation_printing(gen::Integer, values)
```

### Plotting

```@docs
UTrack.plot_pulse(ϕ, ::AbstractString)
UTrack.plot_pulsecycles(p::OptimizationParameters, ϕ, ::AbstractString)
UTrack.plot_many_convergence(traces, filename::AbstractString)
UTrack.plot_cost_profile
```

### I/O

The following functions are used for reading and writing parameters and pulses from and to files.

Parameters are stored in JSON files.

```@docs
UTrack.read_parameters(filename::AbstractString)
UTrack.write_parameters(filename::AbstractString, p::OptimizationParameters)
```

Pulses (of the same length) are stored in CSV files, one pulse per column.

```@docs
UTrack.read_all_pulses_full(filename::AbstractString)
UTrack.read_all_pulses_xur(filename::AbstractString, p::OptimizationParameters)
UTrack.write_all_pulses_full(filename::AbstractString, pulses::AbstractArray{UTrack.FullPulse})
UTrack.write_all_pulses_xur(filename::AbstractString, p::OptimizationParameters, xur_pulses::AbstractArray{UTrack.XUR_Pulse})
```

For convenience, a single pulse can be stored in a Matlab file.

```@docs
UTrack.read_pulse_matlab
UTrack.write_pulse_matlab
```

## Pulses

```@docs
UTrack.UR_cycle
UTrack.UR_cycles
UTrack.PulseType
UTrack.UR_pulse
UTrack.initialize_pulse
UTrack.initialize_pulses
UTrack.renormalize_pulses!
```
