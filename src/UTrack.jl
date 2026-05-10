# This file is copyrighted under the latest version of the EUPL.
# Please see LICENCE file for your rights under this license.

module UTrack

using Parameters, CUDA, Optim, ProgressMeter, CSV, Tables, Random, JSON3, Plots, MAT, Printf

export OptimizationParameters, new_optimization, continue_optimization, continue_optimization_from

include("types.jl")
include("parameters.jl")
include("utils.jl")

include("printing.jl")
include("plotting.jl")
include("io.jl")
include("persistence.jl")

include("UR_pulses.jl")
include("init_pulse.jl")

include("GRAPE/allocate.jl")
include("GRAPE/pulse.jl")
include("GRAPE/gradient.jl")
include("GRAPE/propagation.jl")
include("GRAPE/optimize.jl")
include("GRAPE/ideal.jl")

include("GRAPE_XUR/allocate.jl")
include("GRAPE_XUR/XUR_Pulse.jl")
include("GRAPE_XUR/propagation.jl")
include("GRAPE_XUR/gradient.jl")
include("GRAPE_XUR/optimize.jl")

# currently unused
#include("GRAPE_shift/allocate.jl")
#include("GRAPE_shift/gradient.jl")
#include("GRAPE_shift/propagation.jl")
#include("GRAPE_shift/optimize.jl")

include("evolution.jl")
include("optimize.jl")

end
