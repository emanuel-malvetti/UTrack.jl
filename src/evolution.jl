# This file is copyrighted under the latest version of the EUPL.
# Please see LICENCE file for your rights under this license.


# combine the two functions

"""
    evolve_generation(p::OptimizationParameters, pulses::Vector{UTrack.FullPulse})

A list of `pulses` (ordered from best to worst) is modified via genetic update and the list of modified
pulses is returned. The best pulse is always preserved (elitism). The new pulses are obtained via 
*recombination* or *permutation*. 
"""
function evolve_generation(p::OptimizationParameters, pulses::Vector{UTrack.FullPulse}, T=0)
    N = length(pulses)
    unit = Int64(floor(N/10))
    keep_n = 1*unit
    recomb_n = 3*unit
    mutate_n = 4*unit + mod(N,10)
    replace_n = unit

    keep_range = 1:keep_n
    recomb_range = (1+keep_n):(keep_n+recomb_n)
    mutate_range = (1+keep_n+recomb_n):(keep_n+recomb_n+mutate_n)
    replace_range_1 = (1+keep_n+recomb_n+mutate_n):(keep_n+recomb_n+mutate_n+replace_n)
    replace_range_2 = (1+keep_n+recomb_n+mutate_n+replace_n):(keep_n+recomb_n+mutate_n+2*replace_n)

    view(pulses, recomb_range) .= recombine_many(p, view(pulses, recomb_range))
    view(pulses, mutate_range) .= permute_many(p, view(pulses, mutate_range))
    view(pulses, replace_range_1) .= recombine_many(p, view(pulses, keep_range))
    view(pulses, replace_range_2) .= permute_many(p, view(pulses, keep_range))

    return pulses
end


"""
    evolve_generation(p::OptimizationParameters, pulses::Vector{UTrack.XUR_Pulse})

A list of XUR `pulses` (ordered from best to worst) is modified via genetic update and the list of modified
pulses is returned. The best pulse is always preserved (elitism). The new pulses are obtained via 
*recombination* or *permutation*. 
"""
function evolve_generation(p::OptimizationParameters, pulses::Vector{UTrack.XUR_Pulse}, T=0)
    pc = OptimizationPrecomputedXUR(p)
    
    N = length(pulses)
    unit = Int64(floor(N/10))
    keep_n = 1*unit
    recomb_n = 3*unit
    mutate_n = 4*unit + mod(N,10)
    replace_n = unit

    keep_range = 1:keep_n
    recomb_range = (1+keep_n):(keep_n+recomb_n)
    mutate_range = (1+keep_n+recomb_n):(keep_n+recomb_n+mutate_n)
    replace_range_1 = (1+keep_n+recomb_n+mutate_n):(keep_n+recomb_n+mutate_n+replace_n)
    replace_range_2 = (1+keep_n+recomb_n+mutate_n+replace_n):(keep_n+recomb_n+mutate_n+2*replace_n)

    view(pulses, recomb_range) .= recombine_xur_many(p, pc, view(pulses, recomb_range), T)
    view(pulses, mutate_range) .= mutate_xur_many(p, pc, view(pulses, mutate_range))
    view(pulses, replace_range_1) .= recombine_xur_many(p, pc, view(pulses, keep_range), T)
    view(pulses, replace_range_2) .= mutate_xur_many(p, pc, view(pulses, keep_range))

    return pulses
end
