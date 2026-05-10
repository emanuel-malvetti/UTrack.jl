# This file is copyrighted under the latest version of the EUPL.
# Please see LICENCE file for your rights under this license.


"""
    view_cycle(p::OptimizationParameters, ϕ::FullPulse, i::Integer)

Returns a `view` into the `i`-th cycle of `ϕ`.
"""
function view_cycle(p::OptimizationParameters, ϕ::FullPulse, i::Integer)
    K = p.NCycles
    @assert i <= K
    L = p.NSteps
    view(ϕ,(i-1)*L+1:i*L)
end


# Permute cycles
function permute_cycles(p::OptimizationParameters, ϕ::FullPulse)
    K = p.NCycles
    L = p.NSteps

    perm = randperm(K)
    ϕ2 = Vector{Float64}(undef,K*L)

    for i in 1:K
        view_cycle(p,ϕ2,i) .= view_cycle(p,ϕ,perm[i])
    end

    ϕ2
end

function permute_cycles_good(p::OptimizationParameters, ϕ::FullPulse, tries=1000)
    pulses = [permute_cycles(p, ϕ) for _ in 1:tries]
    i = findmin([compute_cost(p,ϕ) for ϕ in pulses])[2]
    return pulses[i]
end

function permute_many(p::OptimizationParameters, pulses::AbstractArray{FullPulse}, tries=1000)
    return [permute_cycles_good(p, ϕ, tries) for ϕ in pulses]
end


# Recombine two pulses
function recombination(p::OptimizationParameters, ϕ1::FullPulse, ϕ2::FullPulse)
    K = p.NCycles
    S = Int64(ceil(K/10))

    ϕ3 = copy(ϕ1)
    ϕ4 = copy(ϕ2)

    subset = randperm(K)[1:S]
    for i in subset
        view_cycle(p,ϕ3,i) .= view_cycle(p,ϕ2,i)
        view_cycle(p,ϕ4,i) .= view_cycle(p,ϕ1,i)
    end

    ϕ3,ϕ4
end

function recombine_good(p::OptimizationParameters, ϕ1::FullPulse, ϕ2::FullPulse, tries=1000)
    pulses = [recombination(p, ϕ1, ϕ2) for _ in 1:tries]
    costs = [compute_cost(p,ϕ3)+compute_cost(p,ϕ4) for (ϕ3,ϕ4) in pulses]
    i = findmin(costs)[2]
    return pulses[i]
end

function recombine_many(p::OptimizationParameters, pulses::AbstractArray{FullPulse}, tries=1000)
    out_pulses = Vector{Float64}[]

    for i in 1:Int64(floor(length(pulses)/2))
        ϕ1,ϕ2 = recombine_good(p, pulses[2*i-1], pulses[2*i], tries)
        push!(out_pulses, ϕ1)
        push!(out_pulses, ϕ2)
    end

    if mod(length(pulses),2) == 1
        push!(out_pulses, pulses[end])
    end

    return out_pulses
end

#=

# shift cycles
function shift_cycle(p::OptimizationParameters, ϕ::FullPulse)
    K = p.NCycles
    ϕ2 = copy(ϕ)
    for i in 1:K
        view_cycle(p, ϕ2, i) .= circshift(view_cycle(p, ϕ, i), rand(1:p.NSteps))  
    end
    return ϕ2
end

function shift_cycle_good(p::OptimizationParameters, ϕ::FullPulse)
    pulses = [shift_cycle(p, ϕ) for _ in 1:1000]
    i = findmin([compute_cost(p,ϕ) for ϕ in pulses])[2]
    return pulses[i]
end

function shift_cycle_many(p::OptimizationParameters, pulses::Vector{FullPulse})
    return [shift_cycle(p, ϕ) for ϕ in pulses]
end

function shift_cycle_many_good(p::OptimizationParameters, pulses::Vector{FullPulse})
    return [shift_cycle_good(p, ϕ) for ϕ in pulses]
end


# reverse cycles
function reverse_cycle(p::OptimizationParameters, ϕ::FullPulse)
    L = p.NSteps
    K = p.NCycles

    ϕ2 = copy(ϕ)

    for i in 1:K
        view_cycle(p, ϕ2, i) .= reverse(view_cycle(p, ϕ, i))  
    end

    return ϕ2
end

function reverse_cycle_many(p::OptimizationParameters, pulses::Vector{FullPulse})
    return [reverse_cycle(p, ϕ) for ϕ in pulses]
end

=#