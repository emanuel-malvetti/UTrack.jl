# This file is copyrighted under the latest version of the EUPL.
# Please see LICENCE file for your rights under this license.


function fg_full_ideal!(F, G, ϕ, p, s, pc)
    set_pulse(s,make_pulse_ideal(p,ϕ))
    cost = propagate_state_compute_cost(p,s,pc)

    if G !== nothing
        backpropagate_compute_gradient(p,s,pc)
        get_gradient(s,G)
        modify_gradient(p,G)
    end
    if F !== nothing
      return cost
    end
end

function modify_gradient(p,G)
    n = p.NSteps
    for i in 0:p.NCycles-1
        ∂ = G[i*n+n]
        G[i*n+n] *= n-1
        for j in 1:n-1
            G[i*n+n] += (-1)^(j+1) * G[i*n+j]
            G[i*n+j] += (-1)^(j+1) * ∂
        end
    end
end


function wrap_pi(x::Real)
    x-π*round(x/π)
end

function make_pulse_ideal(p::OptimizationParameters, ϕ::UTrack.FullPulse)
    n = p.NSteps
    for i in 0:p.NCycles-1
        ε = wrap_pi(sum(ϕ[n*i+1:2:n*i+n])-sum(ϕ[n*i+2:2:n*i+n]))
        ϕ[n*i+n] += ε
    end
    ϕ
end

function optimize_pulse_ideal(p::OptimizationParameters, pc::OptimizationPrecomputed, pulse::FullPulse, method, opts)
    s = OptimizationState(p)
    return Optim.optimize(
        Optim.only_fg!((F, G, ϕ) -> fg_full_ideal!(F, G, ϕ, p, s, pc)), 
        pulse, 
        method, 
        opts)
end
