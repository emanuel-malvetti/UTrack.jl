# This file is copyrighted under the latest version of the EUPL.
# Please see LICENCE file for your rights under this license.


"""
    fg_xur!(F, G, Φ, p, s, pc)

Evaluates cost function and gradient for use in Optim.optimize
"""
function fg_xur!(F, G, Φ, p, s, pc)
    set_pulse(s,Φ)
    cost = propagate_state_compute_cost_xur(p,s,pc)

    if G !== nothing
        backpropagate_compute_gradient_xur(p,s,pc)
        get_gradient(s,G)
    end
    if F !== nothing
      return cost
    end
end

"""
    optimize_pulse(p::UTrack.OptimizationParameters, pc::UTrack.OptimizationPrecomputedXUR, xur::UTrack.XUR_Pulse, method, opts)

Optimizes an XUR pulse, returns full optimization result.
"""
function optimize_pulse(p::UTrack.OptimizationParameters, pc::UTrack.OptimizationPrecomputedXUR, xur::UTrack.XUR_Pulse, method, opts)
    s = OptimizationStateXUR(p)
    set_types(s, [t_from_sign_shift(xur.signs[i],xur.shifts[i]) for i in 1:p.NCycles])
    return Optim.optimize(
        Optim.only_fg!((F, G, ϕ) -> fg_xur!(F, G, ϕ, p, s, pc)), 
        xur.phases, 
        method, 
        opts)
end
