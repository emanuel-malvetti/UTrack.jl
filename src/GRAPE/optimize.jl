# This file is copyrighted under the latest version of the EUPL.
# Please see LICENCE file for your rights under this license.


"""
    fg_full!(F, G, ϕ, p, s, pc)

Evaluates cost function and gradient for use in Optim.optimize
"""
function fg_full!(F, G, ϕ, p, s, pc)
    set_pulse(s,ϕ)
    cost = propagate_state_compute_cost(p,s,pc)

    if G !== nothing
        backpropagate_compute_gradient(p,s,pc)
        get_gradient(s,G)
    end
    if F !== nothing
      return cost
    end
end

"""
    optimize_pulse(p::OptimizationParameters, pc::OptimizationPrecomputed, pulse::FullPulse, method, opts)

Optimizes a pulse, returns full optimization result.
"""
function optimize_pulse(p::OptimizationParameters, pc::OptimizationPrecomputed, pulse::FullPulse, method, opts)
    s = OptimizationState(p)
    return Optim.optimize(
        Optim.only_fg!((F, G, ϕ) -> fg_full!(F, G, ϕ, p, s, pc)), 
        pulse, 
        method, 
        opts)
end

# move to common Full&XUR
"""
    optimize_single_print(p::OptimizationParameters, pc, pulse, method=Optim.BFGS())

Optimizes pulse and prints trace.
"""
function optimize_single_print(p::OptimizationParameters, pc, pulse, method=Optim.BFGS())
    opts = Optim.Options(g_tol=p.gradTol, iterations=p.maxIter, store_trace=false, show_trace=false, callback=trace_printing)
    return optimize_pulse(p, pc, pulse, method, opts)
end

# move to common Full&XUR
"""
    optimize_single_trace(p::OptimizationParameters, pc, pulse, method=Optim.BFGS())

Optimizes pulse and stores trace.
"""
function optimize_single_trace(p::OptimizationParameters, pc, pulse, method=Optim.BFGS())
    opts = Optim.Options(g_tol=p.gradTol, iterations=p.maxIter, store_trace=true)
    return optimize_pulse(p, pc, pulse, method, opts)
end

function precompute(p::OptimizationParameters, pulse::FullPulse)
    return OptimizationPrecomputed(p)
end

function precompute(p::OptimizationParameters, pulse::XUR_Pulse)
    return OptimizationPrecomputedXUR(p)
end

function convert_result!(pulse::FullPulse, result)
    pulse .= result
end

function convert_result!(pulse::XUR_Pulse, result)
    pulse.phases .= result
end

# move to common Full&XUR
"""
    optimize_pulses!(p::OptimizationParameters, pulses; print_func=nothing, traces=nothing)

Optimizes a collection of `pulses`. Can be `FullPulse`s or `XUR_Pulse`s.
Sorts pulses according to increasing cost function value.
"""
function optimize_pulses!(p::OptimizationParameters, pulses; print_func=nothing, traces=nothing)
    pc = precompute(p, pulses[1])
    vals = zeros(length(pulses))

    store_traces = traces !== nothing
    if store_traces
        @assert length(traces) == p.NPulses
    end

    @showprogress Threads.@threads for i in eachindex(pulses)
        res = optimize_single_trace(p, pc, pulses[i])
        vals[i] = Optim.minimum(res)
        convert_result!(pulses[i], Optim.minimizer(res))
        if store_traces
            traces[i] = Optim.f_trace(res)
        end
    end

    perm = sortperm(vals)

    if print_func !== nothing 
        print_func(vals[perm])
    end

    pulses .= pulses[perm]
    if store_traces
        traces .= traces[perm]
    end
    
    return vals[perm]
end
