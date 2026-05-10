# This file is copyrighted under the latest version of the EUPL.
# Please see LICENCE file for your rights under this license.


"""
    plot_pulse(ϕ, filename::AbstractString)

Plot a pulse `ϕ` to file.
"""
function plot_pulse(ϕ, filename::AbstractString)
    f = Plots.plot(vcat(ϕ,ϕ[end]), linetype=:steppost)
    savefig(f, filename)
end

"""
    plot_pulsecycles(p::OptimizationParameters, ϕ, folder::AbstractString)

Plot each cycle separately in file `"\$folder/cycle-\$i.pdf"`.
"""
function plot_pulsecycles(p::OptimizationParameters, ϕ, folder::AbstractString)
    for i in 1:p.NCycles
        plot_pulse(view_cycle(p,ϕ,i), "$folder/cycle-$i.pdf")
    end
end

"""
    plot_many_convergence(traces, filename::AbstractString)

Plot convergence traces of several optimizations.
"""
function plot_many_convergence(traces, filename::AbstractString)
    f=plot()
    for trace in traces
        if length(trace) > 1
            plot!(f, trace, xaxis=:log, yaxis=:log, legend=false) 
        end
    end
    savefig(f,filename)
end

"""
    plot_cost_profile(p::OptimizationParameters, ϕ::FullPulse, filename::AbstractString)

Plot cost profile of the pulse `ϕ`.
"""
function plot_cost_profile(p::OptimizationParameters, ϕ::FullPulse, filename::AbstractString)
    p = OptimizationParameters(p, NDet=512, NRab=512)
    M = p.NRab
    K = p.NDet

    pc = OptimizationPrecomputed(p)
    s = OptimizationState(p)
    set_pulse(s,ϕ)
    propagate_state_compute_cost(p,s,pc)

    x = p.Rabi * range(1.0-p.RabiFDev, 1.0+p.RabiFDev, M)
    y = p.Rabi * range(-p.Detuning, p.Detuning, K)

    contour(x, y, log10.(1 .-(real.(Array(s.Ψ)[1,:,:])).^2), levels=6, fill=true, c = :turbo, lw=0.) #clabels=true,

    savefig(filename)
end