# This file is copyrighted under the latest version of the EUPL.
# Please see LICENCE file for your rights under this license.


function fg_phases_shifts!(F, G, ϕσ, p, s, pc)
    s.ϕ .= CuArray(ϕσ[1:p.NCycles*p.NSteps])
    s.σ .= CuArray(smooth_steps.(ϕσ[1+p.NCycles*p.NSteps:end]))
    cost = propagate_state_compute_cost_shift(p,s,pc)

    if G !== nothing
        backpropagate_compute_gradient_shift(p,s,pc)
        G .= vcat(Array(s.δϕ), Array(s.δσ) .* smooth_steps_derv.(ϕσ[1+p.NCycles*p.NSteps:end]))
    end
    if F !== nothing
      return cost
    end
end

function optimize_phases_shifts(p::OptimizationParameters, pc::UTrack.OptimizationPrecomputed, ϕσ0)
    s = OptStateShift(p)
    res = Optim.optimize(
        Optim.only_fg!((F, G, ϕσ) -> fg_phases_shifts!(F, G, ϕσ, p, s, pc)), 
        ϕσ0, 
        Optim.GradientDescent(), 
        Optim.Options(g_tol=p.gradTol, iterations=p.maxIter, store_trace = true, show_trace=true #=, show_trace=true, show_warnings=true, show_every=1 =# ))
    
    return res #Optim.minimum(res), Optim.minimizer(res)
end


function fg_phases_only!(F, G, ϕ, σ, p, s, pc)
    s.ϕ .= CuArray(ϕ)
    s.σ .= CuArray(σ)
    cost = propagate_state_compute_cost_shift(p,s,pc)

    if G !== nothing
        backpropagate_compute_gradient_shift(p,s,pc)
        G .= Array(s.δϕ)
    end
    if F !== nothing
      return cost
    end
end

function optimize_phases_only(p::OptimizationParameters, pc::UTrack.OptimizationPrecomputed, ϕ0, σ)
    s = OptStateShift(p)
    res = Optim.optimize(
        Optim.only_fg!((F, G, ϕ) -> fg_phases_only!(F, G, ϕ, σ, p, s, pc)), 
        ϕ0, 
        Optim.BFGS(), 
        Optim.Options(g_tol=p.gradTol, iterations=p.maxIter, store_trace = true, show_trace=true #=, show_trace=true, show_warnings=true, show_every=1 =# ))
    
    return res #Optim.minimum(res), Optim.minimizer(res)
end


function fg_shifts_only!(F, G, ϕ, σ, p, s, pc)
    s.ϕ .= CuArray(ϕ)
    s.σ .= CuArray(σ)
    cost = propagate_state_compute_cost_shift(p,s,pc)

    if G !== nothing
        backpropagate_compute_gradient_shift(p,s,pc)
        G .= Array(s.δσ)
    end
    if F !== nothing
      return cost
    end
end

function optimize_shifts_only(p::OptimizationParameters, pc::UTrack.OptimizationPrecomputed, ϕ, σ0)
    s = OptStateShift(p)
    res = Optim.optimize(
        Optim.only_fg!((F, G, σ) -> fg_shifts_only!(F, G, ϕ, σ, p, s, pc)), 
        σ0, 
        Optim.ConjugateGradient(), 
        Optim.Options(g_tol=p.gradTol, iterations=p.maxIter, store_trace = true, show_trace=true #=, show_trace=true, show_warnings=true, show_every=1 =# ))
    
    return res #Optim.minimum(res), Optim.minimizer(res)
end

function fg_shifts_smooth!(F, G, ϕ, σ, p, s, pc)
    s.ϕ .= CuArray(ϕ)
    s.σ .= CuArray(smooth_steps.(σ))
    cost = propagate_state_compute_cost_shift(p,s,pc)

    if G !== nothing
        backpropagate_compute_gradient_shift(p,s,pc)
        G .= Array(s.δσ) .* smooth_steps_derv.(σ)
    end
    if F !== nothing
      return cost
    end
end


function optimize_shifts_smooth(p::OptimizationParameters, pc::UTrack.OptimizationPrecomputed, ϕ, σ0)
    s = OptStateShift(p)
    res = Optim.optimize(
        Optim.only_fg!((F, G, σ) -> fg_shifts_smooth!(F, G, ϕ, σ, p, s, pc)), 
        σ0, 
        Optim.BFGS(), 
        Optim.Options(g_tol=p.gradTol, iterations=p.maxIter, store_trace = true, show_trace=true #=, show_trace=true, show_warnings=true, show_every=1 =# ))
    
    return res #Optim.minimum(res), Optim.minimizer(res)
end




function cycle_gradient(p,grad)
    vcat([sum(UTrack.view_cycle(p,grad,i)) * ones(p.NSteps) for i in 1:p.NCycles]...)
end


function fg_cycles!(F, G, ϕσ, p, s, pc)
    N=p.NCycles*p.NSteps
    s.ϕ .= CuArray(ϕσ[1:N])
    s.σ .= CuArray(smooth_steps.(ϕσ[1+N:end]))
    cost = propagate_state_compute_cost_shift(p,s,pc)

    if G !== nothing
        backpropagate_compute_gradient_shift(p,s,pc)
        G .= vcat(cycle_gradient(p,Array(s.δϕ)), Array(s.δσ) .* smooth_steps_derv.(ϕσ[1+N:end]))
    end
    if F !== nothing
      return cost
    end
end

function optimize_cycles(p::OptimizationParameters, pc::UTrack.OptimizationPrecomputed, ϕσ0)
    s = OptStateShift(p)
    res = Optim.optimize(
        Optim.only_fg!((F, G, ϕσ) -> fg_cycles!(F, G, ϕσ, p, s, pc)), 
        ϕσ0, 
        Optim.GradientDescent(), 
        Optim.Options(g_tol=p.gradTol, iterations=p.maxIter, store_trace = true, show_trace=true))
    
    return res #Optim.minimum(res), Optim.minimizer(res)
end

# test:
function eval_cost(p,s,pc,ϕ, σ,Φ)
    s.ϕ .= CuArray(ϕ .+ kron(Φ, ones(p.NSteps)) )
    s.σ .= CuArray(smooth_steps.(σ))
    return propagate_state_compute_cost_shift(p,s,pc)
end

function finite_differences2(p::OptimizationParameters, pc::UTrack.OptimizationPrecomputed, ϕ, σ, Φ, h = 1e-8)
    N=p.NSteps*p.NCycles
    s = OptStateShift(p)
    grad_Φ = zeros(p.NCycles)
    grad_σ = zeros(p.NCycles)
    c1 = eval_cost(p,s,pc,ϕ, σ,Φ)

    for i in 1:p.NCycles 
        Φ[i] += h
        c2 = eval_cost(p,s,pc,ϕ, σ,Φ)
        Φ[i] -= h
        grad_Φ[i] = (c2-c1)/h
    end

    for i in 1:p.NCycles
        σ[i] += h
        c2 = eval_cost(p,s,pc,ϕ, σ,Φ)
        σ[i] -= h
        grad_σ[i] = (c2-c1)/h
    end

    vcat(grad_Φ, grad_σ)
end

function gradient_exact(p::OptimizationParameters, pc::UTrack.OptimizationPrecomputed, ϕ, σ, Φ)
    s = OptStateShift(p)
    s.ϕ .= CuArray(ϕ .+ kron(Φ, ones(p.NSteps)) )
    s.σ .= CuArray(smooth_steps.(σ))
    propagate_state_compute_cost_shift(p,s,pc)
    backpropagate_compute_gradient_shift(p,s,pc)
    vcat(
        [sum(UTrack.view_cycle(p,Array(s.δϕ),i)) for i in 1:p.NCycles], 
        Array(s.δσ) .* smooth_steps_derv.(σ))
end

#=
p = OptimizationParameters(NSteps=100,NCycles=10,Δt=π/10,maxIter=10000)
pc = UTrack.OptimizationPrecomputed(p)
ϕ = [sin(2π * n/p.NSteps) for n in 1:p.NSteps*p.NCycles]
#ϕ = [2π *randn() for _ in 1:p.NSteps*p.NCycles]

#ϕ = UTrack.UR_pulse(p)
σ = [rand()*p.NSteps for _ in 1:p.NCycles]
Φ = [2π * randn() for _ in 1:p.NCycles]
#σ = [0.0 for _ in 1:p.NCycles]
#s.ϕ .= CuArray(ϕ)
#s.σ .= CuArray(σ)

res=optimize_shifts_smooth(p,pc,ϕ,σ)

res=optimize_shifts_only(p,pc,ϕ,σ)

res=optimize_phases_shifts(p,pc,vcat(ϕ,σ))

res=optimize_phases_only(p,pc,ϕ,σ)

res=optimize_cycles(p,pc,vcat(ϕ,σ))


res=UTrack.optimize_single(p,pc,ϕ)

opt = Optim.minimizer(res)
UTrack.plot_pulse(opt,"test.pdf")
=#