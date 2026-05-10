# This file is copyrighted under the latest version of the EUPL.
# Please see LICENCE file for your rights under this license.


"""
    compute_cost(p::OptimizationParameters, ϕ::FullPulse, pc = UTrack.OptimizationPrecomputed(p))

Helper function to call `propagate_state_compute_cost` for a given pulse `ϕ`.
"""
function compute_cost(p::OptimizationParameters, ϕ::FullPulse, pc = UTrack.OptimizationPrecomputed(p))
    s = UTrack.OptimizationState(p)
    set_pulse(s,ϕ)
    return UTrack.propagate_state_compute_cost(p,s,pc)
end

"""
    propagate_state_compute_cost(p::OptimizationParameters, s::OptimizationState, pc::OptimizationPrecomputed)

Initializes and propagates the state, and computes the cost function at the same time.
Returns the value of the cost function. Uses CUDA if available.
"""
function propagate_state_compute_cost(p::OptimizationParameters, s::OptimizationState{T,S}, pc::OptimizationPrecomputed{T,S}) where {T<: CuArray, S <: CuArray}
    M = p.NDet
    K = p.NRab
    @cuda threads=M blocks=K propagate_kernel_full(s.Ψ, s.ϕ, p.NSteps, p.NCycles, s.cost, pc.w, pc.Q21, pc.P11, pc.P22)
    return sum(s.cost)/p.NCycles
end

function propagate_state_compute_cost(p::OptimizationParameters, s::OptimizationState{T,S}, pc::OptimizationPrecomputed{T,S}) where {T<: Array, S <: Array}
    propagate_cpu_full(s.Ψ, s.ϕ, p.NSteps, p.NCycles, s.cost, pc.w, pc.Q21, pc.P11, pc.P22)
    return sum(s.cost)/p.NCycles
end

# CUDA kernel
function propagate_kernel_full(Ψ, ϕ, NSteps, NCycles, cost, w, Q21, P11, P22)
    m = threadIdx().x
    k = blockIdx().x

    Ψ[1,m,k]=1.0
    Ψ[2,m,k]=0.0
    cost[m,k]=0.0

    for cycle in 1:NCycles
        for r in 1:NSteps
            n = r + (cycle-1)*NSteps 
            P21 = Q21[m,k]*exp(im*ϕ[n])
            P12 = -conj(P21)
            Ψ[1,m,k], Ψ[2,m,k] = P11[m,k]*Ψ[1,m,k] + P12*Ψ[2,m,k],  P21*Ψ[1,m,k] + P22[m,k]*Ψ[2,m,k]
        end
        cost[m,k] += real(Ψ[1,m,k])^2
    end

    cost[m,k] = w[m,k] * (NCycles - cost[m,k])

    return
end

# CPU fallback
function propagate_cpu_full(Ψ, ϕ, NSteps, NCycles, cost, w, Q21, P11, P22)
    for (m,k) in Tuple.(CartesianIndices(w))
        Ψ[1,m,k]=1.0
        Ψ[2,m,k]=0.0
        cost[m,k]=0.0

        for cycle in 1:NCycles
            for r in 1:NSteps
                n = r + (cycle-1)*NSteps 
                P21 = Q21[m,k]*exp(im*ϕ[n])
                P12 = -conj(P21)
                Ψ[1,m,k], Ψ[2,m,k] = P11[m,k]*Ψ[1,m,k] + P12*Ψ[2,m,k],  P21*Ψ[1,m,k] + P22[m,k]*Ψ[2,m,k]
            end
            cost[m,k] += real(Ψ[1,m,k])^2
        end

        cost[m,k] = w[m,k] * (NCycles - cost[m,k])
    end

    return
end