# This file is copyrighted under the latest version of the EUPL.
# Please see LICENCE file for your rights under this license.


function compute_cost(p::OptimizationParameters, xur::XUR_Pulse, pcx = OptimizationPrecomputedXUR(p))
    sx = UTrack.OptimizationStateXUR(p)
    set_pulse(sx, xur.phases)
    set_types(sx, [UTrack.t_from_sign_shift(xur.signs[i],xur.shifts[i]) for i in 1:p.NCycles])
    return UTrack.propagate_state_compute_cost_xur(p,sx,pcx)
end

"""
    propagate_state_compute_cost_xur(p::OptimizationParameters, s::OptimizationStateXUR, pc::OptimizationPrecomputedXUR)

Initializes and propagates the state, and computes the cost function at the same time.
Returns the value of the cost function.
"""
function propagate_state_compute_cost_xur(p::OptimizationParameters, s::OptimizationStateXUR{T,S,R}, pc::OptimizationPrecomputedXUR{T,S}) where {T<: CuArray, S <: CuArray, R <: CuArray}
    M = p.NDet
    K = p.NRab
    @cuda threads=M blocks=K propagate_kernel_full_xur(s.Ψ, s.Φ, s.τ, p.NCycles, s.cost, pc.w, pc.Q21, pc.P11, pc.P22)    
    return sum(s.cost)/p.NCycles
end

function propagate_state_compute_cost_xur(p::OptimizationParameters, s::OptimizationStateXUR{T,S,R}, pc::OptimizationPrecomputedXUR{T,S}) where {T<: Array, S <: Array, R <: Array}
    propagate_cpu_full_xur(s.Ψ, s.Φ, s.τ, p.NCycles, s.cost, pc.w, pc.Q21, pc.P11, pc.P22)    
    return sum(s.cost)/p.NCycles
end

function propagate_kernel_full_xur(Ψ, Φ, t, NCycles, cost, w, Q21, P11, P22)
    m = threadIdx().x
    k = blockIdx().x

    Ψ[1,m,k]=1.0
    Ψ[2,m,k]=0.0
    cost[m,k]=0.0

    for n in 1:NCycles
        P21 = Q21[m,k,t[n]]*exp(im*Φ[n])
        P12 = -conj(P21)
        Ψ[1,m,k], Ψ[2,m,k] = P11[m,k,t[n]]*Ψ[1,m,k] + P12*Ψ[2,m,k],  P21*Ψ[1,m,k] + P22[m,k,t[n]]*Ψ[2,m,k]
        cost[m,k] += real(Ψ[1,m,k])^2
    end

    cost[m,k] = w[m,k] * (NCycles - cost[m,k])

    return
end

function propagate_cpu_full_xur(Ψ, Φ, t, NCycles, cost, w, Q21, P11, P22)

    for (m,k) in Tuple.(CartesianIndices(w))
        Ψ[1,m,k]=1.0
        Ψ[2,m,k]=0.0
        cost[m,k]=0.0

        for n in 1:NCycles
            P21 = Q21[m,k,t[n]]*exp(im*Φ[n])
            P12 = -conj(P21)
            Ψ[1,m,k], Ψ[2,m,k] = P11[m,k,t[n]]*Ψ[1,m,k] + P12*Ψ[2,m,k],  P21*Ψ[1,m,k] + P22[m,k,t[n]]*Ψ[2,m,k]
            cost[m,k] += real(Ψ[1,m,k])^2
        end

        cost[m,k] = w[m,k] * (NCycles - cost[m,k])
    end

    return
end
