# This file is copyrighted under the latest version of the EUPL.
# Please see LICENCE file for your rights under this license.


"""
    backpropagate_compute_gradient_xur(p::OptimizationParameters, s::OptimizationStateXUR, pc::OptimizationPrecomputedXUR)

Backpropagates the state and computes the gradient stored in `s`.
Only call this after `propagate_state_compute_cost_xur`.
"""
function backpropagate_compute_gradient_xur(p::OptimizationParameters, s::OptimizationStateXUR{S,T,R}, pc::OptimizationPrecomputedXUR{S,T}) where {T<: CuArray, S <: CuArray, R <: CuArray}
    M = p.NRab
    K = p.NDet    
    @cuda threads=M blocks=K gradient_kernel_full_xur(s.Ψ,s.X,s.Φ,s.τ, p.NCycles, pc.P11,pc.P22,pc.Q21,pc.A21,pc.w, s.grad)
    s.δΦ .= view(sum(s.grad,dims=[1,2]),1,1,:)
    return 
end

function backpropagate_compute_gradient_xur(p::OptimizationParameters, s::OptimizationStateXUR{S,T,R}, pc::OptimizationPrecomputedXUR{S,T}) where {T<: Array, S <: Array, R <: Array} 
    gradient_cpu_full_xur(s.Ψ,s.X,s.Φ,s.τ, p.NCycles, pc.P11,pc.P22,pc.Q21,pc.A21,pc.w, s.grad)
    s.δΦ .= view(sum(s.grad,dims=[1,2]),1,1,:)
    return 
end

function gradient_kernel_full_xur(Ψ,X,Φ, t,NCycles, P11,P22,Q21,A21,w, grad)
    m = threadIdx().x
    k = blockIdx().x

    X[1,m,k] = 0.0
    X[2,m,k] = 0.0

    for n in NCycles:-1:1
        X[1,m,k] += -2 * w[m,k] * real(Ψ[1,m,k])

        expphi = exp(im*Φ[n])
        P21 = Q21[m,k,t[n]] * expphi;
        P12 = -conj(P21);

        Ψ[1,m,k], Ψ[2,m,k] = P22[m,k,t[n]]*Ψ[1,m,k] - P12     *Ψ[2,m,k], 
                            -P21    *Ψ[1,m,k]  + P11[m,k,t[n]]*Ψ[2,m,k]

        B21 = A21[m,k,t[n]] * expphi;
        B12 = -conj(B21);

        grad[m,k,n] = real( X[1,m,k]*Ψ[2,m,k]*B12 + X[2,m,k]*Ψ[1,m,k]*B21 )

        X[1,m,k], X[2,m,k] = P11[m,k,t[n]]*X[1,m,k] + P21*X[2,m,k],
                                P12*X[1,m,k] + P22[m,k,t[n]]*X[2,m,k]
    end

    return
end

function gradient_cpu_full_xur(Ψ,X,Φ, t,NCycles, P11,P22,Q21,A21,w, grad)

    for (m,k) in Tuple.(CartesianIndices(w))
        X[1,m,k] = 0.0
        X[2,m,k] = 0.0

        for n in NCycles:-1:1
            X[1,m,k] += -2 * w[m,k] * real(Ψ[1,m,k])

            expphi = exp(im*Φ[n])
            P21 = Q21[m,k,t[n]] * expphi;
            P12 = -conj(P21);

            Ψ[1,m,k], Ψ[2,m,k] = P22[m,k,t[n]]*Ψ[1,m,k] - P12     *Ψ[2,m,k], 
                                -P21    *Ψ[1,m,k]  + P11[m,k,t[n]]*Ψ[2,m,k]

            B21 = A21[m,k,t[n]] * expphi;
            B12 = -conj(B21);

            grad[m,k,n] = real( X[1,m,k]*Ψ[2,m,k]*B12 + X[2,m,k]*Ψ[1,m,k]*B21 )

            X[1,m,k], X[2,m,k] = P11[m,k,t[n]]*X[1,m,k] + P21*X[2,m,k],
                                    P12*X[1,m,k] + P22[m,k,t[n]]*X[2,m,k]
        end
    end

    return
end