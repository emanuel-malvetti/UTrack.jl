# This file is copyrighted under the latest version of the EUPL.
# Please see LICENCE file for your rights under this license.


"""
    backpropagate_compute_gradient(p::OptimizationParameters, s::OptimizationState, pc::OptimizationPrecomputed)

Backpropagates the state and computes the gradient stored in `s`.
Only call this after `propagate_state_compute_cost`. Uses CUDA if available. 
"""
function backpropagate_compute_gradient(p::OptimizationParameters, s::OptimizationState{T,S}, pc::OptimizationPrecomputed{T,S}) where {T<: CuArray, S <: CuArray}
    M = p.NRab
    K = p.NDet
    N = p.NSteps*p.NCycles
    
    @cuda threads=M blocks=K gradient_kernel_full(s.Ψ,s.X,s.ϕ,p.NSteps,p.NCycles,N, pc.P11,pc.P22,pc.Q21,pc.A21,pc.w, s.grad)
    s.δϕ .= view(sum(s.grad,dims=[1,2]),1,1,:) # check normalization (divide by p.NCycles ?)
    
    return 
end

function backpropagate_compute_gradient(p::OptimizationParameters, s::OptimizationState{T,S}, pc::OptimizationPrecomputed{T,S}) where {T<: Array, S <: Array}
    M = p.NRab
    K = p.NDet
    N = p.NSteps*p.NCycles
    
    gradient_cpu_full(s.Ψ,s.X,s.ϕ,p.NSteps,p.NCycles,N, pc.P11,pc.P22,pc.Q21,pc.A21,pc.w, s.grad)
    s.δϕ .= view(sum(s.grad,dims=[1,2]),1,1,:) # check normalization (divide by p.NCycles ?)
    
    return 
end


function gradient_kernel_full(Ψ,X,ϕ, NSteps,NCycles,N, P11,P22,Q21,A21,w, grad)
    m = threadIdx().x
    k = blockIdx().x

    X[1,m,k] = 0.0
    X[2,m,k] = 0.0

    for cycle in 1:NCycles
        X[1,m,k] += -2 * w[m,k] * real(Ψ[1,m,k]) 

        for r in 1:NSteps
            n = N-(r + (cycle-1)*NSteps )+1
            expphi = exp(im*ϕ[n])
            P21 = Q21[m,k] * expphi;
            P12 = -conj(P21);
    
            Ψ[1,m,k], Ψ[2,m,k] = P22[m,k]*Ψ[1,m,k] - P12     *Ψ[2,m,k], 
                                -P21    *Ψ[1,m,k]  + P11[m,k]*Ψ[2,m,k]
    
            B21 = A21[m,k] * expphi;
            B12 = -conj(B21);
    
            grad[m,k,n] = real( X[1,m,k]*Ψ[2,m,k]*B12 + X[2,m,k]*Ψ[1,m,k]*B21 )
    
            X[1,m,k], X[2,m,k] = P11[m,k]*X[1,m,k] + P21*X[2,m,k],
                                 P12*X[1,m,k] + P22[m,k]*X[2,m,k]
        end
    end
    return
end


function gradient_cpu_full(Ψ,X,ϕ, NSteps,NCycles,N, P11,P22,Q21,A21,w, grad)
    for (m,k) in Tuple.(CartesianIndices(w))
        X[1,m,k] = 0.0
        X[2,m,k] = 0.0

        for cycle in 1:NCycles
            X[1,m,k] += -2 * w[m,k] * real(Ψ[1,m,k]) 

            for r in 1:NSteps
                n = N-(r + (cycle-1)*NSteps )+1
                expphi = exp(im*ϕ[n])
                P21 = Q21[m,k] * expphi;
                P12 = -conj(P21);
        
                Ψ[1,m,k], Ψ[2,m,k] = P22[m,k]*Ψ[1,m,k] - P12     *Ψ[2,m,k], 
                                    -P21    *Ψ[1,m,k]  + P11[m,k]*Ψ[2,m,k]
        
                B21 = A21[m,k] * expphi;
                B12 = -conj(B21);
        
                grad[m,k,n] = real( X[1,m,k]*Ψ[2,m,k]*B12 + X[2,m,k]*Ψ[1,m,k]*B21 )
        
                X[1,m,k], X[2,m,k] = P11[m,k]*X[1,m,k] + P21*X[2,m,k],
                                    P12*X[1,m,k] + P22[m,k]*X[2,m,k]
            end
        end
    end
    return
end

