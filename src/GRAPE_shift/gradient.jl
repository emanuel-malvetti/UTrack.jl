# This file is copyrighted under the latest version of the EUPL.
# Please see LICENCE file for your rights under this license.


function backpropagate_compute_gradient_shift(p::OptimizationParameters, s::OptStateShift, pc::UTrack.OptimizationPrecomputed)
    M = p.NRab
    K = p.NDet
    N = p.NSteps*p.NCycles
    
    grad2 = CuArray{Float64}(undef,M,K,p.NCycles) # move to OptStateShift

    @cuda threads=M blocks=K gradient_kernel_full_shift(s.Ψ, s.X, s.ϕ, s.σ, pc.Ω, pc.Δ, p.Δt, p.NSteps, p.NCycles, pc.P11,pc.P22,pc.Q21,pc.A21,pc.w, s.grad, grad2)
    s.δϕ .= view(sum(s.grad,dims=[1,2]),1,1,:) ./ p.NCycles
    s.δσ .= view(sum(grad2,dims=[1,2]),1,1,:) ./ p.NCycles .* p.Δt

    return 
end

function gradient_kernel_full_shift(Ψ, X, ϕ, σ, Ω, Δ, Δt, NSteps,NCycles, P11,P22,Q21,A21,w, grad, grad2)
    m = threadIdx().x
    k = blockIdx().x

    X[1,m,k] = 0.0
    X[2,m,k] = 0.0
    #grad[]=0 if necessary

    for cycle in NCycles:-1:1
        int_shift,frac_shift = divrem(σ[cycle],1.)
        int_shift = mod(Int64(int_shift),NSteps)

        X[1,m,k] += -2 * w[m,k] * real(Ψ[1,m,k])

        n = 1+int_shift + (cycle-1)*NSteps
        H11 = Δ[k]/2
        H22 = -H11
        H12 = Ω[m] * exp(-im*ϕ[n]) / 2
        H21 = conj(H12)

        grad2[m,k,cycle] = imag( X[1,m,k]*Ψ[1,m,k]*H11 + X[1,m,k]*Ψ[2,m,k]*H12 
                        + X[2,m,k]*Ψ[1,m,k]*H21 + X[2,m,k]*Ψ[2,m,k]*H22)

        if frac_shift != 0.0
            expphi = exp(im*ϕ[n])

            θ = sqrt(Ω[m]^2 + Δ[k]^2) * (frac_shift)*Δt
            s,c = sin(θ/2), cos(θ/2)
            α = (frac_shift)*Δt*s/θ
            p11 = c - im * α * Δ[k]
            p22 = conj(p11)
            q21 = -im * α * Ω[m]
            p21 = q21 * expphi
            p12 = -conj(p21)
            a21 = α * Ω[m] 

            Ψ[1,m,k], Ψ[2,m,k] = p22*Ψ[1,m,k] - p12*Ψ[2,m,k], - p21*Ψ[1,m,k] + p11*Ψ[2,m,k]

            B21 = a21 * expphi;
            B12 = -conj(B21);
    
            grad[m,k,n] = real( X[1,m,k]*Ψ[2,m,k]*B12 + X[2,m,k]*Ψ[1,m,k]*B21 )
    
            X[1,m,k], X[2,m,k] = p11*X[1,m,k] + p21*X[2,m,k], p12*X[1,m,k] + p22*X[2,m,k]
        else 
            grad[m,k,n] = 0
        end 

        for r in NSteps:-1:2
            n = mod1(r + int_shift,NSteps) + (cycle-1)*NSteps             
            expphi = exp(im*ϕ[n])
            P21 = Q21[m,k] * expphi;
            P12 = -conj(P21);
    
            Ψ[1,m,k], Ψ[2,m,k] = P22[m,k]*Ψ[1,m,k] - P12     *Ψ[2,m,k], 
                                -P21    *Ψ[1,m,k]  + P11[m,k]*Ψ[2,m,k]
    
            B21 = A21[m,k] * expphi;
            B12 = -conj(B21);
    
            grad[m,k,n] = real( X[1,m,k]*Ψ[2,m,k]*B12 + X[2,m,k]*Ψ[1,m,k]*B21 )
    
            X[1,m,k], X[2,m,k] = P11[m,k]*X[1,m,k] + P21*X[2,m,k], P12*X[1,m,k] + P22[m,k]*X[2,m,k]
        end

  
        n = 1+int_shift + (cycle-1)*NSteps
        expphi = exp(im*ϕ[n])

        θ = sqrt(Ω[m]^2 + Δ[k]^2) * (1-frac_shift)*Δt
        s,c = sin(θ/2), cos(θ/2)
        α = (1-frac_shift)*Δt*s/θ
        p11 = c - im * α * Δ[k]
        p22 = conj(p11)
        q21 = -im * α * Ω[m]
        p21 = q21 * expphi
        p12 = -conj(p21)
        a21 = α * Ω[m] 

        Ψ[1,m,k], Ψ[2,m,k] = p22*Ψ[1,m,k] - p12*Ψ[2,m,k], - p21*Ψ[1,m,k] + p11*Ψ[2,m,k]

        B21 = a21 * expphi;
        B12 = -conj(B21);

        grad[m,k,n] += real( X[1,m,k]*Ψ[2,m,k]*B12 + X[2,m,k]*Ψ[1,m,k]*B21 )

        X[1,m,k], X[2,m,k] = p11*X[1,m,k] + p21*X[2,m,k], p12*X[1,m,k] + p22*X[2,m,k]

        grad2[m,k,cycle] -= imag( X[1,m,k]*Ψ[1,m,k]*H11 + X[1,m,k]*Ψ[2,m,k]*H12 
                         + X[2,m,k]*Ψ[1,m,k]*H21 + X[2,m,k]*Ψ[2,m,k]*H22)

    end

    return
end

function finite_differences(p::OptimizationParameters, s::OptStateShift, pc::UTrack.OptimizationPrecomputed, h = 1e-8)
    N=p.NSteps*p.NCycles
    grad_ϕ = zeros(N)
    grad_σ = zeros(p.NCycles)
    c1 = propagate_state_compute_cost_shift(p,s,pc)
    
    for i in 1:N 
        CUDA.@allowscalar s.ϕ[i] += h
        c2 = propagate_state_compute_cost_shift(p,s,pc)
        CUDA.@allowscalar s.ϕ[i] -= h
        grad_ϕ[i] = (c2-c1)/h
    end

    for i in 1:p.NCycles
        CUDA.@allowscalar s.σ[i] += h
        c2 = propagate_state_compute_cost_shift(p,s,pc)
        CUDA.@allowscalar s.σ[i] -= h
        grad_σ[i] = (c2-c1)/h
    end

    grad_ϕ, grad_σ
end

#=
p = OptimizationParameters()
s = OptStateShift(p)
pc = UTrack.OptimizationPrecomputed(p)
ϕ = [randn() for _ in 1:p.NSteps*p.NCycles]
σ = [randn() for _ in 1:p.NCycles]
s.ϕ .= CuArray(ϕ)
s.σ .= CuArray(σ)
propagate_state_compute_cost_shift(p,s,pc)
backpropagate_compute_gradient_shift(p,s,pc)
fd = finite_differences(p,s,pc)[1]
gr = Array(s.δϕ)
fd./gr |> sort
=#