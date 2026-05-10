# This file is copyrighted under the latest version of the EUPL.
# Please see LICENCE file for your rights under this license.


function propagate_state_compute_cost_shift(p::OptimizationParameters, s::OptStateShift, pc::UTrack.OptimizationPrecomputed)
    M = p.NDet
    K = p.NRab

    @cuda threads=M blocks=K propagate_kernel_full_shift(s.Ψ, s.ϕ, s.σ, pc.Ω, pc.Δ, p.Δt, p.NSteps, p.NCycles, s.cost, pc.w, pc.Q21, pc.P11, pc.P22)
    
    return sum(s.cost)/p.NCycles
end

function propagate_kernel_full_shift(Ψ, ϕ, σ, Ω, Δ, Δt, NSteps, NCycles, cost, w, Q21, P11, P22)
    m = threadIdx().x
    k = blockIdx().x

    Ψ[1,m,k]=1.0
    Ψ[2,m,k]=0.0
    cost[m,k]=0.0


    for cycle in 1:NCycles
        int_shift,frac_shift = divrem(σ[cycle],1.)
        #int_shift = Int64(int_shift)
        int_shift = mod(Int64(int_shift),NSteps)

        # apply ϕ[1+int_shift] for time (1-frac_shift)*Δt
        #H = Ω[m] * (cos(ϕ[1+int_shift]) * σx/2 + sin(ϕ[1+int_shift]) * σy/2) + Δ[k] * σz/2
        #view(Ψ,:,m,k) = exp(-im * H * (1-frac_shift)*Δt) * view(Ψ,:,m,k)
        θ = sqrt(Ω[m]^2 + Δ[k]^2) * (1-frac_shift)*Δt
        s,c = sin(θ/2), cos(θ/2)
        α = (1-frac_shift)*Δt*s/θ
        p11 = c - im * α * Δ[k]
        p22 = conj(p11)
        q21 = -im * α * Ω[m]
        p21 = q21*exp(im*ϕ[1+int_shift + (cycle-1)*NSteps])
        p12 = -conj(p21)
        Ψ[1,m,k], Ψ[2,m,k] = p11*Ψ[1,m,k] + p12*Ψ[2,m,k],  p21*Ψ[1,m,k] + p22*Ψ[2,m,k]

        for r in 2:NSteps
            n = mod1(r + int_shift,NSteps) + (cycle-1)*NSteps 
            P21 = Q21[m,k]*exp(im*ϕ[n])
            P12 = -conj(P21)
            Ψ[1,m,k], Ψ[2,m,k] = P11[m,k]*Ψ[1,m,k] + P12*Ψ[2,m,k],  P21*Ψ[1,m,k] + P22[m,k]*Ψ[2,m,k]
        end

        # apply ϕ[1+int_shift] for time (frac_shift)*Δt
        #H=Ω[m] * (cos(ϕ[1+int_shift]) * σx/2 + sin(ϕ[1+int_shift]) * σy/2) + Δ[k] * σz/2
        #view(Ψ,:,m,k) = exp(-im * H * (frac_shift)*Δt) * view(Ψ,:,m,k)
        
        if frac_shift != 0.0
            θ = sqrt(Ω[m]^2 + Δ[k]^2) * (frac_shift)*Δt
            s,c = sin(θ/2), cos(θ/2)
            α = (frac_shift)*Δt*s/θ
            p11 = c - im * α * Δ[k]
            p22 = conj(p11)
            q21 = -im * α * Ω[m]
            p21 = q21*exp(im*ϕ[1+int_shift + (cycle-1)*NSteps])
            p12 = -conj(p21)
            Ψ[1,m,k], Ψ[2,m,k] = p11*Ψ[1,m,k] + p12*Ψ[2,m,k],  p21*Ψ[1,m,k] + p22*Ψ[2,m,k]
        end 

        cost[m,k] += real(Ψ[1,m,k])^2
    end

    cost[m,k] = w[m,k] * (NCycles - cost[m,k])

    return
end

function shift_cost(p,ϕ)
    s = UTrack.OptimizationState(p)
    pc = UTrack.OptimizationPrecomputed(p)
    out = Float64[]
    for n in 0:(p.NSteps-1)
        for i in 1:p.NCycles
            UTrack.view_cycle(p,s.ϕ,i) .= CuArray(circshift(UTrack.view_cycle(p,ϕ,i),-n))
        end
        
        push!(out,UTrack.propagate_state_compute_cost(p,s,pc))
    end
    out
end
 
function test_propagate_kernel_full_shift(p,ϕ,samples=1000)
    out = Float64[]
    #ϕ = [randn() for _ in 1:p.NSteps*p.NCycles]#UTrack.UR_pulse(p)
    pc = UTrack.OptimizationPrecomputed(p)
    s = OptStateShift(p)
    s.ϕ .= CuArray(ϕ)
    for i in 0.0:p.NSteps/samples:p.NSteps
        s.σ .= CuArray([(i) for _ in 1:p.NCycles])
        push!(out,propagate_state_compute_cost_shift(p,s,pc))
    end
    out
end

#plot(test_propagate_kernel_full_shift(p,ϕ))
#savefig("test3.pdf")

function test_propagate_kernel_full_shift_2d(p,ϕ)
    N=100
    out = zeros(N,N)
    pc = UTrack.OptimizationPrecomputed(p)
    s = OptStateShift(p)
    s.ϕ .= CuArray(ϕ)
    for i in 1:N
        for j in 1:N
            s.σ .= CuArray([0.0 for k in 1:p.NCycles])
            CUDA.@allowscalar s.σ[1] = 0.
            CUDA.@allowscalar s.σ[2] = (5.0 *i/N)
            CUDA.@allowscalar s.σ[3] = 0.
            CUDA.@allowscalar s.σ[4] = (5.0 *j/N)
            out[i,j] = propagate_state_compute_cost_shift(p,s,pc)
        end
    end
    out
end

#=
p = OptimizationParameters(NSteps=100,NCycles=4,Δt=π/10)
s = OptStateShift(p)
pc = UTrack.OptimizationPrecomputed(p)
#ϕ = [sin(2π * n/p.NSteps) for n in 1:p.NSteps*p.NCycles]
#ϕ = [randn() for _ in 1:p.NSteps*p.NCycles]
data=test_propagate_kernel_full_shift_2d(p,ϕ)
contour(1:100, 1:100,data)
savefig("contour-shift-non-smooth.pdf")
=#

function smooth_steps(x)
    sin(π/2*(mod(x,1.)))^2 + floor(x)
end

function smooth_steps_inv(x)
    2/π * asin(sqrt(mod(x,1.))) + floor(x)
end

function smooth_steps_derv(x)
    π/2 * abs(sin(π*x))
end