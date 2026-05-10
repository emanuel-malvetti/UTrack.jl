# This file is copyrighted under the latest version of the EUPL.
# Please see LICENCE file for your rights under this license.


"""
    OptimizationState{T <: AbstractArray{Float64}, S <: AbstractArray{ComplexF64}}

Stores the state for the full GRAPE optimization. Uses CUDA if `CUDA.functional() == true`. 
"""
struct OptimizationState{T <: AbstractArray{Float64}, S <: AbstractArray{ComplexF64}}
    Ψ::S    # state
    X::S    # costate
    ϕ::T    # pulse
    δϕ::T   # pulse gradient
    cost::T # temporary storage for cost computation
    grad::T # temporary storage for gradient computation

    @doc """
        OptimizationState(p::OptimizationParameters)

    Create an empty state according to the given parameters.
    """
    function OptimizationState(p::OptimizationParameters)
        M = p.NRab
        K = p.NDet
        N = p.NSteps*p.NCycles

        if CUDA.functional()
            Ψ=CuArray{ComplexF64}(undef,2,M,K)
            X=CuArray{ComplexF64}(undef,2,M,K)
            ϕ=CuArray{Float64}(undef,N)
            δϕ=CuArray{Float64}(undef,N)
            cost=CuArray{Float64}(undef,M,K)
            grad=CuArray{Float64}(undef,M,K,N)
            new{CuArray{Float64},CuArray{ComplexF64}}(Ψ,X,ϕ,δϕ,cost,grad)
        else
            Ψ=Array{ComplexF64}(undef,2,M,K)
            X=Array{ComplexF64}(undef,2,M,K)
            ϕ=Array{Float64}(undef,N)
            δϕ=Array{Float64}(undef,N)
            cost=Array{Float64}(undef,M,K)
            grad=Array{Float64}(undef,M,K,N)
            new{Array{Float64},Array{ComplexF64}}(Ψ,X,ϕ,δϕ,cost,grad)
        end
    end
end

"""
    set_pulse(s::OptimizationState{T,S}, ϕ::FullPulse) where {T<: CuArray, S <: CuArray}

Assign the pulse `ϕ` to the state `s`. Automatically converts the pulse to a `CuArray`.
"""
function set_pulse(s::OptimizationState{T,S}, ϕ::FullPulse) where {T<: CuArray, S <: CuArray}
    s.ϕ .= CuArray(ϕ)
    return
end

"""
    set_pulse(s::OptimizationState{T,S}, ϕ::FullPulse) where {T<: Array, S <: Array}

Assign the pulse `ϕ` to the state `s`.
"""
function set_pulse(s::OptimizationState{T,S}, ϕ::FullPulse) where {T<: Array, S <: Array}
    s.ϕ .= ϕ
    return
end

"""
    get_gradient(s::OptimizationState, δϕ)

Store the gradient from state `s` in `δϕ`.
"""
function get_gradient(s::OptimizationState, δϕ) 
    δϕ .= Array(s.δϕ)
    return
end


"""
    OptimizationPrecomputed{T <: AbstractArray{Float64}, S <: AbstractArray{ComplexF64}}

Stores some information that only has to be computed once and can be shared among all optimizations 
using the same parameters. Uses CUDA if available. 
"""
struct OptimizationPrecomputed{T <: AbstractArray{Float64}, S <: AbstractArray{ComplexF64}}
    Ω::T # Rabi frequency deviations
    Δ::T # detunings
    w::T # weights

    # some precomputable matrix elements
    P11::S 
    P22::S
    Q21::S
    A21::S

    @doc """
        OptimizationPrecomputed(p::OptimizationParameters)

    Precomputes some information for parameters `p`.
    """
    function OptimizationPrecomputed(p::OptimizationParameters)
        M = p.NRab
        K = p.NDet

        if CUDA.functional()
            Ω = CuArray{Float64}(p.Rabi * range(1.0-p.RabiFDev, 1.0+p.RabiFDev, M))
            Δ = CuArray{Float64}(p.Rabi * range(-p.Detuning, p.Detuning, K))
            w = CuArray{Float64}(undef,M,K)

            @cuda threads=M blocks=K init_weights_kernel(w,Ω,Δ,p.Detuning*p.sigDet,p.RabiFDev*p.sigRab) 
            w ./= sum(w)

            P11 = CuArray{ComplexF64}(undef,M,K)
            P22 = CuArray{ComplexF64}(undef,M,K)
            Q21 = CuArray{ComplexF64}(undef,M,K)
            A21 = CuArray{ComplexF64}(undef,M,K)

            @cuda threads=M blocks=K init_propagator_kernel(Ω,Δ,P11,P22,Q21,A21,p.Δt)

            new{CuArray{Float64},CuArray{ComplexF64}}(Ω,Δ,w,P11,P22,Q21,A21)
        else
            Ω = Array{Float64}(p.Rabi * range(1.0-p.RabiFDev, 1.0+p.RabiFDev, M))
            Δ = Array{Float64}(p.Rabi * range(-p.Detuning, p.Detuning, K))
            w = Array{Float64}(undef,M,K)

            init_weights_cpu(w,Ω,Δ,p.Detuning*p.sigDet,p.RabiFDev*p.sigRab) 
            w ./= sum(w)

            P11 = Array{ComplexF64}(undef,M,K)
            P22 = Array{ComplexF64}(undef,M,K)
            Q21 = Array{ComplexF64}(undef,M,K)
            A21 = Array{ComplexF64}(undef,M,K)

            init_propagator_cpu(Ω,Δ,P11,P22,Q21,A21,p.Δt)

            new{Array{Float64},Array{ComplexF64}}(Ω,Δ,w,P11,P22,Q21,A21)
        end
    end
end

function init_weights_kernel(w,Ω,Δ,σΩ,σΔ)
    m = threadIdx().x
    k = blockIdx().x

    w[m,k] = 1 + 100*exp( -(Ω[m]-1.0)^2 / (2*σΩ^2) - Δ[k]^2 / (2*σΔ^2) )

    return
end

function init_weights_cpu(w,Ω,Δ,σΩ,σΔ)
    for (m,k) in Tuple.(CartesianIndices(w))
        w[m,k] = 1 + 100*exp( -(Ω[m]-1.0)^2 / (2*σΩ^2) - Δ[k]^2 / (2*σΔ^2) )
    end
    return
end

function init_propagator_kernel(Ω,Δ,P11,P22,Q21,A21,Δt)
    m = threadIdx().x
    k = blockIdx().x

    θ = sqrt(Ω[m]^2 + Δ[k]^2) * Δt
    s,c = sin(θ/2), cos(θ/2)
    α = Δt*s/θ

    P11[m,k] = c - im * α * Δ[k]
    P22[m,k] = conj(P11[m,k])
    Q21[m,k] = -im * α * Ω[m]
    A21[m,k] = α * Ω[m] 
    
    return
end

function init_propagator_cpu(Ω,Δ,P11,P22,Q21,A21,Δt)

    for (m,k) in Tuple.(CartesianIndices(P11))
        θ = sqrt(Ω[m]^2 + Δ[k]^2) * Δt
        s,c = sin(θ/2), cos(θ/2)
        α = Δt*s/θ

        P11[m,k] = c - im * α * Δ[k]
        P22[m,k] = conj(P11[m,k])
        Q21[m,k] = -im * α * Ω[m]
        A21[m,k] = α * Ω[m] 
    end 

    return
end
