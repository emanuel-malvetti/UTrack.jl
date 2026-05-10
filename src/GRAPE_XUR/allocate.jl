# This file is copyrighted under the latest version of the EUPL.
# Please see LICENCE file for your rights under this license.


"""
    OptimizationStateXUR {T <: AbstractArray{Float64}, S <: AbstractArray{ComplexF64}, R <: AbstractArray{Int64}}

Stores the state for the cycle GRAPE optimization.
"""
struct OptimizationStateXUR{T <: AbstractArray{Float64}, S <: AbstractArray{ComplexF64}, R <: AbstractArray{Int64}}
    τ::R     # pulse cycle types
    Ψ::S     # state
    X::S     # costate
    Φ::T     # pulse
    δΦ::T    # pulse gradient
    cost::T  # temporary storage for cost computation
    grad::T  # temporary storage for gradient computation

    @doc """
        OptimizationStateXUR(p::OptimizationParameters)

    Create an empty state according to the given parameters.
    """
    function OptimizationStateXUR(p::OptimizationParameters)
        M = p.NRab
        K = p.NDet
        N = p.NCycles
        
        if CUDA.functional()
            τ=CuArray{Int64}(undef,N)
            Ψ=CuArray{ComplexF64}(undef,2,M,K)
            X=CuArray{ComplexF64}(undef,2,M,K)
            Φ=CuArray{Float64}(undef,N)
            δΦ=CuArray{Float64}(undef,N)
            cost=CuArray{Float64}(undef,M,K)
            grad=CuArray{Float64}(undef,M,K,N)
            new{CuArray{Float64},CuArray{ComplexF64},CuArray{Int64}}(τ,Ψ,X,Φ,δΦ,cost,grad)
        else
            τ=Array{Int64}(undef,N)
            Ψ=Array{ComplexF64}(undef,2,M,K)
            X=Array{ComplexF64}(undef,2,M,K)
            Φ=Array{Float64}(undef,N)
            δΦ=Array{Float64}(undef,N)
            cost=Array{Float64}(undef,M,K)
            grad=Array{Float64}(undef,M,K,N)
            new{Array{Float64},Array{ComplexF64},Array{Int64}}(τ,Ψ,X,Φ,δΦ,cost,grad)
        end
    end
end

"""
    struct OptimizationPrecomputedXUR{T <: AbstractArray{Float64}, S <: AbstractArray{ComplexF64}}

Stores some information that only has to be computed once and can be shared among all optimizations 
using the same parameters. Uses CUDA if available. 
"""
struct OptimizationPrecomputedXUR{T <: AbstractArray{Float64}, S <: AbstractArray{ComplexF64}}
    Ω::T # Rabi frequency deviations
    Δ::T # detunings
    w::T # weights

    # some precomputable matrix elements
    P11::S 
    P22::S
    Q21::S
    A21::S

    @doc """
        OptimizationPrecomputedXUR(p::OptimizationParameters)

    Precomputes some information for parameters `p` for the cycle optimization.
    """
    function OptimizationPrecomputedXUR(p::OptimizationParameters)
        M = p.NRab
        K = p.NDet
        T = 2*p.NSteps 

        if CUDA.functional()
            Ω = CuArray{Float64}(p.Rabi * range(1.0-p.RabiFDev, 1.0+p.RabiFDev, M))
            Δ = CuArray{Float64}(p.Rabi * range(-p.Detuning, p.Detuning, K))
            w = CuArray{Float64}(undef,M,K)

            @cuda threads=M blocks=K init_weights_kernel(w,Ω,Δ,p.Detuning*p.sigDet,p.RabiFDev*p.sigRab) 
            w ./= sum(w)

            P11 = CuArray{ComplexF64}(undef,M,K,T)
            P22 = CuArray{ComplexF64}(undef,M,K,T)
            Q21 = CuArray{ComplexF64}(undef,M,K,T)
            A21 = CuArray{ComplexF64}(undef,M,K,T)

            ϕ = CuArray(UR_cycles(p.NSteps)) # make argument
            @cuda threads=M blocks=K,T init_propagator_kernel_xur(Ω,Δ,ϕ, P11,P22,Q21,A21, p.Δt, T)

            new{CuArray{Float64},CuArray{ComplexF64}}(Ω,Δ,w,P11,P22,Q21,A21)
        else
            Ω = Array{Float64}(p.Rabi * range(1.0-p.RabiFDev, 1.0+p.RabiFDev, M))
            Δ = Array{Float64}(p.Rabi * range(-p.Detuning, p.Detuning, K))
            w = Array{Float64}(undef,M,K)

            init_weights_cpu(w,Ω,Δ,p.Detuning*p.sigDet,p.RabiFDev*p.sigRab) 
            w ./= sum(w)

            P11 = Array{ComplexF64}(undef,M,K,T)
            P22 = Array{ComplexF64}(undef,M,K,T)
            Q21 = Array{ComplexF64}(undef,M,K,T)
            A21 = Array{ComplexF64}(undef,M,K,T)

            ϕ = UR_cycles(p.NSteps)
            init_propagator_cpu_xur(Ω,Δ,ϕ, P11,P22,Q21,A21, p.Δt, T)

            new{Array{Float64},Array{ComplexF64}}(Ω,Δ,w, P11,P22,Q21,A21)
        end
    end 
end

function init_propagator_kernel_xur(Ω,Δ,ϕ, P11,P22,Q21,A21, Δt, T)
    m = threadIdx().x
    k = blockIdx().x
    t = blockIdx().y

    α::ComplexF64=1
    β::ComplexF64=0
    for n in 1:div(T,2)
        θ = sqrt(Ω[m]^2 + Δ[k]^2) * Δt
        s,c = sin(θ/2), cos(θ/2)
        r = im*s*Δt/θ
        γ = c-r*Δ[k]
        δ = -r*(cos(ϕ[n,t])+im*sin(ϕ[n,t]))*Ω[m]
        α,β = α*γ-conj(δ)*β, δ*α+conj(γ)*β
    end

    P11[m,k,t] = α
    P22[m,k,t] = conj(P11[m,k,t])
    Q21[m,k,t] = β
    A21[m,k,t] = im * Q21[m,k,t]
    
    return
end

function init_propagator_cpu_xur(Ω,Δ,ϕ, P11,P22,Q21,A21, Δt, T)
    for (m,k,t) in Tuple.(CartesianIndices(P11))
        α::ComplexF64=1
        β::ComplexF64=0
        for n in 1:div(T,2)
            θ = sqrt(Ω[m]^2 + Δ[k]^2) * Δt
            s,c = sin(θ/2), cos(θ/2)
            r = im*s*Δt/θ
            γ = c-r*Δ[k]
            δ = -r*(cos(ϕ[n,t])+im*sin(ϕ[n,t]))*Ω[m]
            α,β = α*γ-conj(δ)*β, δ*α+conj(γ)*β
        end

        P11[m,k,t] = α
        P22[m,k,t] = conj(P11[m,k,t])
        Q21[m,k,t] = β
        A21[m,k,t] = im * Q21[m,k,t]
    end
    return
end


"""
    set_pulse(s::OptimizationStateXUR{T,S,R}, Φ) where {T<: CuArray, S <: CuArray, R <: CuArray}

Assign the pulse phases `Φ` to the state `s`. Automatically converts the pulse phases to a `CuArray`.
"""
function set_pulse(s::OptimizationStateXUR{T,S,R}, Φ) where {T<: CuArray, S <: CuArray, R <: CuArray}
    s.Φ .= CuArray(Φ)
    return
end

"""
    set_pulse(s::OptimizationStateXUR{T,S,R}, Φ) where {T<: Array, S <: Array, R <: Array}

Assign the pulse phases `Φ` to the state `s`.
"""
function set_pulse(s::OptimizationStateXUR{T,S,R}, Φ) where {T<: Array, S <: Array, R <: Array}
    s.Φ .= Φ
    return
end

"""
    set_types(s::OptimizationStateXUR{T,S,R}, τ) where {T<: CuArray, S <: CuArray, R <: CuArray}

Sets the UR cycle types of the state `s`. Automatically converts to a `CuArray`.
"""
function set_types(s::OptimizationStateXUR{T,S,R}, τ) where {T<: CuArray, S <: CuArray, R <: CuArray}
    s.τ .= CuArray(τ)
    return
end

"""
    set_types(s::OptimizationStateXUR{T,S,R}, τ) where {T<: Array, S <: Array, R <: Array}

Sets the UR cycle types of the state `s`. 
"""
function set_types(s::OptimizationStateXUR{T,S,R}, τ) where {T<: Array, S <: Array, R <: Array}
    s.τ .= τ
    return
end

"""
    get_gradient(s::OptimizationStateXUR, δΦ)

Store the gradient from state `s` in `δΦ`.
"""
function get_gradient(s::OptimizationStateXUR, δΦ)
    δΦ .= Array(s.δΦ)
    return
end