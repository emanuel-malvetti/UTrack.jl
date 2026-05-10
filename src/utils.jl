# This file is copyrighted under the latest version of the EUPL.
# Please see LICENCE file for your rights under this license.

"""
    renormalize_pulses!(pulses::AbstractArray{FullPulse})

Normalizes an array of full pulses to have phases in [0,2π).
"""
function renormalize_pulses!(pulses::AbstractArray{FullPulse})
    for ϕ in pulses 
        for i in eachindex(ϕ)
            ϕ[i] = mod(ϕ[i],2π)
        end
    end
end


"""
    renormalize_pulses!(pulses::AbstractArray{XUR_Pulse})

Normalizes an array of XUR pulses to have phases in [0,2π).
"""
function renormalize_pulses!(pulses::AbstractArray{XUR_Pulse})
    for ϕ in pulses 
        for i in eachindex(ϕ.phases)
            ϕ.phases[i] = mod(ϕ.phases[i],2π)
        end
    end
end

#= Currently unused, can be used for testing purposes

"""
    hamiltonian(Ω,Δ,ϕ)

Compute the Hamiltonian for a given amplitude deviation `Ω`, detuning `Δ`, and control phase `ϕ`.
"""
function hamiltonian(Ω,Δ,ϕ)
    σx = ComplexF64[0 1; 1 0]
    σy = ComplexF64[0 -im; im 0]
    σz = ComplexF64[1 0; 0 -1]
    Ω * (cos(ϕ) * σx/2 + sin(ϕ) * σy/2) + Δ * σz/2
end


"""
    propagator(Ω,Δ,ϕ,Δt)

Compute the propagator corresponding to the Hamiltonian `hamiltonian(Ω,Δ,ϕ)` applied for a time step `Δt`.
"""
function propagator(Ω,Δ,ϕ,Δt)
    exp(-im * hamiltonian(Ω,Δ,ϕ) * Δt)
end


"""
    precompute_propagator_elements(Ω,Δ,ϕ,Δt)

TBW
"""
function precompute_propagator_elements(Ω,Δ,ϕ,Δt)
    U = Float64[1 0; 0 1]
    for n in eachindex(ϕ)
        U = propagator(Ω,Δ,ϕ[n],Δt)*U
    end
    U[:,1]
end

# Faster versions of the above functions

function propagator_fast(Ω,Δ,ϕ,Δt)
    A = ComplexF64[1 0; 0 1]
    θ = sqrt(Ω^2 + Δ^2) * Δt
    s,c = sin(θ/2), cos(θ/2)
    r = im*s*Δt/θ
    @inbounds A[1,1]=c-r*Δ
    @inbounds A[2,1]=-r*(cos(ϕ)+im*sin(ϕ))*Ω
    @inbounds A[1,2]=-r*(cos(ϕ)-im*sin(ϕ))*Ω
    @inbounds A[2,2]=c+r*Δ
    A
end

function precompute_propagator_elements_fast(Ω,Δ,ϕ,Δt)
    α::ComplexF64=1
    β::ComplexF64=0
    for n in eachindex(ϕ)
        θ = sqrt(Ω^2 + Δ^2) * Δt
        s,c = sin(θ/2), cos(θ/2)
        r = im*s*Δt/θ
        γ = c-r*Δ
        δ = -r*(cos(ϕ[n])+im*sin(ϕ[n]))*Ω
        α,β = α*γ-conj(β)*δ, β*γ+conj(α)*δ
    end
    [α,β]
end

=#
