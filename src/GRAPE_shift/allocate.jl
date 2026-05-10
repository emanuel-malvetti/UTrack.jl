# This file is copyrighted under the latest version of the EUPL.
# Please see LICENCE file for your rights under this license.


struct OptStateShift
    Ψ::CuArray{ComplexF64}  # state
    X::CuArray{ComplexF64}  # costate
    ϕ::CuArray{Float64}     # pulse phases
    δϕ::CuArray{Float64}    # pulse gradient
    σ::CuArray{Float64}     # shifts
    δσ::CuArray{Float64}    # shift gradients
    cost::CuArray{Float64}  # temporary storage for cost computation
    grad::CuArray{Float64}  # temporary storage for gradient computation
end

function OptStateShift(p::OptimizationParameters)
    M = p.NRab
    K = p.NDet
    N = p.NSteps*p.NCycles
    
    Ψ=CuArray{ComplexF64}(undef,2,M,K)
    X=CuArray{ComplexF64}(undef,2,M,K)
    ϕ=CuArray{Float64}(undef,N)
    δϕ=CuArray{Float64}(undef,N)
    σ=CuArray{Float64}(undef,p.NCycles)
    δσ=CuArray{Float64}(undef,p.NCycles)
    
    cost=CuArray{Float64}(undef,M,K)
    grad=CuArray{Float64}(undef,M,K,N)

    OptStateShift(Ψ,X,ϕ,δϕ,σ,δσ,cost,grad)
end

