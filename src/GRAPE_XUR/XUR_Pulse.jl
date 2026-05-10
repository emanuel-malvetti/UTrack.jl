# This file is copyrighted under the latest version of the EUPL.
# Please see LICENCE file for your rights under this license.


# TODO explain
function angle_error(θ::Real)
    1-cos(θ)
end


"""
    closest_xur_pulse(p::OptimizationParameters, ϕ::FullPulse)

Construct an `XUR_Pulse` which is close to a given `FullPulse` `ϕ`.
"""
function closest_xur_pulse(p::OptimizationParameters, ϕ::UTrack.FullPulse) 
    n = p.NSteps
    candidates = [c for c in eachcol(UR_cycles(n))]

    shifts = zeros(Int64,p.NCycles)
    phases = zeros(Float64,p.NCycles)
    signs = zeros(Int64,p.NCycles)

    for i in 1:p.NCycles
        cycle = ϕ[(1:n) .+ (i-1)*n]
        phases[i] = cycle[1]

        dists = [sum(abs.(angle_error.(cycle-c .- (cycle[1])))) for c in candidates]
        signs[i],shifts[i] = sign_shift_from_t(findmin(dists)[2])
    end

    XUR_Pulse(phases,signs,shifts)
end 

"""
    random_xur_pulse(p::OptimizationParameters)

Construct a random `XUR_Pulse`.
"""
function random_xur_pulse(p::OptimizationParameters)
    return XUR_Pulse(2π*rand(p.NCycles), rand([-1,1],p.NCycles), rand(0:Int(p.NSteps-1),p.NCycles))
end


"""
    get_full_pulse(p::OptimizationParameters, xur::XUR_Pulse)

Return the `FullPulse` represented by the given `XUR_Pulse`.
"""
function get_full_pulse(p::OptimizationParameters, xur::XUR_Pulse)
    ϕ = zeros(Float64, p.NCycles*p.NSteps)
    for i in 1:p.NCycles 
        view_cycle(p,ϕ,i) .= UR_cycle(p.NSteps, xur.shifts[i], xur.signs[i]) .+ xur.phases[i]
    end
    renormalize_pulses!([ϕ])
    ϕ
end


# genetic update functions

function random_step_xur!(p, xur, xur2)
    xur2.phases .= xur.phases
    xur2.signs .= xur.signs
    xur2.shifts .= xur.shifts
    #xur2.phases[rand(1:p.NCycles)] = 2π*rand()
    for _ in 1:5
        xur2.signs[rand(1:p.NCycles)] *= -1
        xur2.shifts[rand(1:p.NCycles)] = rand(0:Int64(p.NSteps/2-1))
    end
    xur2
end

function mutate_xur(p,xur)
    xur2=random_xur_pulse(p)
    random_step_xur!(p, xur, xur2)
    return xur2
end

function mutate_xur_good(p,pcx,xur)
    new_xurs = [mutate_xur(p,xur) for _ in 1:1000]
    i = findmin([cost(p,pcx,xur) for xur in new_xurs])[2]
    return new_xurs[i]
end

function mutate_xur_many(p,pcx,xurs)
    return [mutate_xur_good(p,pcx,xur) for xur in xurs]
end

function recombine_xur(p,xur1,xur2)
    xur3=copy(xur1)
    xur4=copy(xur2)

    subset = randperm(p.NCycles)[1:min(p.NCycles,5)]
    for i in subset
        xur3.phases[i] = xur2.phases[i]
        xur3.signs[i] = xur2.signs[i]
        xur3.shifts[i] = xur2.shifts[i]
        xur4.phases[i] = xur1.phases[i]
        xur4.signs[i] = xur1.signs[i]
        xur4.shifts[i] = xur1.shifts[i]
    end

    return xur3,xur4
end

function recombine_xur_good(p,pcx,xur1,xur2)
    out1,out2 = recombine_xur(p,xur1,xur2)
    val = cost(p,pcx,out1) + cost(p,pcx,out2)
    for _ in 1:100
        tmp1,tmp2 = recombine_xur(p,xur1,xur2)
        val2 = cost(p,pcx,tmp1) + cost(p,pcx,tmp2)
        if val2 < val 
            val - val2 
            out1,out2 = tmp1,tmp2
        end
    end
    return out1,out2
end

function recombine_xur_many(p, pcx, xurs, T)
    out_xurs = XUR_Pulse[] 

    for i in 1:Int64(floor(length(xurs)/2))
        ϕ1,ϕ2 = recombine_xur_good(p, pcx, xurs[2*i-1], xurs[2*i])
        push!(out_xurs, ϕ1)
        push!(out_xurs, ϕ2)
    end

    if mod(length(xurs),2) == 1
        push!(out_xurs, xurs[end])
    end

    return out_xurs
end

function Base.copy(xur::XUR_Pulse)
    XUR_Pulse(copy(xur.phases), copy(xur.signs), copy(xur.shifts))
end

