# This file is copyrighted under the latest version of the EUPL.
# Please see LICENCE file for your rights under this license.

"""
    UR_cycle(n::Integer, ϕ2::Float=nothing, sign::Integer=1)

Computes the UR pulse sequence (single cycle) of length `n` with first phase zero, 
second phase equal to `2πk/n`, and sign given by `sign`.
We require k=0,...,n-1, and s=±1.
"""
function UR_cycle(n::Integer, k::Integer=0, sign::Integer=1)
    if n < 4 || mod(n,2) != 0
        throw(DomainError(n, "Cycle length n must be at least 4 and even."))
    elseif !(sign == 1 || sign == -1)
        throw(DomainError(sign, "sign must be +1 or -1."))
    end

    if mod(n,4) == 0
        m = n/4
        Φ = sign * π/m
    else
        m = (n-2)/4
        Φ = sign*(2π*m)/(2*m+1)
    end

    cycle = zeros(n)

    for i = 1:n
        cycle[i] = (i-1)*(i-2)/2*Φ + (i-1)*2π*k/n
        cycle[i] = mod(cycle[i],2π)
    end

    return cycle
end


"""
    UR_cycles(n::Integer)

Returns a Matrix of size n x 2n containing all UR_n pulses as columns. 
"""
function UR_cycles(n::Integer)
    cycles = [UR_cycle(n,k,sign) for k in 0:(n-1) for sign in [1,-1]]
    hcat(cycles...)
end

# helper functions to convert between indices
function t_from_sign_shift(sign::Integer, shift::Integer)
    return Int64(2*shift + (3-sign)/2)
end

function sign_shift_from_t(t::Integer)
    s = Int64((-1)^(t+1))
    return s, Int64(round((t-(3-s)/2)/2))
end