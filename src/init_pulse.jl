# This file is copyrighted under the latest version of the EUPL.
# Please see LICENCE file for your rights under this license.


"""
    @enum PulseType

Pulse types used for initialization.
"""
@enum PulseType begin 
    Constant 
    Rand 
    UR
end

"""
    UR_pulse(p::OptimizationParameters, ϕ2::Float=0, sign::Integer=1)

Computes a periodic UR pulse sequence with cycle length `p.NSteps` and number of cycles `p.NCycles`.
The remaining arguments are as in `UR_cycle`.
"""
function UR_pulse(p::OptimizationParameters, k::Integer=0, sign::Integer=1)
    cycle = UR_cycle(p.NSteps, k, sign)
    return repeat(cycle, p.NCycles);
end

"""
    initialize_pulse(p::OptimizationParameters, type::PulseType)

Initialize a single pulse of given pulse type according to the given parameters.
"""
function initialize_pulse(p::OptimizationParameters, type::PulseType)
    if type == Constant
        return ones(p.NSteps*p.NCycles)/100
    elseif type == Rand
        return randn(p.NSteps*p.NCycles)
    elseif type == UR
        return UR_pulse(p)
    end
end


"""
    initialize_pulses(p::OptimizationParameters, type::PulseType=Rand)

Initialize population of pulses of given pulse type according to the given parameters.
"""
function initialize_pulses(p::OptimizationParameters, type::PulseType=Rand)
    return [initialize_pulse(p, type) for _ in 1:p.NPulses]
end

