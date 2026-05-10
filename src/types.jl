# This file is copyrighted under the latest version of the EUPL.
# Please see LICENCE file for your rights under this license.

"""
    FullPulse

Type of a full pulse. Currently just an alias for `Vector{Float64}`.
"""
FullPulse = Vector{Float64}

"""
    mutable struct XUR_Pulse

Type of an extended UR (XUR) pulse. 
"""
mutable struct XUR_Pulse
    phases::Vector{Float64}
    signs::Vector{Int64}
    shifts::Vector{Int64}
end
