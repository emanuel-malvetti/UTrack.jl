# This file is copyrighted under the latest version of the EUPL.
# Please see LICENCE file for your rights under this license.


"""
    write_parameters(filename::AbstractString, p::OptimizationParameters)

Write optimization parameters to JSON file.
"""
function write_parameters(filename::AbstractString, p::OptimizationParameters)
    open(filename, "w") do io
        JSON3.pretty(io, p)
    end
end

"""
    read_parameters(filename::AbstractString)

Read optimization parameters from JSON file.
"""
function read_parameters(filename::AbstractString)
    return JSON3.read(filename, OptimizationParameters)
end


"""
    write_all_pulses_full(filename::AbstractString, pulses::AbstractArray{FullPulse})

Write a list of pulses of equal length to a CSV file.
"""
function write_all_pulses_full(filename::AbstractString, pulses::AbstractArray{FullPulse})
    CSV.write(filename, Tables.table(hcat(pulses...)), header=false)
end

"""
    read_all_pulses_full(filename::AbstractString)

Read pulses from CSV file.
"""
function read_all_pulses_full(filename::AbstractString)
    mat = CSV.File(filename, header=false) |> Tables.matrix
    return [mat[:,i] for i in axes(mat,2)]
end

"""
    write_all_pulses_xur(filename::String, p::OptimizationParameters, xur_pulses::AbstractArray{XUR_Pulse})

Write a list of XUR pulses (converted to full pulses) of equal length to a CSV file.
"""
function write_all_pulses_xur(filename::AbstractString, p::OptimizationParameters, xur_pulses::AbstractArray{XUR_Pulse})
    write_all_pulses_full(filename, [get_full_pulse(p, xur) for xur in xur_pulses])
end

"""
    read_all_pulses_xur(filename::AbstractString, p::OptimizationParameters)

Read pulses from CSV file and return as XUR pulses.
"""
function read_all_pulses_xur(filename::AbstractString, p::OptimizationParameters)
    pulses = read_all_pulses_full(filename)
    return [closest_xur_pulse(p, ϕ) for ϕ in pulses]
end


"""
    write_pulse_matlab(filename::AbstractString, pulse::FullPulse)

Write single pulse to MATLAB file. The variable name is set to `phi`.
"""
function write_pulse_matlab(filename::AbstractString, pulse::FullPulse)
    file = matopen(filename, "w")
    write(file, "phi", pulse)
    close(file)
end

"""
    read_pulse_matlab(filename::AbstractString)

Read single pulse from MATLAB file.
"""
function read_pulse_matlab(filename::AbstractString)
    var=matread(filename)
    var["phi"][:,1]
end 
