# This file is copyrighted under the latest version of the EUPL.
# Please see LICENCE file for your rights under this license.


# files and folders used for storing optimization results and plots
const PARAM_FILE = "/parameters.json"

"""
    setup_folder(folder::AbstractString, p::OptimizationParameters)

Creates folder structure in `folder` to store all output related to the optimization as in the
following example:

    folder
    ├── parameters.json
    ├── plots
    │   ├── final
    │   │   ├── cost-profile.pdf
    │   │   ├── full-pulse.pdf
    │   │   └── pulse-cycles
    │   │       ├── cycle-1.pdf
    │   │       ├── ...
    │   │       └── cycle-10.pdf
    │   ├── optimization-cycle
    │   │   ├── evolution.pdf
    │   │   └── generations
    │   │       ├── gen-1.pdf
    │   │       ├── ...
    │   │       └── gen-20.pdf
    │   └── optimization-full
    │       ├── evolution.pdf
    │       └── generations
    │           ├── gen-1.pdf
    │           ├── ...
    │           └── gen-15.pdf
    └── pulses
        ├── final
        │   └── optimal-pulse.mat
        ├── optimization-cycle
        │   └── generations
        │       ├── gen-0.csv
        │       ├── ...
        │       └── gen-20.csv
        └── optimization-full
            └── generations
                ├── gen-0.csv
                ├── ...
                └── gen-15.csv
"""
function setup_folder(folder::AbstractString, p::OptimizationParameters)
    @assert !isdir(folder)

    mkdir(folder)
    save_parameters(folder, p) # parameters.json

    mkdir("$folder/pulses/")
    mkdir("$folder/pulses/optimization-cycle/")
    mkdir("$folder/pulses/optimization-cycle/generations/") # gen-$i.pdf
    mkdir("$folder/pulses/optimization-full/")
    mkdir("$folder/pulses/optimization-full/generations/") # gen-$i.pdf
    mkdir("$folder/pulses/final/") # optimal-pulse.mat, optimal-pulse.csv

    mkdir("$folder/plots/")
    mkdir("$folder/plots/optimization-cycle/") # evolution.pdf
    mkdir("$folder/plots/optimization-cycle/generations/") # gen-$i.pdf
    mkdir("$folder/plots/optimization-full/") # evolution.pdf
    mkdir("$folder/plots/optimization-full/generations/") # gen-$i.pdf
    mkdir("$folder/plots/final/") # cost-profile.pdf, full-pulse.pdf, 
    mkdir("$folder/plots/final/pulse-cycles") # cycle-$i.pdf

    return
end

"""
    save_parameters(folder::AbstractString, p::OptimizationParameters)

Store the optimization parameters `p` in the `folder`.
"""
function save_parameters(folder::AbstractString, p::OptimizationParameters)
    write_parameters("$folder/$PARAM_FILE", p)
end

"""
    load_parameters(folder::AbstractString)::OptimizationParameters

Read and return the optimization parameters stored in the `folder`.
"""
function load_parameters(folder::AbstractString)::OptimizationParameters
    return read_parameters("$folder/$PARAM_FILE")
end


"""
    save_generation(folder::AbstractString, gen::Integer, p::OptimizationParameters, pulses::AbstractArray{FullPulse})

Save all `pulses` in a CSV file. 
"""
function save_generation(folder::AbstractString, gen::Integer, ::OptimizationParameters, pulses::AbstractArray{FullPulse})
    renormalize_pulses!(pulses)
    write_all_pulses_full("$folder/pulses/optimization-full/generations/gen-$gen.csv", pulses)
end

"""
    save_generation(folder::AbstractString, gen::Integer, p::OptimizationParameters, xurs::AbstractArray{XUR_Pulse})

Save all `pulses` in a CSV file. The XUR pulses are stored as full pulses.
"""
function save_generation(folder::AbstractString, gen::Integer, p::OptimizationParameters, xurs::AbstractArray{XUR_Pulse})
    renormalize_pulses!(xurs)
    write_all_pulses_xur("$folder/pulses/optimization-cycle/generations/gen-$gen.csv", p, xurs)
end

"""
    load_generation(folder::AbstractString, gen::Integer, p::OptimizationParameters, type)

Load pulses from generation `gen` of given type, where
`type` can be `UTrack.FullPulse` or `UTrack.XUR_Pulse`.
"""
function load_generation(folder::AbstractString, gen::Integer, p::OptimizationParameters, type)
    if type === FullPulse
        subfolder = "$folder/pulses/optimization-full/generations/"
        pulses = read_all_pulses_full("$subfolder/gen-$gen.csv")
    elseif type === XUR_Pulse
        subfolder = "$folder/pulses/optimization-cycle/generations/"
        pulses = read_all_pulses_xur("$subfolder/gen-$gen.csv", p)
    end

    renormalize_pulses!(pulses)
    return pulses
end

"""
    latest_gen(folder::AbstractString, type=UTrack.FullPulse)

Return index of latest generation in `folder` of given `type` (starting at `0`). Returns `-1` if empty.
"""
function latest_gen(folder::AbstractString, type=FullPulse)
    if type === FullPulse
        files = readdir("$folder/pulses/optimization-full/generations/")
    elseif type === XUR_Pulse
        files = readdir("$folder/pulses/optimization-cycle/generations/")
    end
    matches = [parse(Int64,match(r"[0-9]+",f).match) for f in files if occursin(r"gen-[0-9]+\.csv",f)]
    return maximum(matches, init=-1)
end


"""
    load_latest_generation(folder::AbstractString)

Load the latest generation in `folder` overall.
"""
function load_latest_generation(folder::AbstractString)
    p = load_parameters(folder)

    latest_xur = latest_gen(folder, XUR_Pulse)
    latest_full = latest_gen(folder, FullPulse)

    @assert latest_full >= 0 || latest_xur >= 0

    if latest_full >= 0
        return load_generation(folder, latest_full, p, FullPulse)
    end

    return load_generation(folder, latest_xur, p, XUR_Pulse)
end

#=
"""
    p, pulses = load_optimization(folder::AbstractString)

Load parameters and latest generation from `folder`.
"""
function load_optimization(folder::AbstractString)
    p = load_parameters(folder)
    pulses = load_latest_generation(folder)
    return p, pulses 
end
=#

"""
    load_best_pulse(folder::AbstractString)

Load best pulse from latest generation from `folder`.
"""
function load_best_pulse(folder::AbstractString)
    pulses = load_latest_generation(folder)
    return pulses[1]
end


"""
    plot_optimization(folder::AbstractString, filename::AbstractString, type)

Plot the costs of all pulses of the given `type` over all generations.
"""
function plot_optimization(folder::AbstractString, filename::AbstractString, type)
    @assert isdir(folder)
    p = load_parameters(folder)

    latest = latest_gen(folder, type)
    if latest < 1
        return 
    end
    
    costs = []

    for i in 1:latest
        pulses = load_generation(folder, i, p, type)
        push!(costs, [compute_cost(p,ϕ) for ϕ in pulses])
    end

    data = hcat(costs...)'
    f = plot(data, legend=false);
    savefig(f, filename) 
end

"""
    plot_grape_convergence(folder::AbstractString, traces, gen::Integer, type)

Plot the traces of the GRAPE optimizations in generation `gen` for a given `type`. 
"""
function plot_grape_convergence(folder::AbstractString, traces, gen::Integer, type)
    if type === FullPulse
        subfolder = "$folder/plots/optimization-full/generations/"
    elseif type === XUR_Pulse
        subfolder = "$folder/plots/optimization-cycle/generations/"
    end
    plot_many_convergence(traces, "$subfolder/gen-$gen.pdf")
end


"""
    plot_and_write_final(p::OptimizationParameters, folder::AbstractString)

Create all remaining plots and output files after the optimization is done.
"""
function plot_and_write_final(p::OptimizationParameters, folder::AbstractString)
    ϕ = load_best_pulse(folder)

    if ϕ isa XUR_Pulse
        ϕ = get_full_pulse(p, ϕ)
    end

    write_pulse_matlab("$folder/pulses/final/optimal-pulse.mat", ϕ)
    write_all_pulses_full("$folder/pulses/final/optimal-pulse.csv", [ϕ])

    plot_cost_profile(p, ϕ, "$folder/plots/final/cost-profile.pdf")
    plot_pulse(ϕ, "$folder/plots/final/full-pulse.pdf")
    plot_pulsecycles(p, ϕ, "$folder/plots/final/pulse-cycles/")
    plot_optimization(folder, "$folder/plots/optimization-full/evolution.pdf", FullPulse)
    plot_optimization(folder, "$folder/plots/optimization-cycle/evolution.pdf", XUR_Pulse)
end