# This file is copyrighted under the latest version of the EUPL.
# Please see LICENCE file for your rights under this license.


# number of generations of given type
function ngens(p::OptimizationParameters, ::Type{FullPulse})
    return p.NGens 
end

function ngens(p::OptimizationParameters, ::Type{XUR_Pulse})
    return p.NGensXUR 
end

"""
    perform_optimization_steps!(p::OptimizationParameters, pulses, folder::AbstractString, latest=0)

Optimizes a list of pulses. Saves pulses and plots to folder. The `pulses` can be of type `FullPulse` or `XUR_Pulse`.
"""
function perform_optimization_steps!(p::OptimizationParameters, pulses, folder::AbstractString, latest=0)
    type = typeof(pulses[1])
    if ngens(p,type) == 0
        return
    end 

    traces = [Float64[] for _ in 1:p.NPulses] 
    optimize_pulses!(p, pulses; print_func=vals->generation_printing(latest,vals));
    for i in (1+latest):ngens(p,type)
        pulses = evolve_generation(p, pulses);
        optimize_pulses!(p, pulses; print_func=vals->generation_printing(i,vals), traces=traces);
        save_generation(folder, i, p, pulses)
        plot_grape_convergence(folder, traces, i, type)
    end
end


"""
    new_optimization(folder::AbstractString, params::OptimizationParameters)

Starts new optimization based on OptimizationParameters `params` in `folder`.
The `folder` will be created, so it may not exist, but its direct parent folder has to exist, see `Base.Filesystem.mkdir`.
"""
function new_optimization(folder::AbstractString, p::OptimizationParameters) 
    @assert !isdir(folder)
    setup_folder(folder,p)

    xur_pulses = [random_xur_pulse(p) for _ in 1:p.NPulses]
    save_generation(folder, 0, p, xur_pulses)
    perform_optimization_steps!(p, xur_pulses, folder)

    full_pulses = [get_full_pulse(p,xur) for xur in xur_pulses]
    save_generation(folder, 0, p, full_pulses)
    perform_optimization_steps!(p, full_pulses, folder)

    plot_and_write_final(p, folder)
end


"""
    new_optimization_no_xur(folder::AbstractString, p::OptimizationParameters)

Starts new optimization based on OptimizationParameters `params` in `folder`. 
This function only uses full pulse optimization and no XUR. It is recommended to use 
`new_optimization` in practice for better performance.
The `folder` will be created, so it may not exist, but the direct parent folder has to exist, see `Base.Filesystem.mkdir`.
"""
function new_optimization_no_xur(folder::AbstractString, p::OptimizationParameters) 
    @assert !isdir(folder)
    setup_folder(folder,p)

    full_pulses = initialize_pulses(p)
    save_generation(folder, 0, p, full_pulses)
    perform_optimization_steps!(p, full_pulses, folder)

    plot_and_write_final(p, folder)
end


"""
    continue_optimization(folder::AbstractString)

Continue optimization started in `folder`. Use this if the optimization started using `new_optimization` was interrupted.
"""
function continue_optimization(folder::AbstractString)
    @assert isdir(folder)

    p = load_parameters(folder)
    pulses = load_latest_generation(folder)

    if pulses[1] isa XUR_Pulse
        latest = latest_gen(folder, XUR_Pulse)
        perform_optimization_steps!(p, pulses, folder, latest)
        pulses = [get_full_pulse(p,xur) for xur in pulses]
        save_generation(folder, 0, p, full_pulses)
    end
        
    latest = latest_gen(folder, FullPulse)
    perform_optimization_steps!(p, pulses, folder, latest)
    plot_and_write_final(p, folder)
end
