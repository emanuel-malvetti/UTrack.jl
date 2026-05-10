# This file is copyrighted under the latest version of the EUPL.
# Please see LICENCE file for your rights under this license.


# helper function for uniform printing of floats
function sprint_float(x, digits=16)
    Printf.format(Printf.Format("%.$(digits)e"), x)
end

"""
    trace_printing(x)

Print single iteration of GRAPE optimization. Use as follows: `Optim.Options(..., show_trace=false, callback=trace_printing)`.
"""
function trace_printing(x)
    if x.iteration == 0
        println("Iter\t", "Function Value\t\t", "Gradient Size\t\t", "Time")
    end
    println(x.iteration, "\t", sprint_float(x.value), "\t", sprint_float(x.g_norm), "\t", sprint_float(x.metadata["time"],3))
    false
end

"""
    generation_printing(gen::Integer, values)

Print single generation of genetic algorithm.
"""
function generation_printing(gen::Integer, values)
    print(gen, "\t")
    L=length(values)
    for i in 1:min(3,L)
        print(sprint_float(values[i],10), "\t")
    end
    if L >= 4
        for i in 4:min(10,L)
            print(sprint_float(values[i],3), "\t")
        end
    end
    if L >= 11
        println("...")
    end
end