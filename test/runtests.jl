using UTrack, Test, CUDA

# test without CUDA
# CUDA.functional() = false

@testset "rw_pulses" begin
    p = OptimizationParameters()
    pulses = UTrack.initialize_pulses(p, UTrack.Rand)
    filename = "_pulses.csv"
    UTrack.write_all_pulses_full(filename,pulses)
    pulses2 = UTrack.read_all_pulses_full(filename)
    for i in eachindex(pulses)
        @test sum(abs.(pulses[i] .- pulses2[i])) < 1e-8
    end
    rm(filename)
end

@testset "rw_xur" begin
    p = OptimizationParameters()
    xur_pulses = [UTrack.random_xur_pulse(p) for _ in 1:p.NPulses]
    filename = "_xur_pulses.csv"
    UTrack.write_all_pulses_xur(filename, p, xur_pulses)
    xur_pulses2 = UTrack.read_all_pulses_xur(filename, p)
    for i in eachindex(xur_pulses)
        @test sum(abs.(UTrack.get_full_pulse(p,xur_pulses[i]) .- UTrack.get_full_pulse(p,xur_pulses2[i]))) < 1e-8
    end
    rm(filename)
end 

@testset "rw_pulse_mat" begin
    p = OptimizationParameters()
    ϕ = UTrack.initialize_pulse(p, UTrack.Rand)
    filename = "_pulse.mat"
    UTrack.write_pulse_matlab(filename,ϕ)
    ϕ2 = UTrack.read_pulse_matlab(filename)
    @test sum(abs.(ϕ .- ϕ2)) < 1e-8
    rm(filename)
end

@testset "rw_params" begin
    p = OptimizationParameters()
    filename = "_params.json"
    UTrack.write_parameters(filename, p)
    p2 = UTrack.read_parameters(filename)
    @test p == p2
    rm(filename)
end

@testset "out_folder" begin 
    p = OptimizationParameters()
    folder = "_folder"
    UTrack.setup_folder(folder,p)
    @test isdir(folder)
    rm(folder, recursive=true)
end

@testset "latest_gen" begin
    p = OptimizationParameters()
    folder = "_folder"
    UTrack.setup_folder(folder,p)

    @test UTrack.latest_gen(folder, UTrack.FullPulse) == -1
    @test UTrack.latest_gen(folder, UTrack.XUR_Pulse) == -1

    pulses = UTrack.initialize_pulses(p, UTrack.Rand)
    xur_pulses = [UTrack.random_xur_pulse(p) for _ in 1:p.NPulses]

    for i = 0:5
        UTrack.save_generation(folder, i, p, pulses)
        @test UTrack.latest_gen(folder, UTrack.FullPulse) == i
        UTrack.save_generation(folder, i, p, xur_pulses)
        @test UTrack.latest_gen(folder, UTrack.XUR_Pulse) == i     
    end

    pulses2 = UTrack.load_generation(folder, 5, p, UTrack.FullPulse)
    xur_pulses2 = UTrack.load_generation(folder, 5, p, UTrack.XUR_Pulse)
    
    @test pulses2[1] isa UTrack.FullPulse
    @test xur_pulses2[1] isa UTrack.XUR_Pulse

    rm(folder, recursive=true)
end

@testset "UR_cycle" begin
    @test sum(abs.(UTrack.UR_cycle(4,0,1)  - [0,0,π,π])) < 1e-8
    @test sum(abs.(UTrack.UR_cycle(4,0,-1) - [0,0,π,π])) < 1e-8
    @test sum(abs.(UTrack.UR_cycle(4,1,1)  - [0,π/2,0,π/2])) < 1e-8
    @test sum(abs.(UTrack.UR_cycle(4,1,-1) - [0,π/2,0,π/2])) < 1e-8
    @test sum(abs.(UTrack.UR_cycle(4,2,1)  - [0,π,π,0])) < 1e-8
    @test sum(abs.(UTrack.UR_cycle(4,2,-1) - [0,π,π,0])) < 1e-8
    @test sum(abs.(UTrack.UR_cycle(4,3,1)  - [0,3π/2,0,3π/2])) < 1e-8
    @test sum(abs.(UTrack.UR_cycle(4,3,-1) - [0,3π/2,0,3π/2])) < 1e-8
end

@testset "index_conversions" begin
    @test UTrack.sign_shift_from_t(UTrack.t_from_sign_shift(-1,7)) == (-1,7)
    @test [UTrack.t_from_sign_shift(UTrack.sign_shift_from_t(i)...) for i in 1:10] == collect(1:10)
    @test sum(abs.(UTrack.UR_cycles(4)[:,UTrack.t_from_sign_shift(-1,3)] - UTrack.UR_cycle(4,3,-1))) < 1e-8
end

@testset "XUR_Pulse" begin 
    p = OptimizationParameters()
    xur = UTrack.random_xur_pulse(p)
    ϕ = UTrack.get_full_pulse(p,xur)
    xur2 = UTrack.closest_xur_pulse(p,ϕ)
    ϕ2 = UTrack.get_full_pulse(p,xur2)
    @test sum(abs.(mod2pi.(ϕ-ϕ2))) < 1e-8
end

function test_propagation_backpropagation(p,s,pc)
    UTrack.propagate_state_compute_cost(p,s,pc)
    UTrack.backpropagate_compute_gradient(p,s,pc)
end

@testset "propagation" begin
    p=UTrack.OptimizationParameters()
    s=UTrack.OptimizationState(p)
    pc=UTrack.OptimizationPrecomputed(p)
    
    ϕ=UTrack.initialize_pulse(p,UTrack.Rand)
    UTrack.set_pulse(s,ϕ)

    test_propagation_backpropagation(p,s,pc)
    @test sum(abs.(view(s.Ψ,2,:,:))) < 1e-8
end;


function test_gradient_fast(p::UTrack.OptimizationParameters, s::UTrack.OptimizationState, pc::UTrack.OptimizationPrecomputed, ε = 1e-8)
    M = p.NRab
    K = p.NDet
    L = p.NCycles
    N = p.NSteps*p.NCycles

    J = UTrack.propagate_state_compute_cost(p,s,pc)
    UTrack.backpropagate_compute_gradient(p,s,pc)
    δϕ = copy(s.δϕ)
    δϕ2 = zeros(N)
    ϕ2 = copy(s.ϕ)
    for n in 1:N
        s.ϕ .= copy(ϕ2)
        CUDA.@allowscalar s.ϕ[n] += ε
        J2 = UTrack.propagate_state_compute_cost(p,s,pc)
        δϕ2[n] = (J2-J)/ε
    end

    return Array(δϕ) ./ δϕ2 / p.NCycles
end

@testset "gradient" begin
    p=UTrack.OptimizationParameters()
    s=UTrack.OptimizationState(p)
    pc=UTrack.OptimizationPrecomputed(p)
    UTrack.set_pulse(s, UTrack.UR_pulse(p))

    r=sort(test_gradient_fast(p,s,pc))
    @test 0.99 < r[1] < 1.01
    @test 0.99 < r[end] < 1.01
end


function test_gradient_kernel_full_xur(p)
    xur = UTrack.random_xur_pulse(p)
    ϕ = UTrack.get_full_pulse(p,xur)

    s = UTrack.OptimizationState(p)
    pc = UTrack.OptimizationPrecomputed(p) 
    UTrack.set_pulse(s,ϕ)

    pcx = UTrack.OptimizationPrecomputedXUR(p);
    sx = UTrack.OptimizationStateXUR(p)
    UTrack.set_pulse(sx,xur.phases)
    UTrack.set_types(sx,[UTrack.t_from_sign_shift(xur.signs[i],xur.shifts[i]) for i in 1:p.NCycles])

    c1=UTrack.propagate_state_compute_cost(p,s,pc)
    c2=UTrack.propagate_state_compute_cost_xur(p,sx,pcx)
    UTrack.backpropagate_compute_gradient(p,s,pc)
    UTrack.backpropagate_compute_gradient_xur(p,sx,pcx)

    CUDA.@allowscalar return abs(c1 - c2), abs(sum(s.δϕ[1:p.NSteps]) - sx.δΦ[1])
end

@testset "xur pr&gr" begin
    p=UTrack.OptimizationParameters()
    a,b = test_gradient_kernel_full_xur(p)
   @test a < 1e-8
   @test b < 1e-8
end;