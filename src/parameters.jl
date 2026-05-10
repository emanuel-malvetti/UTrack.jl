# This file is copyrighted under the latest version of the EUPL.
# Please see LICENCE file for your rights under this license.


"""
    OptimizationParameters

Stores all parameters that define an optimization.
This uses the `@with_kw` macro, so one only has to provide parameters that
differ from the given default: `OptimizationParameters(NSteps=6)`.
Note that `NSteps` must be even when using π-pulses.

# Fields

| Field         | Type    | Default | Description |
|---------------|---------|---------|-------------|
| Rabi          | Float64 | 1.0     | Rabi frequency of the system in rad/s |
| Δt            | Float64 | π       | Time step in seconds, defaults to π-pulses |
| NSteps        | Int64   | 10      | Number of pulse phases per cycle |
| NCycles       | Int64   | 40      | Number of cycles |
| Detuning      | Float64 | 0.4     | Maximum detuning / Rabi |
| RabiFDev      | Float64 | 0.4     | Maximum amplitude deviation / Rabi |
| NDet          | Int64   | 64      | Number of points along the detuning axis |
| NRab          | Int64   | 64      | Number of points along the amplitude deviation axis |
| sigDet        | Float64 | 0.3     | Weight Gaussian parameter for robustness range of detuning |
| sigRab        | Float64 | 0.3     | Weight Gaussian parameter for robustness range of amplitude deviation |
| gradTol       | Float64 | 1e-12   | Terminate GRAPE when gradient is smaller than tolerance |
| maxIter       | Int64   | 10,000  | Terminate GRAPE after number of iterations |
| NPulses       | Int64   | 12      | Number of pulses in population |
| NGens         | Int64   | 10      | Number of full pulse generations |
| NGensXUR      | Int64   | 10      | Number of XUR generations |
"""
@with_kw struct OptimizationParameters
    Rabi::Float64          = 1.0

    # Pulse 
    Δt::Float64            = π
    NSteps::Int64          = 10
    NCycles::Int64         = 40

    # Robustness
    Detuning::Float64      = 0.4
    RabiFDev::Float64      = 0.4
    NDet::Int64            = 64
    NRab::Int64            = 64
    sigDet::Float64        = 0.3
    sigRab::Float64        = 0.3

    # Gradient 
    gradTol::Float64        = 1e-12
    maxIter::Int64          = 10_000

    # Evolution
    NPulses::Int64          = 12
    NGens::Int64            = 10
    NGensXUR::Int64         = 10
end

 