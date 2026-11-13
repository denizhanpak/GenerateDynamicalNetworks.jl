"""
    GenerateDynamicalNetworks

Build a noisy dynamical system whose switching between states reproduces a given transition graph
and mean dwell times. The core layer here holds the specification ([`TransitionSpec`](@ref)),
example networks, one frame-based stochastic simulator shared by every backend ([`simulate`](@ref)),
the measurement and acceptance rules ([`measure`](@ref), [`verdict`](@ref), [`validate`](@ref),
[`accept`](@ref)) and the dynamical classifier ([`classify`](@ref)). Backends plug in by subtyping
[`AbstractNetworkModel`](@ref).
"""
module GenerateDynamicalNetworks

using Graphs: SimpleDiGraph, adjacency_matrix, is_strongly_connected
using LinearAlgebra: Diagonal, I, cond, eigvals, norm
using Printf: @sprintf
using Random: AbstractRNG, shuffle
using Roots: Bisection, find_zero
using SciMLBase: DiscreteCallback, EnsembleProblem, EnsembleThreads, SDEProblem, remake, solve, terminate!
using Serialization: deserialize, serialize
using StaticArrays: SMatrix, SVector
using Statistics: mean, std
using StochasticDiffEqLowOrder: EM

export TransitionSpec, spec_problems, nstates, edge_list, successors
export ring, example, example_names, random_network
export AbstractNetworkModel, simulate
export measure, off_graph, verdict, score, validate, accept
export classify
export AbstractBackend, fit, OutOfTime
export CTRNNModel, CTRNNDesign

include("spec.jl")
include("examples.jl")
include("model.jl")
include("simulate.jl")
include("measure.jl")
include("classify.jl")
include("backend.jl")
include("backends/ctrnn.jl")

end
