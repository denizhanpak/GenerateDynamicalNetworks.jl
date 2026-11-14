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

import CMAEvolutionStrategy as CES
using Distributions: Normal, cdf, truncated
using Graphs: SimpleDiGraph, adjacency_matrix, is_strongly_connected
using JuMP: Model, @constraint, @objective, @variable, optimize!, set_attribute, set_silent, value
using LinearAlgebra: Diagonal, I, Tridiagonal, cond, eigvals, norm
import OSQP
using Printf: @sprintf
using QuadGK: quadgk
using Random: AbstractRNG, Xoshiro, randperm, shuffle
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
export TwoPopModel, TwoPopulation
export SyncModel, Synchrony
export HopModel, Hopfield
export LVModel, Horchler
export Simplex
export CylinderModel, Cylinder

include("spec.jl")
include("examples.jl")
include("model.jl")
include("simulate.jl")
include("measure.jl")
include("classify.jl")
include("backend.jl")
include("fit/seed.jl")
include("fit/search.jl")
include("fit/edges.jl")
include("fit/correct.jl")
include("fit/race.jl")
include("backends/ctrnn.jl")
include("backends/twopop.jl")
include("backends/synchrony.jl")
include("backends/hopfield.jl")
include("backends/horchler.jl")
include("backends/simplex.jl")
include("backends/cylinder.jl")

end
