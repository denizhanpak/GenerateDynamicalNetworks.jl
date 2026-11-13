using GenerateDynamicalNetworks
using Test
using Aqua
using JET
using Graphs: SimpleDiGraph, add_edge!
using Random: Xoshiro
using StaticArrays: SMatrix, SVector
using Statistics: mean

@testset "GenerateDynamicalNetworks.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(GenerateDynamicalNetworks)
    end
    # JET follows Julia's internals closely and doesn't support prereleases
    if isempty(VERSION.prerelease)
        @testset "Code linting (JET.jl)" begin
            JET.test_package(GenerateDynamicalNetworks; target_modules=(GenerateDynamicalNetworks,))
        end
    end
    @testset "Core" include("core.jl")
    @testset "Dynamics" include("dynamics.jl")
end
