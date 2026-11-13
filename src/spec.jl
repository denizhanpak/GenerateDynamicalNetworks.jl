# The specification: a transition graph, mean dwell times, noise and the decoder's frame interval.

"""
    spec_problems(P, τ, noise) -> Vector{String}

Every reason `(P, τ, noise)` is not a valid specification, in plain words; empty if it is valid.
`P[i, j]` is the probability that a visit to state `i` is followed by a visit to `j`, so rows sum to
1. Every dwell must be positive, the noise must be positive (noise-free dwell times are not set),
and every state must be reachable from every other.
"""
function spec_problems(P::AbstractMatrix, τ::AbstractVector, noise::Real)
    size(P, 1) == size(P, 2) || return ["P must be square, not $(size(P))"]
    n = size(P, 1)
    problems = String[]
    n >= 2 || push!(problems, "a network needs at least two states")
    length(τ) == n || push!(problems, "P has $n states but τ has $(length(τ))")
    all(isfinite, P) || push!(problems, "P has a non-finite entry")
    all(>=(0), P) || push!(problems, "P has a negative entry")
    vec(sum(P; dims=2)) ≈ ones(n) || push!(problems, "rows of P must sum to 1")
    all(t -> isfinite(t) && t > 0, τ) || push!(problems, "every dwell must be positive")
    noise > 0 || push!(problems, "noise must be positive: noise-free dwell times are not fixed")
    n >= 1 && is_strongly_connected(SimpleDiGraph(P .> 0)) ||
        push!(problems, "every state must be reachable from every other")
    return problems
end

"""
    TransitionSpec(P, τ; noise, sample=minimum(τ)/10, names=["state 1", …])
    TransitionSpec(g::SimpleDiGraph, τ; kwargs...)

What a model has to reproduce: the transition matrix `P` (`P[i, j]` is the probability that a visit
to state `i` is followed by `j`), the mean dwell `τ[i]` in each state, the noise level `noise`, and
`sample`, the interval at which the decoder reads which state the system is in. A graph alone means
every branch out of a state is equally likely. Throws an `ArgumentError` listing every problem found
by [`spec_problems`](@ref).
"""
struct TransitionSpec
    P::Matrix{Float64}
    τ::Vector{Float64}
    noise::Float64
    sample::Float64
    names::Vector{String}
    function TransitionSpec(P::AbstractMatrix, τ::AbstractVector, noise::Real, sample::Real, names::AbstractVector)
        problems = spec_problems(P, τ, noise)
        sample > 0 || push!(problems, "the frame interval `sample` must be positive")
        length(names) == length(τ) || push!(problems, "there are $(length(names)) names for $(length(τ)) states")
        isempty(problems) || throw(ArgumentError(join(problems, "; ")))
        return new(Matrix{Float64}(P), Vector{Float64}(τ), noise, sample, String.(names))
    end
end

function TransitionSpec(P::AbstractMatrix, τ::AbstractVector; noise::Real, sample::Union{Real,Nothing}=nothing,
                        names=["state $i" for i in eachindex(τ)])
    s = sample === nothing ? (isempty(τ) ? 1.0 : minimum(τ) / 10) : sample
    return TransitionSpec(P, τ, noise, s, names)
end

function TransitionSpec(g::SimpleDiGraph, τ::AbstractVector; kwargs...)
    A = Matrix{Float64}(adjacency_matrix(g))
    return TransitionSpec(A ./ max.(sum(A; dims=2), 1), τ; kwargs...)
end

"""Number of states of a specification."""
nstates(spec::TransitionSpec) = length(spec.τ)

function Base.show(io::IO, spec::TransitionSpec)
    print(io, "TransitionSpec(", nstates(spec), " states, ", length(edge_list(spec.P)), " edges, noise ", spec.noise,
          ", sample ", round(spec.sample; sigdigits=4), ")")
end

"""
    edge_list(P) -> Vector{Tuple{Int,Int}}

The edges `(i, j)` with `P[i, j] > 0`, row by row. Backends number their per-edge parameters in
this order.
"""
edge_list(P::AbstractMatrix) = [(i, j) for i in axes(P, 1) for j in axes(P, 2) if P[i, j] > 0]
edge_list(spec::TransitionSpec) = edge_list(spec.P)

"""
    successors(P, i) -> Vector{Int}

The states that can follow state `i`.
"""
successors(P::AbstractMatrix, i::Integer) = findall(>(0), P[i, :])
successors(spec::TransitionSpec, i::Integer) = successors(spec.P, i)
