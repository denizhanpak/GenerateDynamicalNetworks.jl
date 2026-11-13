# Example networks used by the tests, tutorials and benchmarks. All are made up except where noted.

"""
    ring(n) -> Matrix{Float64}

The `n`-state cycle 1 → 2 → … → n → 1.
"""
ring(n::Integer) = [mod1(i + 1, n) == j ? 1.0 : 0.0 for i in 1:n, j in 1:n]

# The four-state ring's dwells are the mean bout lengths of the four locomotion states of C. elegans
# printed in Roberts et al. (2016), Table 1, and its frame interval is that study's 29.97 Hz camera.
const WORM_SAMPLE = 1 / 29.97

const EXAMPLES = Tuple{String,Matrix{Float64},Vector{Float64}}[
    ("foraging", [0 1 0 0 0; 0 0 0.7 0.3 0; 0 0 0 0 1; 0 0 0 0 1; 1 0 0 0 0], [2.0, 1.0, 0.5, 1.5, 1.0]),
    ("worm ring", ring(4), [5.329, 0.441, 1.945, 0.208]),
    ("figure-eight", [0 0.75 0 0.25 0; 0 0 1 0 0; 1 0 0 0 0; 0 0 0 0 1; 1 0 0 0 0], [1.0, 0.6, 1.4, 2.0, 0.8]),
    ("Δ-clique", [0 0.4 0.6 0; 0 0 1 0; 0 0 0 1; 1 0 0 0], [1.5, 1.0, 0.8, 2.0]),
    # State 3 follows both branching states; the two branch ratios chain, so one noise vector fits both.
    ("shared, solvable", [0 0.7 0.3 0 0 0; 0 0 0 1 0 0; 0 0 0 0 1 0; 0 0 0.6 0 0 0.4; 1 0 0 0 0 0; 1 0 0 0 0 0],
        [1.5, 1.0, 0.8, 1.2, 1.0, 0.7]),
    # Both branching states choose between 2 and 3 at different odds: no single noise vector fits.
    ("shared, conflicting", [0 0.7 0.3 0 0 0; 0 0 0 0 1 0; 0 0 0 0 0 1; 0 0.4 0.6 0 0 0; 0 0 0 1 0 0; 1 0 0 0 0 0],
        [1.5, 1.0, 0.8, 1.2, 1.0, 0.7]),
    # Graphs with two-way edges (i → j and j → i).
    ("flip-flop", [0 1; 1 0], [1.0, 0.3]),
    ("two-way and exit", [0 1 0; 0.4 0 0.6; 1 0 0], [2.0, 0.5, 1.2]),
    ("hub with two returns", [0 0.5 0.5; 1 0 0; 1 0 0], [1.0, 0.4, 0.8]),
    # The worm ring with a return from state 2 to state 1.
    ("worm with return", [0 1 0 0; 0.47 0 0.53 0; 0 0 0 1; 1 0 0 0], [5.329, 0.441, 1.945, 0.208]),
]

"""
    example_names() -> Vector{String}

The names [`example`](@ref) accepts: six one-way graphs ("foraging", "worm ring", "figure-eight",
"Δ-clique", "shared, solvable", "shared, conflicting") and four with two-way edges ("flip-flop",
"two-way and exit", "hub with two returns", "worm with return").
"""
example_names() = first.(EXAMPLES)

"""
    example(name; noise=0.1, sample=nothing) -> TransitionSpec

One of the example networks (see [`example_names`](@ref)) at noise `noise`. The frame interval is
a tenth of the shortest dwell, or 1/29.97 for the two worm graphs, unless `sample` is given. The worm
graphs' dwells are the mean bout lengths in Roberts et al. (2016), Table 1.
"""
function example(name::AbstractString; noise::Real=0.1, sample::Union{Real,Nothing}=nothing)
    k = findfirst(c -> c[1] == name, EXAMPLES)
    k === nothing && throw(ArgumentError("no example named \"$name\"; the names are $(join(repr.(example_names()), ", "))"))
    _, P, τ = EXAMPLES[k]
    s = sample !== nothing ? sample : startswith(name, "worm") ? WORM_SAMPLE : minimum(τ) / 10
    return TransitionSpec(P, τ; noise, sample=s)
end

"""
    random_network(rng, n, d) -> Matrix{Float64}

A random transition matrix on `n` states: each ordered pair is an edge with probability `d`,
branches are cut to at most two, and the graph is redrawn until it is strongly connected. Two-way
branches get uneven probabilities `p` and `1 − p`, with `p` uniform on [0.15, 0.85].
"""
function random_network(rng::AbstractRNG, n::Integer, d::Real)
    while true
        A = [i != j && rand(rng) < d for i in 1:n, j in 1:n]
        for i in 1:n
            s = findall(A[i, :])
            length(s) > 2 && (A[i, shuffle(rng, s)[3:end]] .= false)
        end
        is_strongly_connected(SimpleDiGraph(A)) || continue
        P = Matrix{Float64}(A)
        for i in 1:n
            s = findall(A[i, :])
            length(s) == 2 && (p = 0.15 + 0.7 * rand(rng); P[i, s] = [p, 1 - p])
        end
        return P
    end
end
