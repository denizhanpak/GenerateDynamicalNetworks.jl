# Test graphs shared by sweeps.jl and the Phase 2 notebook.

ring(n) = [mod1(i + 1, n) == j ? 1.0 : 0.0 for i in 1:n, j in 1:n]
const NAMED = [
    ("foraging", [0 1 0 0 0; 0 0 .7 .3 0; 0 0 0 0 1; 0 0 0 0 1; 1 0 0 0 0.], [2.0, 1.0, 0.5, 1.5, 1.0]),
    ("worm ring", ring(4), [5.329, 0.441, 1.945, 0.208]),
    ("figure-eight", [0 .75 0 .25 0; 0 0 1 0 0; 1 0 0 0 0; 0 0 0 0 1; 1 0 0 0 0.], [1.0, 0.6, 1.4, 2.0, 0.8]),
    ("Δ-clique", [0 .4 .6 0; 0 0 1 0; 0 0 0 1; 1 0 0 0.], [1.5, 1.0, 0.8, 2.0]),
    # 3 follows both branching states. The two ratios chain (ε₂/ε₃, ε₃/ε₆), so one noise
    # vector satisfies both; the greedy branch_noise rejected this.
    ("shared, solvable", [0 .7 .3 0 0 0; 0 0 0 1 0 0; 0 0 0 0 1 0; 0 0 .6 0 0 .4; 1 0 0 0 0 0; 1 0 0 0 0 0.],
        [1.5, 1.0, 0.8, 1.2, 1.0, 0.7]),
    # Both branching states pick between 2 and 3, at different odds: no single noise vector works,
    # so repair adds a hidden state.
    ("shared, conflicting", [0 .7 .3 0 0 0; 0 0 0 0 1 0; 0 0 0 0 0 1; 0 .4 .6 0 0 0; 0 0 0 1 0 0; 1 0 0 0 0 0.],
        [1.5, 1.0, 0.8, 1.2, 1.0, 0.7]),
]

"""Phase 1's random graph: strongly connected, no self-loops, branches at most two-way and uneven."""
function random_spec_matrix(rng, n, d)
    while true
        A = [i != j && rand(rng) < d for i in 1:n, j in 1:n]
        for i in 1:n
            s = findall(A[i, :])
            length(s) > 2 && (A[i, shuffle(rng, s)[3:end]] .= false)
        end
        is_strongly_connected(SimpleDiGraph(A)) || continue
        P = float(A)
        for i in 1:n
            s = findall(A[i, :])
            length(s) == 2 && (p = rand(rng, Uniform(0.15, 0.85)); P[i, s] = [p, 1 - p])
        end
        return P
    end
end
