# Simplex backend (Ashwin & Postlethwaite 2013): a Lotka–Volterra network on the simplex, one node per state,
# one global speed, a rate per edge and a contraction per state, one shared noise, no hidden states; branches
# come from the edge rates. The model is a u-form LVModel, the type the Horchler backend builds.

"""
    simplex_model(spec, θ) -> LVModel

dxⱼ = s·xⱼ(1 − |x|² + Σᵢ aᵢⱼxᵢ²) dt + ε dW as a u-form [`LVModel`](@ref), with
`θ = [log s; log aᵢⱼ for each edge in edge_list order; log cᵢ for each state]`: aᵢⱼ is the edge's rate on
an edge i → j, −cᵢ off the graph and 0 on the diagonal. The noise ε is the specification's.
"""
function simplex_model(spec::TransitionSpec, θ::AbstractVector)
    edges = edge_list(spec.P)
    n, ne = nstates(spec), length(edges)
    length(θ) == 1 + ne + n || throw(ArgumentError("θ must hold 1 + $ne + $n values (speed, edges, states), not $(length(θ))"))
    s, e, c = exp(θ[1]), exp.(θ[2:1+ne]), exp.(θ[2+ne:1+ne+n])
    a = [i == j ? 0.0 : -c[i] for i in 1:n, j in 1:n]
    for (k, (i, j)) in enumerate(edges)
        a[i, j] = e[k]
    end
    ρ = [s * (i == j ? 1.0 : 1.0 - a[j, i]) for i in 1:n, j in 1:n]
    return LVModel(:u, spec.P, n, 0.1, ones(n), fill(s, n), fill(float(spec.noise), n), ρ)
end

"""The simplex fit's start and box: the edge seed ([`edge_seed_rates`](@ref)) with unit contractions."""
function simplex_start(spec::TransitionSpec)
    edges = edge_list(spec.P)
    n, ne = nstates(spec), length(edges)
    r = edge_seed_rates(spec, edges)
    θ0 = [log(r.α); r.logs; zeros(n)]
    lower = [log(0.05); fill(log(1e-4), ne); fill(log(0.25), n)]
    upper = [log(200.0); fill(log(4.0), ne); fill(log(4.0), n)]
    return (; θ0, lower, upper, r.seeded)
end

"""
    Simplex(; passes=4, runs=16, T=nothing, budget=500, popsize=12, seconds=300.0)

The simplex backend's fit (Ashwin & Postlethwaite 2013): one global speed, a rate per edge and a
contraction per state ([`simplex_model`](@ref)), one shared noise, no hidden states. A seed from the Horchler
dwell design ([`edge_seed_rates`](@ref)), `passes` correction passes of `runs` runs of length `T` (default
125Στ), and if the best of those fails a check on 32 held-out runs, a CMA-ES search over every parameter of
at most `budget` evaluations or `seconds` seconds, then two correction passes as a polish. (The prototype
stopped at 420 s by default.) A binding time limit makes the result depend on the machine's speed;
`seconds=Inf` limits the search by `budget` alone.

    fit(spec, Simplex(); verbose=false) -> LVModel

!!! warning "Status: experimental"
    Fails the acceptance protocol on half or more of the benchmark cases; see the Results page of the documentation.
"""
struct Simplex <: AbstractBackend
    passes::Int
    runs::Int
    T::Union{Float64,Nothing}
    budget::Int
    popsize::Int
    seconds::Float64
    function Simplex(passes::Integer, runs::Integer, T::Union{Real,Nothing}, budget::Integer, popsize::Integer, seconds::Real)
        passes >= 1 || throw(ArgumentError("passes must be at least 1"))
        return new(passes, runs, T === nothing ? nothing : float(T), budget, popsize, seconds)
    end
end
Simplex(; passes::Integer=4, runs::Integer=16, T::Union{Real,Nothing}=nothing, budget::Integer=500, popsize::Integer=12,
        seconds::Real=300.0) = Simplex(passes, runs, T, budget, popsize, seconds)

function fit(spec::TransitionSpec, b::Simplex; verbose::Bool=false)
    T = b.T === nothing ? 125sum(spec.τ) : b.T
    st = simplex_start(spec)
    verbose && !st.seeded && println("  simplex: the dwell design has no solution; starting from a rough guess")
    θ = edge_rate_fit(θ -> simplex_model(spec, θ), spec, st.θ0, st.lower, st.upper; b.passes, b.runs, T, b.budget,
                      b.popsize, b.seconds, label="simplex", verbose)
    return simplex_model(spec, θ)
end
