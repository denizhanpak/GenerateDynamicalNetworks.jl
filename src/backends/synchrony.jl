# Synchrony backend: phase oscillators with the network in synchrony space.
#
# 2M oscillators with one natural frequency, in M pairs. Pair σ has phase difference ψ_σ; it is in phase
# (S, ψ_σ = 0) or anti-phase (D, ψ_σ = π). Each oscillator of pair σ obeys
#
#     θ̇_{σ,k} = a_σ sin Δ_k + b_σ sin 2Δ_k + η ξ,      Δ_k = θ_{σ,other} − θ_{σ,k},
#
# in the frame rotating with the common frequency, where a_σ, b_σ are multilinear in cos ψ_τ of the other
# pairs: nonpairwise coupling, in the spirit of Bick (2018, 2019). Then ψ̇_σ = sin ψ_σ (g_σ + k_σ cos ψ_σ)
# with g = −2a, k = −4b, so the S and D sets of every pair are exactly invariant and the 2^M S/D patterns,
# the corners of a cube, are equilibria. At corner v the rate in direction σ is λ_σ(v) = k_σ + g_σ x_σ
# (x = +1 for S, −1 for D). Choosing g_σ and k_σ at every pattern of the other pairs sets every λ_σ(v)
# independently, and a cube edge whose two ends have opposite signs carries a one-dimensional connection.
#
# Each graph state is a corner; each edge is a path of single-pair flips through hidden transit corners,
# whose time counts toward the state being left. The decoder reads the corner from the signs of cos ψ.

"""
    SyncModel

Phase oscillators in `M` pairs with the graph's states on corners of the S/D cube: `Λ[σ, v]` is the
rate in direction σ at corner `v` (corner `v − 1` has bit `σ − 1` set when pair σ is anti-phase),
`node_of[v]` the graph state at corner `v` (0 for transits and unused corners) and `corner_of[i]` the
corner of state `i`. `α` is the stiffest rate, for the step. Built by `fit(spec, Synchrony())` or
`SyncModel(spec, out)`.
"""
struct SyncModel <: AbstractNetworkModel
    P::Matrix{Float64}
    visible::Int
    α::Float64
    M::Int
    Λ::Matrix{Float64}
    node_of::Vector{Int}
    corner_of::Vector{Int}
    ε::Float64
end

corner_bit(v, σ) = ((v - 1) >> (σ - 1)) & 1          # 1 when pair σ is D at corner v
flip(v, σ) = ((v - 1) ⊻ (1 << (σ - 1))) + 1
corner_name(v, M) = join((corner_bit(v, σ) == 1 ? "D" : "S" for σ in 1:M))

"""
The coefficients g_σ and k_σ at each pattern o of the other pairs (o − 1 lists their D bits in order),
from the corner rates: g = (λ(S) − λ(D))/2, k = (λ(S) + λ(D))/2.
"""
function coupling_tables(Λ, M)
    H = 2^(M - 1)
    G, K = zeros(M, H), zeros(M, H)
    for σ in 1:M, o in 1:H
        others = [τ for τ in 1:M if τ != σ]
        v = 1 + sum((((o - 1) >> (j - 1)) & 1) << (τ - 1) for (j, τ) in enumerate(others); init=0)   # pair σ is S here
        λS, λD = Λ[σ, v], Λ[σ, flip(v, σ)]
        G[σ, o], K[σ, o] = (λS - λD) / 2, (λS + λD) / 2
    end
    return G, K
end

function sync_drift(θ::SVector{N}, q, t) where {N}
    M = N ÷ 2
    c = ntuple(s -> cos(θ[2s] - θ[2s-1]), Val(N ÷ 2))
    gk = ntuple(Val(N ÷ 2)) do σ
        g, k = 0.0, 0.0
        for o in 1:size(q.G, 2)
            w, j = 1.0, 0
            for τ in 1:M
                τ == σ && continue
                w *= (1 + (((o - 1) >> j) & 1 == 1 ? -c[τ] : c[τ])) / 2
                j += 1
            end
            g += w * q.G[σ, o]
            k += w * q.K[σ, o]
        end
        (g, k)
    end
    # the two oscillators of a pair see Δ = ±ψ, so one sine and cosine per pair serve both
    pull = ntuple(Val(N ÷ 2)) do σ
        sψ, cψ = sincos(θ[2σ] - θ[2σ-1])
        -gk[σ][1] / 2 * sψ - gk[σ][2] / 2 * sψ * cψ          # a sin ψ + b sin 2ψ with a = −g/2, b = −k/4
    end
    return SVector{N}(ntuple(i -> isodd(i) ? pull[(i+1)÷2] : -pull[i÷2], Val(N)))
end

sync_corner(θ, M) = 1 + sum((cos(θ[2σ] - θ[2σ-1]) < 0) << (σ - 1) for σ in 1:M)

drift_function(::SyncModel) = sync_drift
function parameters(m::SyncModel)
    G, K = coupling_tables(m.Λ, m.M)
    H = size(G, 2)
    return (; G=SMatrix{m.M,H}(G), K=SMatrix{m.M,H}(K), m.ε, node_of=Tuple(m.node_of))
end
"""Next to the first state's corner, a little off every pair's exact S or D value."""
initial_state(m::SyncModel) = let v = m.corner_of[1]
    SVector{2m.M}(ntuple(i -> iseven(i) ? (corner_bit(v, i ÷ 2) == 1 ? π - 0.05 : 0.05) : 0.0, 2m.M))
end
speed(m::SyncModel) = m.α
in_bounds(::SyncModel, θ) = all(isfinite, θ)
frame_state(::SyncModel, q, θ) = q.node_of[sync_corner(θ, length(θ) ÷ 2)]

# ── Layout: graph states on corners, edges as corner paths ───────────────────────────────────

"""
A layout: `M` pairs, the corner of each graph state, and for each edge (in `edge_list` order) its
corner path from the source state's corner to the target's. Path corners other than the ends are transits.
"""
struct SyncLayout
    M::Int
    corner_of::Vector{Int}
    paths::Vector{Vector{Int}}
end

"""
A shortest corner path from `s` to `t` through unused corners, neighbours tried in random order. The
last step may land on `t` from `s` directly only if `t → s` isn't already a step (a cube edge can carry
one direction only, so a two-way pair needs two separate paths).
"""
function corner_path(s, t, used, steps, M, rng)
    prev = Dict(s => 0)
    queue = [s]
    while !isempty(queue)
        u = popfirst!(queue)
        for σ in shuffle(rng, 1:M)
            w = flip(u, σ)
            haskey(prev, w) && continue
            if w == t
                (u == s && (t, s) in steps) && continue
                p = [t, u]
                while p[end] != s
                    push!(p, prev[p[end]])
                end
                return reverse(p)
            end
            used[w] && continue
            prev[w] = u
            push!(queue, w)
        end
    end
    return nothing
end

function try_layout(P, M, rng)
    n, edges = size(P, 1), edge_list(P)
    corners = randperm(rng, 2^M)[1:n]
    used = falses(2^M)
    used[corners] .= true
    steps = Set{Tuple{Int,Int}}()
    paths = Vector{Vector{Int}}(undef, length(edges))
    for e in randperm(rng, length(edges))
        i, j = edges[e]
        p = corner_path(corners[i], corners[j], used, steps, M, rng)
        p === nothing && return nothing
        used[p[2:end-1]] .= true
        foreach(k -> push!(steps, (p[k], p[k+1])), 1:length(p)-1)
        paths[e] = p
    end
    return SyncLayout(M, corners, paths)
end

"""
    sync_layout(P; Mmax=7, tries=3000, seed=1) -> SyncLayout

Graph states on cube corners and edges as corner paths: the fewest pairs that work, then the fewest
transit corners, over `tries` random placements and routings per pair count. Edges leaving one state
start in different directions, and no corner serves two paths.
"""
function sync_layout(P; Mmax::Integer=7, tries::Integer=3000, seed::Integer=1)
    rng = Xoshiro(seed)
    n = size(P, 1)
    lo = max(2, ceil(Int, log2(n)), maximum(i -> count(>(0), P[i, :]), 1:n))
    for M in lo:Mmax
        best = nothing
        for _ in 1:tries
            L = try_layout(P, M, rng)
            L === nothing && continue
            cost = sum(length, L.paths)
            (best === nothing || cost < best[1]) && (best = (cost, L))
        end
        best === nothing || return best[2]
    end
    throw(ArgumentError("no layout with at most $Mmax pairs"))
end

transits(L::SyncLayout) = sum(p -> length(p) - 2, L.paths)

"""
Corner rates from a layout. Along a path each step u → w leaves u at its out-rate (`out[e]` for a
state, `fast` for a transit) and arrives at w along a contracting direction, −`contract`. Every other
direction at a used corner contracts at `contract`. Unused corners point toward the network, by
distance to it.
"""
function corner_rates(L::SyncLayout, out, fast, contract)
    M = L.M
    Λ = fill(NaN, M, 2^M)
    used = Set(reduce(vcat, L.paths))
    for (e, p) in enumerate(L.paths), s in 1:length(p)-1
        u, w = p[s], p[s+1]
        σ = only(τ for τ in 1:M if flip(u, τ) == w)
        Λ[σ, u] = s == 1 ? out[e] : fast
        Λ[σ, w] = -contract
    end
    for v in used, σ in 1:M
        isnan(Λ[σ, v]) && (Λ[σ, v] = -contract)
    end
    dist = Dict(v => 0 for v in used)
    frontier = collect(used)
    while length(dist) < 2^M
        next = Int[]
        for v in frontier, σ in 1:M
            w = flip(v, σ)
            haskey(dist, w) || (dist[w] = dist[v] + 1; push!(next, w))
        end
        frontier = next
    end
    for v in 1:2^M, σ in 1:M
        v in used && continue
        w = flip(v, σ)
        Λ[σ, v] = (dist[w], w) < (dist[v], v) ? 1.0 : -1.0
    end
    return Λ
end

function sync_model(spec, L::SyncLayout, out; fast, contract)
    Λ = corner_rates(L, out, fast, contract)
    node_of = zeros(Int, 2^L.M)
    for (i, v) in enumerate(L.corner_of)
        node_of[v] = i
    end
    return SyncModel(Matrix{Float64}(spec.P), nstates(spec), maximum(abs, Λ), L.M, Λ, node_of, copy(L.corner_of), float(spec.noise))
end

"""
    SyncModel(spec, out)

The model with out-rates `out` (one per edge, in `edge_list` order) on the layout `fit` uses, with the
transit rate and contraction from [`rough_rates`](@ref).
"""
function SyncModel(spec::TransitionSpec, out::AbstractVector)
    _, fast = rough_rates(spec)
    return sync_model(spec, sync_layout(spec.P), out; fast, contract=fast)
end

"""
    Synchrony(; seed=:rough, passes=12, base=100_000)

The synchrony backend's fit. The layout ([`sync_layout`](@ref)) puts states on corners of the S/D cube
of the fewest oscillator pairs that work. Every edge's departure rate starts from the rule of thumb
`log(1/ε)/τᵢ · √(Pᵢⱼ · out-degree)` (`seed=:rough`), or from the 1-D race of each state's exits solved
for its dwell and shares, which also gives the starting response (`seed=:race`). Then `passes` correction
passes ([`correct_rates`](@ref)); the transit rate and contraction are 4× the largest rule-of-thumb rate.

    fit(spec, Synchrony(); checkpoint=nothing, deadline=Inf, verbose=false) -> SyncModel

!!! note "Status: stable with `seed=:rough`, experimental with `seed=:race`"
    With the default seed it passes the acceptance protocol on most of the benchmark cases; with the race seed it
    fails half of them. See the Results page of the documentation.
"""
struct Synchrony <: AbstractBackend
    seed::Symbol
    passes::Int
    base::Int
    function Synchrony(seed::Symbol, passes::Integer, base::Integer)
        seed in (:rough, :race) || throw(ArgumentError("seed must be :rough or :race, got :$seed"))
        return new(seed, passes, base)
    end
end
Synchrony(; seed::Symbol=:rough, passes::Integer=12, base::Integer=100_000) = Synchrony(seed, passes, base)

function fit(spec::TransitionSpec, b::Synchrony; checkpoint=nothing, deadline::Real=Inf, verbose::Bool=false)
    L = sync_layout(spec.P)
    rough, fast = rough_rates(spec)
    out0, response = b.seed === :race ? (race_seed_rates(spec, rough, fast, spec.noise), race_response(fast, spec.noise)) :
                     (rough, nothing)
    build = out -> sync_model(spec, L, out; fast, contract=fast)
    r = correct_rates(spec, build, out0; b.passes, b.base, response, checkpoint, deadline, verbose)
    return build(r.out)
end
