# Phase 6: phase oscillators with the network in synchrony space.
#
# 2M oscillators with one natural frequency, in M pairs. Pair σ has phase difference ψ_σ; it is in phase
# (S, ψ_σ = 0) or anti-phase (D, ψ_σ = π). Each oscillator of pair σ obeys
#
#     θ̇_{σ,k} = ω + a_σ sin Δ_k + b_σ sin 2Δ_k + η ξ,      Δ_k = θ_{σ,other} − θ_{σ,k},
#
# where a_σ, b_σ are multilinear in cos ψ_τ of the other pairs: nonpairwise coupling, in the spirit of
# Bick (2018, 2019). Then ψ̇_σ = sin ψ_σ (g_σ + k_σ cos ψ_σ) with g = −2a, k = −4b, so the S and D sets of
# every pair are exactly invariant (the swap within each pair), and the 2^M S/D patterns, the corners of
# a cube, are equilibria. At corner v the rate in direction σ is λ_σ(v) = g_σ + k_σ x_σ (x = +1 for S,
# −1 for D). Choosing g_σ and k_σ at every pattern of the other pairs sets every λ_σ(v) independently,
# and a cube edge whose two ends have opposite signs carries a one-dimensional connection.
#
# Each graph state is a corner; each edge is a path of single-pair flips through hidden transit corners,
# whose time counts toward the state being left. The decoder reads the corner from the signs of cos ψ.
# Coupling coefficients differ by pair (choice 1, agreed 2026-11-11). Simulated in the frame rotating
# with the common frequency (ω = 0), which changes no phase difference and lets the classifier find
# equilibria.
include(joinpath(@__DIR__, "..", "phase5", "classes.jl"))      # the pipeline and classifier, backends.jl through it
using Serialization, Random

struct SyncModel
    P::Matrix{Float64}
    visible::Int
    α::Vector{Float64}          # stiffest rate, for the step-size rule
    M::Int
    Λ::Matrix{Float64}          # Λ[σ, v]: rate in direction σ at corner v; corner v − 1 has bit σ − 1 set when pair σ is D
    node_of::Vector{Int}        # corner → graph state, 0 for transit and unused corners
    corner_of::Vector{Int}      # graph state → corner
    ω::Float64
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

function drift_sync(θ::SVector{N}, q, t) where {N}
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
    return SVector{N}(ntuple(i -> q.ω + (isodd(i) ? pull[(i+1)÷2] : -pull[i÷2]), Val(N)))
end

function read_sync!(ig)
    q = ig.p
    ig.iter % q.every == 0 || return nothing
    θ = ig.u
    if !all(isfinite, θ)
        q.V.failed = true
        terminate!(ig)
        return nothing
    end
    v = 1
    for σ in 1:length(θ)÷2
        cos(θ[2σ] - θ[2σ-1]) < 0 && (v += 1 << (σ - 1))
    end
    k = q.node_of[v]
    if k != 0 && (isempty(q.V.state) || last(q.V.state) != k)
        push!(q.V.state, k)
        push!(q.V.t, ig.t)
    end
    return nothing
end
const SYNC_FRAME = DiscreteCallback((u, t, ig) -> true, read_sync!; save_positions=(false, false))

function sync_params(m::SyncModel)
    G, K = coupling_tables(m.Λ, m.M)
    H = size(G, 2)
    return (; G=SMatrix{m.M,H}(G), K=SMatrix{m.M,H}(K), m.ω, m.ε, node_of=Tuple(m.node_of))
end

"""Start next to the first state's corner, a little off every pair's exact S or D value."""
sync_start(m::SyncModel) = let v = m.corner_of[1]
    SVector{2m.M}(ntuple(i -> iseven(i) ? (corner_bit(v, i ÷ 2) == 1 ? π - 0.05 : 0.05) : 0.0, 2m.M))
end

function simulate_frames(m::SyncModel; T, seeds, sample, dtscale=0.01)
    every, dt = frame_steps(m, sample, dtscale)
    prob = SDEProblem(drift_sync, noise_flat, sync_start(m), (0.0, T), merge(sync_params(m), (; every, V=Frames())))
    ens = EnsembleProblem(prob;
        prob_func=(pr, ctx) -> remake(pr; seed=UInt64(seeds[ctx.sim_id]), p=(; pr.p..., V=Frames())),
        output_func=(sol, ctx) -> ((state=sol.prob.p.V.state, t=sol.prob.p.V.t, failed=sol.prob.p.V.failed), false))
    return solve(ens, EM(), EnsembleThreads(); trajectories=length(seeds), merge(quiet(dt), (; callback=SYNC_FRAME))...).u
end

function classify(m::SyncModel; spec, kw...)
    decode(θ) = (v = 1 + sum((cos(θ[2σ] - θ[2σ-1]) < 0) << (σ - 1) for σ in 1:m.M); m.node_of[v] == 0 ? -v : m.node_of[v])
    return classify(drift_sync, sync_params(m), sync_start(m), decode; noise=spec.ε₀, dt=0.02 / m.α[1], kw...)
end

# ── Layout: graph states on corners, edges as corner paths ───────────────────────────────────

"""
A layout: M pairs, the corner of each graph state, and for each edge (in `edge_list` order) its corner
path from the source state's corner to the target's. Path corners other than the ends are transits.
"""
struct SyncLayout
    M::Int
    corner_of::Vector{Int}
    paths::Vector{Vector{Int}}
end

"""
The pilot's 3-ring on 2 pairs: the corner cycle SS → DS → DD → SD → SS, with the three states on SS, DS
and DD in ring order and the transit on SD, rotated so the transit follows the longest dwell.
"""
function ring3_layout(spec)
    long = argmax(spec.τ)
    cyc = [1, 2, 4]                           # SS, DS, DD
    corner = zeros(Int, 3)
    for p in 0:2
        corner[mod1(long + p - 2, 3)] = cyc[p+1]
    end
    paths = [corner[i] == 4 ? [4, 3, corner[j]] : [corner[i], corner[j]] for (i, j) in edge_list(spec.P)]
    return SyncLayout(2, corner, paths)
end

"""
Corner rates from a layout. Along a path each step u → w leaves u at its out-rate (`out[e]` for a
state, `fast` for a transit) and arrives at w along a contracting direction: −`contract`, or with
`arrive` set and nonzero, −`arrive` × max(|departure rate|, `floor`). Every other direction at a used corner
contracts at `contract`. Unused corners point toward the network, by distance to it.
"""
function corner_rates(L::SyncLayout, out, fast, contract; arrive=nothing, floor=0.0)
    M = L.M
    Λ = fill(NaN, M, 2^M)
    used = Set(reduce(vcat, L.paths))
    for (e, p) in enumerate(L.paths), s in 1:length(p)-1
        u, w = p[s], p[s+1]
        σ = only(τ for τ in 1:M if flip(u, τ) == w)
        Λ[σ, u] = s == 1 ? out[e] : fast
        # the arrival contraction also sets how sharply the flow leaves u (the cubic term at λ ≈ 0), so with
        # `arrive` it scales with the step's own departure rate rather than being one global value
        Λ[σ, w] = (arrive === nothing || arrive == 0) ? -contract : -arrive * max(abs(Λ[σ, u]), floor)
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

function sync_model(spec, L::SyncLayout, out; fast, contract, arrive=nothing, floor=0.0, ω=0.0)
    Λ = corner_rates(L, out, fast, contract; arrive, floor)
    node_of = zeros(Int, 2^L.M)
    for (i, v) in enumerate(L.corner_of)
        node_of[v] = i
    end
    return SyncModel(float(spec.P), size(spec.P, 1), [maximum(abs, Λ)], L.M, Λ, node_of, L.corner_of, ω, float(spec.ε₀))
end

# ── Fitting ───────────────────────────────────────────────────────────────────────────────────

"""
Seed out-rates from the passage law T ≈ ln(1/η)/λ, then correction passes: per state, a secant step on
z = asinh(λ/c) against log measured dwell (β learned between passes). z behaves like log λ for large
rates and is linear through zero, so a rate may turn negative: the corner is then weakly stable and
noise has to cross a small barrier inside the edge (the excitable side). At high noise a long dwell
needs that, because even at λ = 0 the flow away from a corner grows like the cube of the distance.
The transit rate and the contraction are fixed at the seed. Single-successor states only, for the
pilot. 10 passes, the last 3 with 96 runs at the finer step; the last update is returned. This is the
rule the pilot used; `fit_sync` below generalises it.
"""
function fit_sync_pilot(spec, L; passes=10, base=100_000)
    n, edges = size(spec.P, 1), edge_list(spec.P)
    all(i -> count(>(0), spec.P[i, :]) == 1, 1:n) || throw(ArgumentError("the pilot fit handles one successor per state"))
    law = log(1 / spec.ε₀)
    out = [law / spec.τ[i] for (i, _) in edges]
    fast, contract = 4maximum(out), 8maximum(out)
    T = 250sum(spec.τ)
    β, px, py = ones(n), fill(NaN, n), fill(NaN, n)
    c = 0.05minimum(out)                 # the asinh scale: well below every seeded rate
    trace = []
    for pass in 1:passes
        late = pass > passes - 3
        m = sync_model(spec, L, out; fast, contract)
        r = simulate_frames(m; T, seeds=base+1000pass+1:base+1000pass+(late ? 96 : 32), spec.sample, dtscale=late ? 0.01 : 0.02)
        s = score(r, spec)
        @printf("  pass %d  f %.4f  dwell %s  off %.4f\n", pass, s.f, round.(s.dwell; sigdigits=4), s.off)
        flush(stdout)
        push!(trace, (; pass, out=copy(out), s.f, s.dwell, s.off))
        all(isfinite, s.dwell) || break
        for (e, (i, _)) in enumerate(edges)
            x, y = asinh(out[e] / c), -log(s.dwell[i])
            if isfinite(px[i]) && abs(x - px[i]) > 0.02
                β[i] = clamp(0.5β[i] + 0.5(y - py[i]) / (x - px[i]), 0.1, 5.0)
            end
            px[i], py[i] = x, y
            out[e] = c * sinh(x + clamp((-log(spec.τ[i]) - y) / β[i], -1, 1))
        end
        println("    response β per state ", round.(β; digits=2))
    end
    return (; model=sync_model(spec, L, out; fast, contract), out, fast, contract, trace)
end

# ── General layouts and fits (after the pilot's go) ──────────────────────────────────────────

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
Graph states on cube corners and edges as corner paths: the fewest pairs that work, then the fewest
transit corners, over random placements and routings. Edges leaving one state start in different
directions, and no corner serves two paths.
"""
function sync_layout(P; Mmax=7, tries=3000, seed=1)
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
The starting response: a faster exit along any edge shortens the dwell, and shifts the share toward that
edge. The share response scales with ln(1/η): at low noise the larger rate wins almost every time, so
shares are far steeper in the rate ratio (2.0 at η = 0.1, as first used; 6.0 at 10⁻³).
"""
function initial_response(k, law=log(10))
    k == 1 && return ones(1, 1)
    J = zeros(k, k)
    J[1, :] .= 1 / k
    b = 2.0law / log(10)
    for a in 1:k-1
        J[a+1, a], J[a+1, k] = b, -b
    end
    return J
end

"""
The rule-of-thumb out-rates (in `edge_list` order), log(1/ε)/τ_i · √(P_ij · out-degree of i), and 4× the
largest of them, the transit rate and contraction fit_sync uses. Every seed and fit starts from these.
"""
function rough_rates(spec)
    n, edges = size(spec.P, 1), edge_list(spec.P)
    outs = [count(e -> e[1] == i, edges) for i in 1:n]
    law = log(1 / spec.ε₀)
    rough = [law / spec.τ[i] * sqrt(spec.P[i, j] * outs[i]) for (i, j) in edges]
    return rough, 4maximum(rough)
end

"""
The general fit. Seeds as the pilot, then per state one Broyden step on x = asinh.(λ_e/c) over its
out-edges against y = [−log dwell; log branch ratios], with the Jacobian learned from pass to pass
(the per-state step of the CTRNN design, Phase 5). Rates may turn negative (the excitable side). The
last 3 of 12 passes use 96 runs at the finer step and steps of at most 0.3 in x, since the pilot's
excitable state swung between 7.3 and 11.9 s on full steps. The last update is returned.

The transit rate and the contraction are both 4× the largest seeded out-rate (the pilot used 4× and
8×): the contraction sets the stiffness, and halving it halves the steps. Passes run in chunks of 16
runs; with `checkpoint` the state is saved after each chunk and a rerun resumes, and `OutOfTime` is
thrown rather than start a chunk expected to end after `deadline` (timed afresh in each batch).
"""
function fit_sync(spec, L; passes=12, base=100_000, checkpoint=nothing, deadline=Inf, seed=nothing, response=nothing,
                  build=nothing, cap=Inf)
    n, edges = size(spec.P, 1), edge_list(spec.P)
    outs = [[e for (e, (a, _)) in enumerate(edges) if a == i] for i in 1:n]
    law = log(1 / spec.ε₀)
    rough, fast = rough_rates(spec)
    contract = fast
    # `seed` replaces the rule-of-thumb start; `response(o, out, c)` gives a state's starting Broyden
    # matrix (both from seed1d.jl). With neither, this is the rule the suite used.
    out = seed === nothing ? rough : copy(seed)
    # one contraction everywhere (arrive = 0). Arrival scaled to each step's departure rate (arrive = 4,
    # floor = law/(4 max τ)) was tried on the real worm graph and made the dwell response swing too hard to fit.
    arrive, floor = 0.0, 0.0
    c = 0.05minimum(abs.(out))
    T = 250sum(spec.τ)
    start_J(o) = response === nothing ? initial_response(length(o), law) : response(o, out, c)
    J = [start_J(o) for o in outs]
    px, py = [Float64[] for _ in 1:n], [Float64[] for _ in 1:n]
    trace, partial, done = [], Any[], 0
    if checkpoint !== nothing && isfile(checkpoint)
        st = deserialize(checkpoint)
        out, J, px, py, trace, partial, done = st.out, st.J, st.px, st.py, st.trace, st.partial, st.done
        # the asinh scale comes from the starting rates, which a resumed caller may not pass again
        # (Phase 7 skips the seed); a changed scale shifts every x and breaks the Broyden update
        c = get(st, :c, c)
        println("  resuming after pass $done", isempty(partial) ? "" : ", with $(length(partial)) runs of the next")
    end
    save() = checkpoint !== nothing && serialize(checkpoint, (; out, J, px, py, trace, partial, done, c))
    per_unit = NaN
    for pass in done+1:passes
        late = pass > passes - 3
        dts = late ? 0.01 : 0.02
        # `build(out)` makes another backend's model from the same out-rates (Phase 7's Hopfield network)
        m = build === nothing ? sync_model(spec, L, out; fast, contract, arrive, floor) : build(out)
        seeds = base+1000pass+1:base+1000pass+(late ? 96 : 32)
        for chunk in Iterators.partition(seeds[length(partial)+1:end], 16)
            if time() + (isnan(per_unit) ? 0.0 : 1.2per_unit * length(chunk) / dts) > deadline
                save()
                throw(OutOfTime())
            end
            started = time()
            append!(partial, simulate_frames(m; T, seeds=collect(chunk), spec.sample, dtscale=dts))
            per_unit = (time() - started) / (length(chunk) / dts)
            save()
        end
        r = [x for x in partial]
        partial = Any[]
        s, meas = score(r, spec), measure(r, n)
        miss = maximum(maximum(abs(meas.counts[i, j] / max(sum(meas.counts[i, :]), 1) - spec.P[i, j]) for j in successors(spec.P, i)) for i in 1:n)
        @printf("  pass %d  f %.4f  dwell %s  off %.4f  largest branch miss %.4f\n", pass, s.f, round.(s.dwell; sigdigits=4), s.off, miss)
        flush(stdout)
        push!(trace, (; pass, out=copy(out), s.f, s.dwell, s.off, miss))
        for i in 1:n
            isfinite(s.dwell[i]) || continue                # a state not visited this pass keeps its rates
            o, k = outs[i], length(outs[i])
            seen = max(sum(meas.counts[i, :]), 1)
            share = [max(meas.counts[i, edges[e][2]] / seen, 1e-3) for e in o]
            saturated = any(e -> meas.counts[i, edges[e][2]] == 0, o)
            want = [spec.P[i, edges[e][2]] for e in o]
            x = asinh.(out[o] ./ c)
            y = [-log(s.dwell[i]); log.(share[1:k-1]) .- log(share[k])]
            goal = [-log(spec.τ[i]); log.(want[1:k-1]) .- log(want[k])]
            # a branch with no visits gives no slope worth learning from, in this pass or the next
            if !saturated && !isempty(px[i]) && all(isfinite, py[i]) && sum(abs2, x .- px[i]) > 0.02^2
                Δx, Δy = x .- px[i], y .- py[i]
                J[i] .+= (Δy .- J[i] * Δx) * Δx' ./ sum(abs2, Δx)
                J[i] .= clamp.(J[i], -10.0, 10.0)
                (cond(J[i]) > 100 || all(iszero, J[i][1, :])) && (J[i] = start_J(o))
            end
            px[i], py[i] = x, saturated ? fill(NaN, k) : y
            step = late ? 0.3 : 1.0
            # `cap` bounds every out-rate's size (Phase 7: 2 × contract, agreed 2026-11-13), so a state
            # whose transits already take nearly its whole dwell can't drive one rate to infinity; the
            # bound holds on the excitable (negative) side too
            out[o] = clamp.(c .* sinh.(x .+ clamp.(J[i] \ (goal .- y), -step, step)), -cap, cap)
        end
        done = pass
        save()
    end
    model = build === nothing ? sync_model(spec, L, out; fast, contract, arrive, floor) : build(out)
    return (; model, out, fast, contract, arrive, floor, trace)
end
