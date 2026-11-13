# Phase 7: a Hopfield network with the graph on the vertices of the cube.
#
# Chossat & Krupa (2016) write the Hopfield firing-rate equation on the cube [−1, 1]^n as
#     ẋ_j = (1 − x_j²) (λ (Jx)_j + λ b_j − f(x_j)),
# whose faces are invariant, so the 2^n vertices are equilibria; they regularise f = arctanh by its
# Taylor polynomial and use f(x) = x at first order. With x_j = cos ψ_j this becomes
#     ψ̇_j = sin ψ_j (g_j + k_j cos ψ_j),     g_j = −(b_j + Σ_{m≠j} J_jm cos ψ_m),
# the Phase 6 equation with pairwise coupling only (λ absorbed into J and b, k_j = 1 − λJ_jj).
# Noise η acts on each ψ_j (choice 2, agreed 2026-11-11): x stays in the cube, and in x the noise is
# scaled by √(1 − x²).
#
# At vertex v (x_j = ±1) the rate along j is r_j(v) = k_j − x_j (b_j + Σ_{m≠j} J_jm x_m): linear in
# neuron j's row, so each neuron is designed on its own (choice 1). Unlike Phase 6, the two ends of a
# cube edge along j share every other x_m, so r_j(v) + r_j(flip_j v) = 2k_j: a departure at rate r
# arrives at rate 2k_j − r. Choosing k_j = (slowest departure along j − contract)/2 makes every
# arrival contract at least at `contract`.
#
# Graph states sit on vertices and edges are paths of single flips through transit vertices, laid out
# by Phase 6's search; time at a transit counts toward the state being left. Per neuron, a small
# constrained least-squares problem (JuMP + OSQP, choice 3) sets each departure rate exactly, keeps
# every other used vertex contracting by at least `margin` (aiming at −contract), and, softly, points
# unused neighbours of the network back toward it. A layout is accepted only if every neuron's
# problem is met; otherwise the search adds a neuron.
include(joinpath(@__DIR__, "..", "phase6", "seed1d.jl"))     # synchrony.jl (layouts, fit_sync) and the 1-D race
using JuMP, OSQP

struct HopModel
    P::Matrix{Float64}
    visible::Int
    α::Vector{Float64}          # bound on every rate, for the step-size rule
    n::Int                      # neurons
    J::Matrix{Float64}          # J[j, m]: m acts on j; zero diagonal
    b::Vector{Float64}
    k::Vector{Float64}
    node_of::Vector{Int}
    corner_of::Vector{Int}
    ε::Float64
end

xbit(v, j) = corner_bit(v, j) == 1 ? -1.0 : 1.0          # D (bit set) is x = −1, ψ = π

"""Rate along neuron j at vertex v."""
vertex_rate(J, b, k, v, j) = k[j] - xbit(v, j) * (b[j] + sum(J[j, m] * xbit(v, m) for m in axes(J, 2) if m != j; init=0.0))
hop_rates(m::HopModel) = [vertex_rate(m.J, m.b, m.k, v, j) for j in 1:m.n, v in 1:2^m.n]

"""Every step of every path: (vertex, neuron, edge, position in path)."""
path_steps(L) = [(p[s], only(j for j in 1:L.M if flip(p[s], j) == p[s+1]), e, s) for (e, p) in enumerate(L.paths) for s in 1:length(p)-1]

"""
Neuron rows from out-rates: for each neuron j, k_j from its slowest departure, then (b_j, J_j·) by
least squares on the used vertices' rates subject to the departures (equalities) and to every other
used vertex's rate along j being ≤ −margin. Unused vertices next to the network are asked, softly, to
leave toward it at ≥ margin. Returns the rows and the worst violation of the hard conditions.
"""
function hop_design(L, out; fast, contract, margin=contract / 4, soft=10.0, ridge=1e-6)
    n = L.M
    steps = path_steps(L)
    used = Set(reduce(vcat, L.paths))
    J, b, k = zeros(n, n), zeros(n), zeros(n)
    worst = 0.0
    for j in 1:n
        dep = [(v, s == 1 ? out[e] : fast) for (v, jj, e, s) in steps if jj == j]
        k[j] = isempty(dep) ? -contract / 2 : (minimum(last, dep) - contract) / 2
        depv = Set(first.(dep))
        others = [v for v in used if !(v in depv)]
        back = unique([u for v in used for u in (flip(v, j),) if !(u in used)])   # unused, one flip away along j
        model = Model(OSQP.Optimizer)
        set_silent(model)
        set_attribute(model, "eps_abs", 1e-9)
        set_attribute(model, "eps_rel", 1e-9)
        set_attribute(model, "max_iter", 200_000)
        set_attribute(model, "polishing", true)
        @variable(model, bj)
        @variable(model, Jj[1:n])
        @variable(model, s[1:length(back)] >= 0)
        rate(v) = k[j] - xbit(v, j) * (bj + sum(Jj[m] * xbit(v, m) for m in 1:n if m != j; init=0.0))
        @constraint(model, Jj[j] == 0)
        for (v, r) in dep
            @constraint(model, rate(v) == r)
        end
        for v in others
            @constraint(model, rate(v) <= -margin)
        end
        for (i, u) in enumerate(back)
            @constraint(model, rate(u) + s[i] >= margin)
        end
        @objective(model, Min, sum((rate(v) + contract)^2 for v in others; init=0.0) / contract^2 +
                               soft * sum(s .^ 2; init=0.0) / contract^2 + ridge * (bj^2 + sum(Jj .^ 2)))
        optimize!(model)
        b[j], J[j, :] = value(bj), value.(Jj)
        J[j, j] = 0.0
        for (v, r) in dep
            worst = max(worst, abs(vertex_rate(J, b, k, v, j) - r) / max(abs(r), margin))
        end
        for v in others
            worst = max(worst, (vertex_rate(J, b, k, v, j) + margin) / margin)
        end
    end
    return (; J, b, k, worst)
end

function hop_model(spec, L, out; fast, contract, margin=contract / 4)
    d = hop_design(L, out; fast, contract, margin)
    n, N = size(spec.P, 1), L.M
    node_of = zeros(Int, 2^N)
    for (i, v) in enumerate(L.corner_of)
        node_of[v] = i
    end
    α = [maximum(abs(d.k[j]) + abs(d.b[j]) + sum(abs, d.J[j, :]) for j in 1:N)]
    return HopModel(spec.P, n, α, N, d.J, d.b, d.k, node_of, copy(L.corner_of), spec.ε₀), d.worst
end

function drift_hop(ψ::SVector{N}, q, t) where {N}
    x = cos.(ψ)
    h = q.J * x
    return SVector{N}(ntuple(j -> sin(ψ[j]) * (-(q.b[j] + h[j]) + q.k[j] * x[j]), Val(N)))
end

function read_hop!(ig)
    q = ig.p
    ig.iter % q.every == 0 || return nothing
    ψ = ig.u
    if !all(isfinite, ψ)
        q.V.failed = true
        terminate!(ig)
        return nothing
    end
    v = 1
    for j in 1:length(ψ)
        cos(ψ[j]) < 0 && (v += 1 << (j - 1))
    end
    s = q.node_of[v]
    if s != 0 && (isempty(q.V.state) || last(q.V.state) != s)
        push!(q.V.state, s)
        push!(q.V.t, ig.t)
    end
    return nothing
end
const HOP_FRAME = DiscreteCallback((u, t, ig) -> true, read_hop!; save_positions=(false, false))

hop_params(m::HopModel) = (; J=SMatrix{m.n,m.n}(m.J), b=SVector{m.n}(m.b), k=SVector{m.n}(m.k), m.ε, node_of=Tuple(m.node_of))

"""Start next to the first state's vertex, a little off every neuron's exact value."""
hop_start(m::HopModel) = let v = m.corner_of[1]
    SVector{m.n}(ntuple(j -> corner_bit(v, j) == 1 ? π - 0.05 : 0.05, m.n))
end

function simulate_frames(m::HopModel; T, seeds, sample, dtscale=0.01)
    every, dt = frame_steps(m, sample, dtscale)
    prob = SDEProblem(drift_hop, noise_flat, hop_start(m), (0.0, T), merge(hop_params(m), (; every, V=Frames())))
    ens = EnsembleProblem(prob;
        prob_func=(pr, ctx) -> remake(pr; seed=UInt64(seeds[ctx.sim_id]), p=(; pr.p..., V=Frames())),
        output_func=(sol, ctx) -> ((state=sol.prob.p.V.state, t=sol.prob.p.V.t, failed=sol.prob.p.V.failed), false))
    return solve(ens, EM(), EnsembleThreads(); trajectories=length(seeds), merge(quiet(dt), (; callback=HOP_FRAME))...).u
end

function classify(m::HopModel; spec, kw...)
    decode(ψ) = (v = 1 + sum((cos(ψ[j]) < 0) << (j - 1) for j in 1:m.n); m.node_of[v] == 0 ? -v : m.node_of[v])
    return classify(drift_hop, hop_params(m), hop_start(m), decode; noise=spec.ε₀, dt=0.02 / m.α[1], kw...)
end

# ── Layout and seed ──────────────────────────────────────────────────────────────────────────

"""Two-way pairs i ↔ j with i < j, as the edges j → i that go through a relay neuron (as in Phase 5)."""
relayed_edges(P) = [(j, i) for i in axes(P, 1) for j in axes(P, 1) if i < j && P[i, j] > 0 && P[j, i] > 0]

"""
One neuron per state (state i is the vertex with only bit i set) plus one relay neuron per two-way
pair. Edge i → j switches j on and then i off: e_i → e_i + e_j → e_j. A relayed edge j → i goes
e_j → e_j + e_r → e_r → e_r + e_i → e_i, since the direct transit vertex is taken by i → j.
"""
function onehot_layout(P)
    n, rel = size(P, 1), relayed_edges(P)
    e(bits...) = 1 + sum(1 << (b - 1) for b in bits)
    paths = map(edge_list(P)) do (i, j)
        r = findfirst(==((i, j)), rel)
        r === nothing ? [e(i), e(i, j), e(j)] : (k = n + r; [e(i), e(i, k), e(k), e(k, j), e(j)])
    end
    return SyncLayout(n + length(rel), [e(i) for i in 1:n], paths)
end

"""
The layout rule: Phase 6's search for the fewest neurons, keeping only layouts whose every neuron's
design is met (worst violation below `tol`) at the rule-of-thumb rates, cheapest first, up to `Msearch`
neurons; if none is met, the one-hot layout with relay neurons. Random layouts of a bigger cube get
less feasible, not more (pairwise flips must be threshold functions of the other neurons), so the
search stops early and the structured layout adds the neurons.
"""
function hop_layout(spec; Msearch=7, tries=3000, keep=40, seed=1, tol=1e-3)
    P = spec.P
    rough, c = rough_rates(spec)
    try
        return hop_search_layout(spec; Mmax=Msearch, tries, keep, seed, tol)
    catch err
        err isa ArgumentError || rethrow()
    end
    L = onehot_layout(P)
    hop_design(L, rough; fast=c, contract=c).worst < tol || throw(ArgumentError("the one-hot layout doesn't meet the pairwise design either"))
    return L
end

function hop_search_layout(spec; Mmax=7, tries=3000, keep=40, seed=1, tol=1e-3, check=nothing)
    P = spec.P
    rng = Xoshiro(seed)
    n = size(P, 1)
    rough, c = rough_rates(spec)
    lo = max(2, ceil(Int, log2(n)), maximum(i -> count(>(0), P[i, :]), 1:n))
    for M in lo:Mmax
        found = Dict{Any,Tuple{Int,SyncLayout}}()
        for _ in 1:tries
            L = try_layout(P, M, rng)
            L === nothing && continue
            key = (L.corner_of, L.paths)
            haskey(found, key) || (found[key] = (sum(length, L.paths), L))
        end
        for (_, L) in sort(collect(values(found)); by=first)[1:min(keep, length(found))]
            hop_design(L, rough; fast=c, contract=c).worst < tol || continue
            check === nothing || check(L) || continue
            return L
        end
    end
    throw(ArgumentError("no layout with at most $Mmax neurons meets the pairwise design"))
end

"""Each edge's arrival contraction once the rows are built: contract + out_e − slowest departure along its neuron."""
function edge_contracts(L, out; fast, contract)
    steps = path_steps(L)
    slow = Dict(j => minimum((s == 1 ? out[e] : fast) for (v, jj, e, s) in steps if jj == j) for j in unique(st[2] for st in steps))
    first_step = Dict(e => j for (v, j, e, s) in steps if s == 1)
    return [contract + out[e] - slow[first_step[e]] for e in eachindex(out)]
end

"""
Seed out-rates from the 1-D race (seed1d.jl), with each exit's own arrival contraction, which
depends on the slowest departure along its neuron: a few rounds of solve-then-update. The race uses
η/√2, since here one neuron carries the noise where Phase 6 had two oscillators.
"""
function hop_seed(spec, L; rounds=4)
    rough, c = rough_rates(spec)
    η = spec.ε₀ / sqrt(2)
    n, edges = size(spec.P, 1), edge_list(spec.P)
    out, _ = seed_out_race(merge(spec, (; ε₀=η)); contract=c, start=rough)
    for _ in 1:rounds
        cs = edge_contracts(L, out; fast=c, contract=c)
        for i in 1:n
            o = [e for (e, (a, _)) in enumerate(edges) if a == i]
            k = length(o)
            want = [spec.P[i, edges[e][2]] for e in o]
            goal = [-log(spec.τ[i]); log.(want[1:k-1]) .- log(want[k])]
            λ = out[o]
            for _ in 1:30
                y = race_outputs(λ, cs[o], η)
                maximum(abs, y .- goal) < 1e-4 && break
                Jm = zeros(k, k)
                for j in 1:k
                    d = zeros(k); d[j] = 1e-4 * max(1.0, abs(λ[j]))
                    Jm[:, j] = (race_outputs(λ .+ d, cs[o], η) .- race_outputs(λ .- d, cs[o], η)) ./ (2d[j])
                end
                step = Jm \ (goal .- y)
                λ = clamp.(λ .+ min(1.0, 2.0 / maximum(abs, step)) .* step, -c, c)
            end
            out[o] = λ
        end
    end
    return out, c
end

"""Starting Broyden matrix from the race with each exit's own contraction, in asinh(λ/c) coordinates."""
function hop_response(L, contract, η)
    return function (o, out, c)
        cs = edge_contracts(L, out; fast=contract, contract)[o]
        k = length(o)
        x = asinh.(out[o] ./ c)
        Jm = zeros(k, k)
        for j in 1:k
            d = zeros(k); d[j] = 1e-3
            Jm[:, j] = (race_outputs(c .* sinh.(x .+ d), cs, η) .- race_outputs(c .* sinh.(x .- d), cs, η)) ./ 2e-3
        end
        return Jm
    end
end

"""Raised when a fit's rates have no pairwise design (a neuron's quadratic program has no solution)."""
struct NoDesign <: Exception end

"""Fit: race seed, then Phase 6's per-state correction passes on the Hopfield network. A rerun that
resumes from `checkpoint` skips the seed, which the checkpoint already holds."""
function fit_hop(spec, L; checkpoint=nothing, kw...)
    _, c = rough_rates(spec)
    # the seed is kept beside the checkpoint so a resumed fit gets the same one (fit_sync's asinh
    # scale comes from it) and reports it; the rates themselves resume from the checkpoint
    seed_file = checkpoint === nothing ? nothing : checkpoint * ".seed"
    out0 = seed_file !== nothing && isfile(seed_file) ? deserialize(seed_file) : first(hop_seed(spec, L))
    seed_file === nothing || serialize(seed_file, out0)
    function build(out)
        m = first(hop_model(spec, L, out; fast=c, contract=c))
        all(isfinite, m.J) && all(isfinite, m.b) || throw(NoDesign())
        return m
    end
    # out-rates capped at 2 × contract (agreed 2026-11-13): on shared-solvable at 0.1 a state's transits
    # alone took nearly its dwell and the fit drove its one exit rate past 400 until no design existed
    f = try
        merge(fit_sync(spec, L; seed=out0, response=hop_response(L, c, spec.ε₀ / sqrt(2)), build, checkpoint, cap=2c, kw...),
              (; stopped=nothing))
    catch err
        # the next pass's rates have no pairwise design: keep the last pass that was simulated
        # (agreed 2026-11-13); its rates are the ones its trace entry measured
        err isa NoDesign && checkpoint !== nothing || rethrow()
        # pass 1 is built from the seed, before anything is saved: then there is no pass to fall back on
        st = isfile(checkpoint) ? deserialize(checkpoint) : nothing
        (st === nothing || isempty(st.trace)) &&
            error("the seed rates have no pairwise design (layout checked at the rule-of-thumb rates only)")
        last = st.trace[end]
        # `stopped` is the first pass whose rates have none; after the final pass that is the returned update
        println("  no pairwise design for the rates after pass $(last.pass); keeping pass $(last.pass)")
        (; model=build(last.out), out=copy(last.out), fast=c, contract=c, trace=st.trace, stopped=last.pass + 1)
    end
    _, worst = hop_model(spec, L, f.out; fast=c, contract=c)
    return (; f..., seed=out0, worst)
end
