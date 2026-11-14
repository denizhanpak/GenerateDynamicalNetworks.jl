# Hopfield backend: a Hopfield network with the graph on the vertices of the cube.
#
# Chossat & Krupa (2016) write the Hopfield firing-rate equation on the cube [−1, 1]^n as
#     ẋ_j = (1 − x_j²) (λ (Jx)_j + λ b_j − f(x_j)),
# whose faces are invariant, so the 2^n vertices are equilibria; with f(x) = x at first order and
# x_j = cos ψ_j this becomes
#     ψ̇_j = sin ψ_j (g_j + k_j cos ψ_j),     g_j = −(b_j + Σ_{m≠j} J_jm cos ψ_m),
# the synchrony backend's equation with pairwise coupling only (λ absorbed into J and b). Noise acts on
# each ψ_j, so x stays in the cube.
#
# At vertex v (x_j = ±1) the rate along j is r_j(v) = k_j − x_j (b_j + Σ_{m≠j} J_jm x_m): linear in
# neuron j's row, so each neuron is designed on its own. The two ends of a cube edge along j share every
# other x_m, so r_j(v) + r_j(flip_j v) = 2k_j: a departure at rate r arrives at rate 2k_j − r. Choosing
# k_j = (slowest departure along j − contract)/2 makes every arrival contract at least at `contract`.
#
# Graph states sit on vertices and edges are paths of single flips through transit vertices, laid out
# by the synchrony backend's search; time at a transit counts toward the state being left. Per neuron, a
# small constrained least-squares problem (JuMP + OSQP) sets each departure rate exactly, keeps every
# other used vertex contracting by at least `margin` (aiming at −contract), and, softly, points unused
# neighbours of the network back toward it.

"""
    HopModel

A Hopfield network of `n` neurons in phase form, `ψ̇_j = sin ψ_j (k_j cos ψ_j − b_j − Σ_m J_jm cos ψ_m)`,
with the graph's states on vertices of the cube: `node_of[v]` is the state at vertex `v` (0 for
transits and unused vertices) and `corner_of[i]` the vertex of state `i`. `α` bounds every rate, for
the step. Built by `fit(spec, Hopfield())` or `HopModel(spec, out)`.
"""
struct HopModel <: AbstractNetworkModel
    P::Matrix{Float64}
    visible::Int
    α::Float64
    n::Int
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
    hop_design(L, out; fast, contract, margin=contract/4, soft=10, ridge=1e-6) -> (; J, b, k, worst)

Neuron rows from out-rates: for each neuron j, k_j from its slowest departure, then (b_j, J_j·) by
least squares on the used vertices' rates subject to the departures (equalities) and to every other
used vertex's rate along j being ≤ −margin. Unused vertices next to the network are asked, softly, to
leave toward it at ≥ margin. `worst` is the largest relative violation of the hard conditions.

Solved by OSQP to 1e-9 with a fixed `adaptive_rho_interval = 25`: OSQP's default sets that interval from
measured setup time, which made the design differ from run to run (by up to 7e-10 relative).
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
        set_attribute(model, "adaptive_rho_interval", 25)
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

"""The designed network for out-rates `out` on layout `L`, and the design's worst violation."""
function hop_model(spec, L, out; fast, contract, margin=contract / 4)
    d = hop_design(L, out; fast, contract, margin)
    N = L.M
    node_of = zeros(Int, 2^N)
    for (i, v) in enumerate(L.corner_of)
        node_of[v] = i
    end
    α = maximum(abs(d.k[j]) + abs(d.b[j]) + sum(abs, d.J[j, :]) for j in 1:N)
    return HopModel(Matrix{Float64}(spec.P), nstates(spec), α, N, d.J, d.b, d.k, node_of, copy(L.corner_of), float(spec.noise)), d.worst
end

function hop_drift(ψ::SVector{N}, q, t) where {N}
    x = cos.(ψ)
    h = q.J * x
    return SVector{N}(ntuple(j -> sin(ψ[j]) * (-(q.b[j] + h[j]) + q.k[j] * x[j]), Val(N)))
end

hop_vertex(ψ) = 1 + sum((cos(ψ[j]) < 0) << (j - 1) for j in eachindex(ψ))

drift_function(::HopModel) = hop_drift
parameters(m::HopModel) = (; J=SMatrix{m.n,m.n}(m.J), b=SVector{m.n}(m.b), k=SVector{m.n}(m.k), m.ε, node_of=Tuple(m.node_of))
"""Next to the first state's vertex, a little off every neuron's exact value."""
initial_state(m::HopModel) = let v = m.corner_of[1]
    SVector{m.n}(ntuple(j -> corner_bit(v, j) == 1 ? π - 0.05 : 0.05, m.n))
end
speed(m::HopModel) = m.α
in_bounds(::HopModel, ψ) = all(isfinite, ψ)
frame_state(::HopModel, q, ψ) = q.node_of[hop_vertex(ψ)]

# ── Layout and seed ──────────────────────────────────────────────────────────────────────────

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

function hop_search_layout(spec; Mmax=7, tries=3000, keep=40, seed=1, tol=1e-3)
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
            return L
        end
    end
    throw(ArgumentError("no layout with at most $Mmax neurons meets the pairwise design"))
end

"""
    hop_layout(spec; Msearch=7, tries=3000, keep=40, seed=1, tol=1e-3) -> SyncLayout

The synchrony backend's search for the fewest neurons, keeping only layouts whose every neuron's design
is met (worst violation below `tol`) at the rule-of-thumb rates, cheapest first, up to `Msearch` neurons;
if none is met, the one-hot layout with relay neurons. Random layouts of a bigger cube get less feasible,
not more (pairwise flips must be threshold functions of the other neurons), so the search stops early
and the structured layout adds the neurons.
"""
function hop_layout(spec; Msearch::Integer=7, tries::Integer=3000, keep::Integer=40, seed::Integer=1, tol::Real=1e-3)
    rough, c = rough_rates(spec)
    try
        return hop_search_layout(spec; Mmax=Msearch, tries, keep, seed, tol)
    catch err
        err isa ArgumentError || rethrow()
    end
    L = onehot_layout(spec.P)
    hop_design(L, rough; fast=c, contract=c).worst < tol || throw(ArgumentError("the one-hot layout doesn't meet the pairwise design either"))
    return L
end

"""Each edge's arrival contraction once the rows are built: contract + out_e − slowest departure along its neuron."""
function edge_contracts(L, out; fast, contract)
    steps = path_steps(L)
    slow = Dict(j => minimum((s == 1 ? out[e] : fast) for (v, jj, e, s) in steps if jj == j) for j in unique(st[2] for st in steps))
    first_step = Dict(e => j for (v, j, e, s) in steps if s == 1)
    return [contract + out[e] - slow[first_step[e]] for e in eachindex(out)]
end

"""
Seed out-rates from the race of 1-D exits ([`race_seed_rates`](@ref)), then a few rounds with each
exit's own arrival contraction, which depends on the slowest departure along its neuron. The race uses
noise ε/√2, since here one neuron carries the noise where the synchrony backend has two oscillators.
"""
function hop_seed(spec, L; rounds=4)
    rough, c = rough_rates(spec)
    η = spec.noise / sqrt(2)
    n, edges = nstates(spec), edge_list(spec.P)
    out = race_seed_rates(spec, rough, c, η)
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
    return out
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

"""Raised inside a fit when a pass's rates have no pairwise design (a neuron's quadratic program fails)."""
struct NoDesign <: Exception end

"""
    HopModel(spec, out)

The model with out-rates `out` (one per edge, in `edge_list` order) on the layout `fit` uses
([`hop_layout`](@ref)), with the transit rate and contraction from [`rough_rates`](@ref).
"""
function HopModel(spec::TransitionSpec, out::AbstractVector)
    _, c = rough_rates(spec)
    return first(hop_model(spec, hop_layout(spec), out; fast=c, contract=c))
end

"""
    Hopfield(; passes=12, base=100_000)

The Hopfield backend's fit. The layout ([`hop_layout`](@ref)) puts states on vertices of the cube of
the fewest neurons whose pairwise design works. Every edge's departure rate starts from the race of
each state's 1-D exits ([`hop_seed`](@ref)), which also gives the starting response; then `passes`
correction passes ([`correct_rates`](@ref)) with every rate capped at 2× the contraction. The transit
rate and contraction are 4× the largest rule-of-thumb rate. If a pass's rates have no pairwise design,
the fit warns and returns the last pass that had one.

    fit(spec, Hopfield(); checkpoint=nothing, deadline=Inf, verbose=false) -> HopModel
"""
struct Hopfield <: AbstractBackend
    passes::Int
    base::Int
end
Hopfield(; passes::Integer=12, base::Integer=100_000) = Hopfield(passes, base)

"""
    keep_last_design(run, build, file)

`run()`, except that if it throws [`NoDesign`](@ref) the model is rebuilt from the last pass saved in
the checkpoint `file`, with a warning. With no pass saved (the seed itself had no design) it throws an
`ArgumentError`.
"""
function keep_last_design(run, build, file)
    try
        return run()
    catch err
        err isa NoDesign || rethrow()
        # pass 1 is built from the seed before anything is saved: then there is no pass to fall back on
        st = isfile(file) ? deserialize(file) : nothing
        (st === nothing || isempty(st.trace)) &&
            throw(ArgumentError("the seed rates have no pairwise design (the layout is checked at the rule-of-thumb rates only)"))
        last = st.trace[end]
        @warn "no pairwise design for the rates after pass $(last.pass); keeping pass $(last.pass)"
        return build(last.out)
    end
end

function fit(spec::TransitionSpec, b::Hopfield; checkpoint=nothing, deadline::Real=Inf, verbose::Bool=false)
    L = hop_layout(spec)
    _, c = rough_rates(spec)
    out0 = hop_seed(spec, L)
    function build(out)
        m = first(hop_model(spec, L, out; fast=c, contract=c))
        all(isfinite, m.J) && all(isfinite, m.b) || throw(NoDesign())
        return m
    end
    # the passes so far are needed to fall back on, so a fit without a checkpoint keeps a temporary one
    file = checkpoint === nothing ? tempname() : checkpoint
    try
        return keep_last_design(build, file) do
            r = correct_rates(spec, build, out0; b.passes, b.base, response=hop_response(L, c, spec.noise / sqrt(2)), cap=2c,
                              checkpoint=file, deadline, verbose)
            build(r.out)
        end
    finally
        checkpoint === nothing && rm(file; force=true)
    end
end
