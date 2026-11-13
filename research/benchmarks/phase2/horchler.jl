# Phase 2 prototype of the HorchlerLV backend. The notebook and the sweep scripts both include
# this file, so there is one copy of the code. It becomes `src/` once the user signs it off.
#
# Two forms of the same network:
#   :x  dx = x ⊙ (α − ρx) dt + ε dW, clamped at x ≥ 0    (Horchler et al. 2015, SHCTools)
#   :u  du = u ⊙ (α − ρu²) dt + ε dW, activity x = u²     (no clamp: the field is odd in each uᵢ)
# Deterministically the u form is the x form with every rate doubled, so the escape rates
# match, the neighbourhood edge κβ in x sits at √(κβ) in u, and the transition halves.

using SHCTools, StochasticDiffEq, StaticArrays, Graphs, Distributions, QuadGK, Roots
using LinearAlgebra, Random, Statistics

# ── Graph rules and repair (Phase 1) ────────────────────────────────────────────────────

self_loops(P) = [i for i in axes(P, 1) if P[i, i] > 0]
two_cycles(P) = [(i, j) for i in axes(P, 1) for j in axes(P, 1) if i < j && P[i, j] > 0 && P[j, i] > 0]
delta_cliques(P) = let A = P .> 0, n = axes(A, 1)
    [(a, b, c) for a in n for b in n for c in n if allunique((a, b, c)) && A[a, b] && A[a, c] && A[b, c]]
end

"""
Branch equations in log ε: one per two-way branch, log εⱼ − log εₖ = log tan(π pⱼ / 2), and
equal noise across an even wider branch. Returns `(A, b)`, or `nothing` for an uneven branch
wider than two, which has no formula.
"""
function branch_equations(P)
    n = size(P, 1)
    rows, b = Vector{Float64}[], Float64[]
    equation!(j, k, c) = (r = zeros(n); r[j] = 1; r[k] = -1; push!(rows, r); push!(b, c))
    for i in 1:n
        s = findall(>(0), P[i, :])
        if length(s) == 2
            equation!(s[1], s[2], log(tan(π * P[i, s[1]] / 2)))
        elseif length(s) > 2
            allequal(P[i, s]) || return nothing
            foreach(k -> equation!(s[1], k, 0.0), s[2:end])
        end
    end
    return isempty(rows) ? (zeros(0, n), Float64[]) : (reduce(vcat, r' for r in rows), b)
end

"""How far the branches are from sharing one noise value per node (0: they can; Inf: no formula)."""
function noise_residual(P)
    eqs = branch_equations(P)
    eqs === nothing && return Inf
    A, b = eqs
    isempty(b) && return 0.0
    return norm(A * (A \ b) - b)
end

"""Noise per node: ε₀ times the minimum-norm solution of the branch equations."""
function branch_noise(P, ε₀)
    A, b = branch_equations(P)
    isempty(b) && return fill(float(ε₀), size(P, 1))
    return ε₀ .* exp.(A \ b)
end

"""
Put a hidden state h on edge a → b. h takes half of a's dwell and a gives up the same amount,
weighted by how often it takes that edge, so a's mean dwell to the next visible state is unchanged.
"""
function subdivide(P, τ, a, b)
    h, n = size(P, 1) + 1, size(P, 1)
    Q = [P zeros(n); zeros(h)']
    Q[a, h], Q[h, b], Q[a, b] = Q[a, b], 1, 0
    τh = τ[a] / 2
    σ = [τ; τh]
    σ[a] -= Q[a, h] * τh
    return Q, σ
end

"""
Repair 2-cycles, Δ-cliques and inconsistent shared successors with hidden states, numbered after
the visible ones. `ok` is false if something is left that no hidden state fixes.
"""
function repair(P, τ)
    Q, σ, visible = float(P), float(τ), size(P, 1)
    reasons = String[]
    while true
        if !isempty(two_cycles(Q))
            a, b = first(two_cycles(Q))
            Q, σ = subdivide(Q, σ, a, b)
            push!(reasons, "2-cycle $a ⇄ $b")
        elseif !isempty(delta_cliques(Q))
            a, b, c = first(delta_cliques(Q))
            Q, σ = subdivide(Q, σ, b, c)
            push!(reasons, "Δ-clique $a → $b → $c")
        elseif isfinite(noise_residual(Q)) && noise_residual(Q) > 1e-9
            branching = [i for i in axes(Q, 1) if count(>(0), Q[i, :]) >= 2]
            shared = [(i, j) for i in branching for j in findall(>(0), Q[i, :])
                      if any(k != i && Q[k, j] > 0 for k in branching)]
            isempty(shared) && return (; P=Q, τ=σ, visible, reasons, ok=false)
            Q, σ = subdivide(Q, σ, shared[1]...)
            push!(reasons, "shared successor $(shared[1][2]) of $(shared[1][1])")
        else
            ok = isempty(self_loops(Q)) && isfinite(noise_residual(Q)) && all(>(0), σ)
            return (; P=Q, τ=σ, visible, reasons, ok)
        end
    end
end

# ── Design ──────────────────────────────────────────────────────────────────────────────

"""
Single-channel noise with the same mean escape time as racing channels with noise `εs`
(SHCTools notebook, step 4).
"""
function effective_noise(εs)
    length(εs) == 1 && return float(only(εs))
    Z = [truncated(Normal(0, e), 0, Inf) for e in εs]
    top = log(maximum(εs)) + 8
    I, _ = quadgk(u -> prod(cdf(z, exp(u)) for z in Z), log(minimum(εs)) - 40, top; rtol=1e-8)
    return exp(top - I + (Base.MathConstants.γ + log(2)) / 2)
end

"""Neighbourhood edge in the form's own variable, and the deterministic transition time."""
edge_size(form, κ, β) = form === :x ? κ * β : sqrt(κ * β)
transition_time(form, κ, α) = (form === :x ? 2 : 1) * log(1 / κ - 1) / α

"""Designed mean dwell: Stone-Holmes passage (leading order, as `lv_params`) plus the transition."""
designed_dwell(form, α, ν, ε, β, κ) =
    passagetime(edge_size(form, κ, β), ε, α / ν, α) + transition_time(form, κ, α)

"""
One shared α, and ν per state setting its dwell (`dwell_rates` in the SHCTools notebook, with
the inversion done by root finding so that both forms share it).
"""
function dwell_rates(form, τ, ε_eff, β; κ=0.1, νmin=2.0)
    # In α the designed dwell is ν/(2α)·(a + ln α), which peaks at α = e^(1−a) and falls below
    # it, so the search starts at the peak (the condition `lv_params` checks with x ≤ −1).
    function αᵢ(i, ν)
        f(logα) = designed_dwell(form, exp(logα), ν, ε_eff[i], β[i], κ) - τ[i]
        k = transition_time(form, κ, 1.0)
        a = 2log(edge_size(form, κ, β[i]) / ε_eff[i]) + Base.MathConstants.γ - log1p(1 / ν) + log(4 / ν) + 2k / ν
        lo = 1 - a
        f(lo) > 0 || throw(ArgumentError("state $i: dwell $(τ[i]) is longer than the $(round(f(lo) + τ[i]; sigdigits=3)) " *
            "that noise $(round(ε_eff[i]; sigdigits=2)) allows"))
        hi = lo + 1
        while f(hi) > 0
            hi += 2
        end
        return exp(find_zero(f, (lo, hi)))
    end
    α = maximum(αᵢ(i, νmin) for i in eachindex(τ))
    # The leading-order passage time ν/(2α)·(c − ln ν) peaks at ν = e^(c−1), so at a given noise
    # and α there is a longest dwell. The state that set α sits exactly at νmin, so the bracket
    # starts below it.
    function νᵢ(i)
        f(v) = designed_dwell(form, α, v, ε_eff[i], β[i], κ) - τ[i]
        c = 2log(edge_size(form, κ, β[i]) / ε_eff[i]) + log(4α) + Base.MathConstants.γ
        top = min(1e6, exp(c - 1))
        f(top) > 0 || throw(ArgumentError("state $i: dwell $(τ[i]) is longer than the $(round(f(top) + τ[i]; sigdigits=3)) " *
            "that noise $(round(ε_eff[i]; sigdigits=2)) allows at the shared rate α = $(round(α; sigdigits=3))"))
        return find_zero(f, (νmin / 2, top))
    end
    return fill(α, length(τ)), νᵢ.(eachindex(τ))
end

"""ρᵢⱼ from the SHCTools notebook: `P[j, i] > 0` (edge j → i) fills `ρ[i, j]`."""
function graph_rho(P, α, β, ν)
    n = length(α)
    ρ = [i == j ? α[i] / β[i] : (α[i] + α[j]) / β[j] for i in 1:n, j in 1:n]
    for j in 1:n, i in 1:n
        P[j, i] > 0 && (ρ[i, j] = (α[i] - α[j] / ν[j]) / β[j])
    end
    return ρ
end

"""
The whole recipe: repair, branch noise, effective noise, rates, ρ. `τ` here is the dwell to
request, which a correction pass may move away from the target.
"""
function design(P, τ; form=:x, ε₀=1e-5, κ=0.1, νmin=2.0, hidden=true)
    r = hidden ? repair(P, τ) : (; P=float(P), τ=float(τ), visible=size(P, 1), reasons=String[], ok=true)
    r.ok || throw(ArgumentError("graph can't be repaired: $(join(r.reasons, ", "))"))
    n = size(r.P, 1)
    β = ones(n)
    ε = branch_noise(r.P, ε₀)
    ε_eff = [effective_noise(ε[findall(>(0), r.P[i, :])]) for i in 1:n]
    α, ν = dwell_rates(form, r.τ, ε_eff, β; κ, νmin)
    return (; form, P=r.P, τ=r.τ, visible=r.visible, reasons=r.reasons, β, κ, ε, ε_eff, α, ν,
              ρ=graph_rho(r.P, α, β, ν))
end

# ── Simulation ──────────────────────────────────────────────────────────────────────────

drift_x(x, q, t) = x .* (q.α .- q.ρ * x)
drift_u(u, q, t) = u .* (q.α .- q.ρ * (u .* u))
diffusion(u, q, t) = q.ε
activity(form, u) = form === :x ? u : u .* u

"""Visits read as the run goes: enter a visible state when its activity crosses `edge`, hold it until another does."""
mutable struct Visits
    state::Vector{Int}
    t::Vector{Float64}
end
Visits() = Visits(Int[], Float64[])

function read_visits!(ig)
    q = ig.p
    q.form === :x && (ig.u = max.(ig.u, 0.0))
    a = activity(q.form, ig.u)
    v, i = findmax(a[k] for k in 1:q.visible)
    if v > q.edge && (isempty(q.V.state) || last(q.V.state) != i)
        push!(q.V.state, i)
        push!(q.V.t, ig.t)
    end
    return nothing
end
const READ = DiscreteCallback((u, t, ig) -> true, read_visits!; save_positions=(false, false))

function sde_problem(m; T, edge=0.5)
    n = length(m.α)
    S = SVector{n}
    q = (; form=m.form, α=S(m.α), ρ=SMatrix{n,n}(m.ρ), ε=S(m.ε), visible=m.visible, edge, V=Visits())
    x0 = fill(minimum(m.ε), n)
    x0[1] = (1 - m.κ) * m.β[1]
    u0 = m.form === :x ? x0 : sqrt.(x0)
    return SDEProblem(m.form === :x ? drift_x : drift_u, diffusion, S(u0), (0.0, T), q)
end

quiet(dt) = (; dt, adaptive=false, save_everystep=false, save_start=false, save_end=false, dense=false,
               callback=READ, maxiters=typemax(Int))

"""Visits of independent runs with seeds `seeds`, in parallel. `dt` defaults to 0.01/α."""
function simulate(m; T, seeds=1:16, dt=0.01 / maximum(m.α), edge=0.5)
    prob = sde_problem(m; T, edge)
    ens = EnsembleProblem(prob;
        prob_func=(pr, ctx) -> remake(pr; seed=UInt64(seeds[ctx.sim_id]), p=(; pr.p..., V=Visits())),
        output_func=(sol, ctx) -> ((state=sol.prob.p.V.state, t=sol.prob.p.V.t), false))
    return solve(ens, EM(), EnsembleThreads(); trajectories=length(seeds), quiet(dt)...).u
end

# ── Measurement (Phase 1) ───────────────────────────────────────────────────────────────

"""
Dwell statistics pooled over runs. Each run drops its first visit (it starts part-way through)
and its last (no end).
"""
function measure(runs, n)
    samples = [Float64[] for _ in 1:n]
    counts = zeros(Int, n, n)
    for v in runs
        s, t = v.state[2:end], v.t[2:end]
        for k in 1:length(s)-1
            push!(samples[s[k]], t[k+1] - t[k])
            counts[s[k], s[k+1]] += 1
        end
    end
    return (; samples, counts, dwell=[isempty(s) ? NaN : mean(s) for s in samples],
              sem=[length(s) < 2 ? NaN : std(s) / sqrt(length(s)) for s in samples])
end

off_graph(m, P) = sum(m.counts[c] for c in CartesianIndices(P) if P[c] == 0; init=0) / max(sum(m.counts), 1)

"""
Acceptance from the plan: each dwell within `tol`, each branch inside its 95% interval, fewer
than 1% of transitions off the graph.
"""
function verdict(meas, P, τ; tol=0.05)
    n = size(P, 1)
    err = meas.dwell ./ τ .- 1
    branch = Tuple{Int,Int,Float64,Float64,Bool}[]
    for i in 1:n, j in 1:n
        P[i, j] > 0 && count(>(0), P[i, :]) >= 2 || continue
        total = sum(meas.counts[i, :])
        f = meas.counts[i, j] / max(total, 1)
        half = 1.96sqrt(P[i, j] * (1 - P[i, j]) / max(total, 1))
        push!(branch, (i, j, P[i, j], f, abs(f - P[i, j]) <= half))
    end
    off = off_graph(meas, P)
    return (; err, branch, off, pass=all(abs.(err) .<= tol) && all(last.(branch)) && off < 0.01)
end
