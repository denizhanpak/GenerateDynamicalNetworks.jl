# Horchler backend (Horchler, Daltorio, Chiel & Quinn 2015; SHCTools.jl): a Lotka–Volterra network with one
# node per state, designed in closed form from the dwell times. Graphs the design can't take directly (2-cycles,
# Δ-cliques, branches whose noise ratios conflict) are repaired with hidden states. Then correction passes on the
# requested dwells, and if those fail a held-out check, CMA-ES on the raw rates and a short polish.

# ── Graph rules and repair ──────────────────────────────────────────────────────────────────────────────

self_loops(P) = [i for i in axes(P, 1) if P[i, i] > 0]
two_cycles(P) = [(i, j) for i in axes(P, 1) for j in axes(P, 1) if i < j && P[i, j] > 0 && P[j, i] > 0]
delta_cliques(P) = let A = P .> 0, n = axes(A, 1)
    [(a, b, c) for a in n for b in n for c in n if allunique((a, b, c)) && A[a, b] && A[a, c] && A[b, c]]
end

"""
Branch equations in log ε: one per two-way branch, log εⱼ − log εₖ = log tan(π pⱼ / 2), and equal noise across
an even wider branch. Returns `(A, b)`, or `nothing` for an uneven branch wider than two, which has no formula.
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

"""Noise per node: `ε₀` times the minimum-norm solution of the branch equations."""
function branch_noise(P, ε₀)
    A, b = branch_equations(P)
    isempty(b) && return fill(float(ε₀), size(P, 1))
    return ε₀ .* exp.(A \ b)
end

"""
    subdivide(P, τ, a, b) -> (P, τ)

Put a hidden state h on edge a → b. h takes half of a's dwell and a gives up the same amount, weighted by how
often it takes that edge, so a's mean dwell to the next visible state is unchanged.
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
    repair(P, τ) -> (; P, τ, visible, reasons, ok)

Repair 2-cycles, Δ-cliques and inconsistent shared successors with hidden states ([`subdivide`](@ref)),
numbered after the visible ones. `ok` is false if something is left that no hidden state fixes.
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

# ── The model ───────────────────────────────────────────────────────────────────────────────────────────

"""
    LVModel(form, P, visible, κ, β, α, ε, ρ)

A Lotka–Volterra network. In the `:u` form, `du = u ⊙ (α − ρ(u ⊙ u)) dt + ε ⊙ dW`, with activity `x = u²`; in
the `:x` form, `dx = x ⊙ (α − ρx) dt + ε ⊙ dW`, put back to `x ≥ 0` after every step. Deterministically the two
are the same system, with every rate doubled in the `u` form. `P` is the graph the model was built on: its first
`visible` states are the specification's, the rest hidden states added by [`repair`](@ref). A frame is read as
the state of largest |u|; a frame won by a hidden state keeps the last visible state. Runs start near node 1 at
`(1 − κ)β₁`. Built by `fit(spec, Horchler())`.
"""
struct LVModel <: AbstractNetworkModel
    form::Symbol
    P::Matrix{Float64}
    visible::Int
    κ::Float64
    β::Vector{Float64}
    α::Vector{Float64}
    ε::Vector{Float64}
    ρ::Matrix{Float64}
    function LVModel(form::Symbol, P::AbstractMatrix, visible::Integer, κ::Real, β::AbstractVector, α::AbstractVector,
                     ε::AbstractVector, ρ::AbstractMatrix)
        form in (:u, :x) || throw(ArgumentError("form must be :u or :x, not $(repr(form))"))
        n = length(α)
        size(P) == size(ρ) == (n, n) && length(β) == length(ε) == n && 1 <= visible <= n ||
            throw(ArgumentError("P, ρ, β, α and ε must describe the same $n nodes, of which 1 to $n are visible"))
        return new(form, Matrix{Float64}(P), visible, κ, Vector{Float64}(β), Vector{Float64}(α), Vector{Float64}(ε),
                   Matrix{Float64}(ρ))
    end
end

lv_drift_x(x, q, t) = x .* (q.α .- q.ρ * x)
lv_drift_u(u, q, t) = u .* (q.α .- q.ρ * (u .* u))
lv_noise(u, q, t) = q.ε

drift_function(m::LVModel) = m.form === :x ? lv_drift_x : lv_drift_u
noise_function(::LVModel) = lv_noise
parameters(m::LVModel) = (n = length(m.α); (; α=SVector{n}(m.α), ρ=SMatrix{n,n}(m.ρ), ε=SVector{n}(m.ε), m.visible))
function initial_state(m::LVModel)
    n = length(m.α)
    x0 = fill(minimum(m.ε), n)
    x0[1] = (1 - m.κ) * m.β[1]
    return SVector{n}(m.form === :x ? x0 : sqrt.(x0))
end
speed(m::LVModel) = maximum(m.α)
frame_state(::LVModel, q, u) = (i = findmax(abs, u)[2]; i <= q.visible ? i : 0)
project(m::LVModel, u) = m.form === :x ? max.(u, 0.0) : nothing
class_state(::LVModel, q, u) = argmax(abs.(u))
class_start(m::LVModel) = (n = length(m.α); SVector{n}(ntuple(i -> i == 1 ? 0.95 : 1e-3, n)))

# ── Design ──────────────────────────────────────────────────────────────────────────────────────────────

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
    horchler_design(P, τ; form=:u, noise, κ=0.1, νmin=2.0, hidden=true) -> NamedTuple

The closed-form recipe: [`repair`](@ref), branch noise (each two-way branch's odds set by the ratio of its
targets' noise), effective noise per state ([`effective_noise`](@ref)), the shared rate α and saddle values ν
([`dwell_rates`](@ref)), then ρ. `τ` is the dwell to request, which a correction pass may move away from the
target. Throws an `ArgumentError` if the graph can't be repaired or a dwell is longer than the noise allows.
"""
function horchler_design(P, τ; form=:u, noise, κ=0.1, νmin=2.0, hidden=true)
    r = hidden ? repair(P, τ) : (; P=float(P), τ=float(τ), visible=size(P, 1), reasons=String[], ok=true)
    r.ok || throw(ArgumentError("graph can't be repaired: $(join(r.reasons, ", "))"))
    n = size(r.P, 1)
    β = ones(n)
    ε = branch_noise(r.P, noise)
    ε_eff = [effective_noise(ε[findall(>(0), r.P[i, :])]) for i in 1:n]
    α, ν = dwell_rates(form, r.τ, ε_eff, β; κ, νmin)
    return (; form, P=r.P, τ=r.τ, visible=r.visible, reasons=r.reasons, β, κ, ε, ε_eff, α, ν,
              ρ=graph_rho(r.P, α, β, ν))
end

lv_model(d) = LVModel(d.form, d.P, d.visible, d.κ, d.β, d.α, d.ε, d.ρ)

"""A design with its rates replaced: `θ = [log α; log ν]`, one of each per node, hidden ones included."""
function raw_design(seed, θ)
    n = length(seed.α)
    α, ν = exp.(θ[1:n]), exp.(θ[n+1:2n])
    return merge(seed, (; α, ν, ρ=graph_rho(seed.P, α, seed.β, ν)))
end

# ── Correction and polish ───────────────────────────────────────────────────────────────────────────────

"""
    correct_requests(spec, form; passes=6, runs=16, T=1000, dtscale=0.01, base=100_000, start=spec.τ)

Multiplicative correction of the requested dwells, request ← request·(τ / measured), each pass on new seeds.
One trace row per pass, with the pass's design (`nothing` and a note if the formula had none, which ends the
passes).
"""
function correct_requests(spec, form; passes=6, runs=16, T=1000.0, dtscale=0.01, base=100_000, start=spec.τ)
    n = nstates(spec)
    request = float.(start)
    trace = NamedTuple[]
    for pass in 1:passes
        d = try
            horchler_design(spec.P, request; form, noise=spec.noise)
        catch err
            err isa ArgumentError || rethrow()
            push!(trace, (; pass, request=copy(request), dwell=fill(NaN, n), off=NaN, f=Inf, design=nothing, note=sprint(showerror, err)))
            break
        end
        seeds = base+1000pass+1:base+1000pass+runs
        s = score(simulate(lv_model(d); T, seeds, spec.sample, dtscale), spec)
        push!(trace, (; pass, request=copy(request), dwell=s.dwell, off=s.off, f=s.f, design=d, note=""))
        all(isfinite, s.dwell) || break
        request .*= spec.τ ./ s.dwell
    end
    return trace
end

"""
    polish_saddles(d, spec; passes=3, runs=64, T=4000, dtscale=0.01, base=200_000) -> (d, trace)

Correction that needs no formula: νᵢ ← νᵢ·(τᵢ / measuredᵢ) on the visible states, ρ rebuilt. Returns the
design after the last update and one row per pass.
"""
function polish_saddles(d, spec; passes=3, runs=64, T=4000.0, dtscale=0.01, base=200_000)
    n = nstates(spec)
    trace = NamedTuple[]
    for pass in 1:passes
        seeds = base+10_000pass+1:base+10_000pass+runs
        s = score(simulate(lv_model(d); T, seeds, spec.sample, dtscale), spec)
        push!(trace, (; pass, ν=copy(d.ν), dwell=s.dwell, off=s.off, f=s.f))
        ν = copy(d.ν)
        ν[1:n] .*= spec.τ ./ s.dwell
        d = merge(d, (; ν, ρ=graph_rho(d.P, d.α, d.β, ν)))
    end
    return d, trace
end

# ── The backend ─────────────────────────────────────────────────────────────────────────────────────────

"""
    Horchler(form=:u; passes=4, runs=16, T=nothing, budget=500, popsize=12, seconds=300.0)

The Horchler backend's fit, in the `:u` form (the default) or the `:x` form of [`LVModel`](@ref). The
closed-form design ([`horchler_design`](@ref)) gives the first model; `passes` correction passes of `runs` runs
of length `T` (default 125Στ) adjust the dwells it is asked for. If the best of those fails a check on 32
held-out runs, a CMA-ES search over every node's α and ν of at most `budget` evaluations or `seconds` seconds,
then two polish passes on ν ([`polish_saddles`](@ref)). (The prototype's search had no time limit.) A binding
time limit makes the result depend on the machine's speed; `seconds=Inf` limits the search by `budget` alone.

    fit(spec, Horchler(); verbose=false) -> LVModel

Throws an `ArgumentError` if the graph can't be repaired, or if no dwell request has a design (the noise is too
large for the dwells: the `:x` form needs far less noise than the `:u` form).

!!! note "Status: stable"
    Passes the acceptance protocol on most of the benchmark cases; see the Results page of the documentation.
"""
struct Horchler <: AbstractBackend
    form::Symbol
    passes::Int
    runs::Int
    T::Union{Float64,Nothing}
    budget::Int
    popsize::Int
    seconds::Float64
    function Horchler(form::Symbol, passes::Integer, runs::Integer, T::Union{Real,Nothing}, budget::Integer, popsize::Integer,
                      seconds::Real)
        form in (:u, :x) || throw(ArgumentError("form must be :u or :x, not $(repr(form))"))
        passes >= 1 || throw(ArgumentError("passes must be at least 1"))
        return new(form, passes, runs, T === nothing ? nothing : float(T), budget, popsize, seconds)
    end
end
Horchler(form::Symbol=:u; passes::Integer=4, runs::Integer=16, T::Union{Real,Nothing}=nothing, budget::Integer=500,
         popsize::Integer=12, seconds::Real=300.0) = Horchler(form, passes, runs, T, budget, popsize, seconds)

function fit(spec::TransitionSpec, b::Horchler; verbose::Bool=false)
    T = b.T === nothing ? 125sum(spec.τ) : b.T
    trace = correct_requests(spec, b.form; b.passes, b.runs, T, dtscale=0.01)
    usable = filter(p -> p.design !== nothing, trace)
    isempty(usable) && throw(ArgumentError("the formula has no design: $(last(trace).note)"))
    chosen = lv_model(argmin(p -> p.f, usable).design)
    if validate(chosen, spec; seeds=8_000_001:8_000_032, T=2T, dtscale=0.01).pass
        verbose && println("  Horchler: the correction passes pass the held-out check; no search")
        return chosen
    end
    seed = first(usable).design
    n = length(seed.α)
    lower, upper = [fill(log(0.05), n); fill(log(0.5), n)], [fill(log(200.0), n); fill(log(1e4), n)]
    θ0 = clamp.([log.(seed.α); log.(seed.ν)], lower .+ 1e-6, upper .- 1e-6)
    s = cma_search(θ -> lv_model(raw_design(seed, θ)), spec, θ0, 0.3, lower, upper; b.budget, b.popsize, b.runs, T,
                   dtscale=0.04, maxtime=b.seconds)
    verbose && println("  Horchler: search used $(s.evals) evaluations")
    d, _ = polish_saddles(raw_design(seed, s.θ), spec; passes=2, runs=32, T=4T)
    return lv_model(d)
end
