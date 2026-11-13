# Phase 3 prototype: the optimisation layer on top of the Phase 2 backend. The notebook and the
# run scripts both include this file. It becomes `src/` once the user signs it off.
#
# A spec is a NamedTuple: P (visible graph), τ (target dwell), form, ε₀ (noise), sample (the
# decoder's frame interval).

include(joinpath(@__DIR__, "..", "phase2", "horchler.jl"))
import CMAEvolutionStrategy as CES
using Printf

# ── Decoder: sampled argmax ─────────────────────────────────────────────────────────────────

"""
Bouts read on a frame clock: at each frame the current state is the one with the largest |u|.
A frame won by a hidden state holds the last visible state, which is the dwell `subdivide` designs.
`failed` marks a run that left the bounded region.
"""
mutable struct Frames
    state::Vector{Int}
    t::Vector{Float64}
    failed::Bool
end
Frames() = Frames(Int[], Float64[], false)

function read_frame!(ig)
    q = ig.p
    q.form === :x && (ig.u = max.(ig.u, 0.0))
    ig.iter % q.every == 0 || return nothing
    v, i = findmax(abs, ig.u)
    if !(v < 100)    # also catches NaN
        q.V.failed = true
        terminate!(ig)
    elseif i <= q.visible && (isempty(q.V.state) || last(q.V.state) != i)
        push!(q.V.state, i)
        push!(q.V.t, ig.t)
    end
    return nothing
end
const FRAME = DiscreteCallback((u, t, ig) -> true, read_frame!; save_positions=(false, false))

"""Steps per frame and the step itself: the largest step no coarser than `dtscale/α` that lands on every frame."""
function frame_steps(m, sample, dtscale)
    every = ceil(Int, sample * maximum(m.α) / dtscale)
    return every, sample / every
end

"""Independent runs, one per seed, read by sampled argmax."""
function simulate_frames(m; T, seeds, sample, dtscale=0.01)
    every, dt = frame_steps(m, sample, dtscale)
    base = sde_problem(m; T)
    prob = remake(base; p=(; base.p..., every, V=Frames()))
    ens = EnsembleProblem(prob;
        prob_func=(pr, ctx) -> remake(pr; seed=UInt64(seeds[ctx.sim_id]), p=(; pr.p..., V=Frames())),
        output_func=(sol, ctx) -> ((state=sol.prob.p.V.state, t=sol.prob.p.V.t, failed=sol.prob.p.V.failed), false))
    return solve(ens, EM(), EnsembleThreads(); trajectories=length(seeds), merge(quiet(dt), (; callback=FRAME))...).u
end

# ── Objective ───────────────────────────────────────────────────────────────────────────────

"""
Σ log(mᵢ/τᵢ)² + Σ log(f/p)² over two-way-or-wider branches + penalties. Off-graph transitions cost
`off_weight` per unit above `off_free`, so a run inside the allowance isn't pulled away from its dwells.
"""
function score(runs, spec; off_free=0.005, off_weight=5.0)
    n = size(spec.P, 1)
    any(r.failed for r in runs) && return (; f=1000.0, dwell=fill(NaN, n), off=1.0, branch_term=NaN, dwell_term=NaN)
    meas = measure(runs, n)
    dwell_term = sum(isfinite(meas.dwell[i]) ? min(log(meas.dwell[i] / spec.τ[i])^2, 25.0) : 35.0 for i in 1:n)
    branch_term = 0.0
    for i in 1:n
        s = findall(>(0), spec.P[i, :])
        length(s) >= 2 || continue
        total = max(sum(meas.counts[i, :]), 1)
        branch_term += sum(log(max(meas.counts[i, j] / total, 1e-3) / spec.P[i, j])^2 for j in s)
    end
    off = off_graph(meas, spec.P)
    return (; f=dwell_term + branch_term + off_weight * max(off - off_free, 0.0), dwell=meas.dwell, off, branch_term, dwell_term)
end

# ── Two ways to turn a search point into a model ────────────────────────────────────────────

"""θ = log of the dwell to request; the design formula does the rest."""
request_model(spec, θ) = design(spec.P, exp.(θ); form=spec.form, ε₀=spec.ε₀)

"""θ = [log α; log ν], one of each per state of the seed model, hidden states included."""
function raw_model(seed, θ)
    n = length(seed.α)
    α, ν = exp.(θ[1:n]), exp.(θ[n+1:2n])
    return merge(seed, (; α, ν, ρ=graph_rho(seed.P, α, seed.β, ν)))
end

# ── Correction passes ───────────────────────────────────────────────────────────────────────

"""
Multiplicative correction of the dwell request: request ← request·(τ/measured). Each pass uses
new seeds. Stops early if the formula has no design for the request.
"""
function correct(spec; passes=6, runs=16, T=1000.0, dtscale=0.01, base=100_000, start=spec.τ)
    n = size(spec.P, 1)
    request = float.(start)
    trace = []
    for pass in 1:passes
        m = try
            design(spec.P, request; form=spec.form, ε₀=spec.ε₀)
        catch err
            err isa ArgumentError || rethrow()
            push!(trace, (; pass, request=copy(request), dwell=fill(NaN, n), off=NaN, f=Inf, model=nothing, note=sprint(showerror, err)))
            break
        end
        seeds = base + 1000pass + 1 : base + 1000pass + runs
        s = score(simulate_frames(m; T, seeds, spec.sample, dtscale), spec)
        push!(trace, (; pass, request=copy(request), dwell=s.dwell, off=s.off, f=s.f, model=m, note=""))
        all(isfinite, s.dwell) || break
        request .*= spec.τ ./ s.dwell
    end
    return trace
end

"""
Correction that needs no formula, for models the formula can't reach: νᵢ ← νᵢ·(τᵢ/measuredᵢ) on the
visible states. Returns the model after the last update and one row per pass.
"""
function correct_rates(m, spec; passes=3, runs=64, T=4000.0, dtscale=0.01, base=200_000)
    n = size(spec.P, 1)
    trace = []
    for pass in 1:passes
        seeds = base + 10_000pass + 1 : base + 10_000pass + runs
        s = score(simulate_frames(m; T, seeds, spec.sample, dtscale), spec)
        push!(trace, (; pass, ν=copy(m.ν), dwell=s.dwell, off=s.off, f=s.f))
        ν = copy(m.ν)
        ν[1:n] .*= spec.τ ./ s.dwell
        m = merge(m, (; ν, ρ=graph_rho(m.P, m.α, m.β, ν)))
    end
    return m, trace
end

# ── CMA-ES on fresh noise ───────────────────────────────────────────────────────────────────

"""Objective wrapper: every call gets seeds nobody has used; infeasible points cost 100 plus their distance from the start."""
mutable struct Search{F}
    build::F
    spec::NamedTuple
    start::Vector{Float64}
    runs::Int
    T::Float64
    dtscale::Float64
    next_seed::Int
    n::Int
    log::Vector{NamedTuple}
end

function (s::Search)(θ)
    seeds = s.next_seed:s.next_seed+s.runs-1
    s.next_seed += s.runs
    s.n += 1
    m = try
        s.build(collect(θ))
    catch err
        err isa ArgumentError || rethrow()
        v = 100.0 + sum(abs2, θ .- s.start)
        push!(s.log, (; n=s.n, f=v))
        return v
    end
    v = score(simulate_frames(m; T=s.T, seeds, s.spec.sample, s.dtscale), s.spec).f
    push!(s.log, (; n=s.n, f=v))
    return v
end

"""
CMA-ES from `θ0`, restarting with twice the population while budget lasts. Each restart's population
mean is a candidate; candidates are compared on unused seeds with four times the usual runs.
"""
function search(build, spec, θ0, σ0, lower, upper; budget, restarts=0, popsize=8, runs=16, T=1000.0, dtscale=0.04,
                seed=1, base=1_000_000)
    s = Search(build, spec, copy(θ0), runs, T, dtscale, base, 0, NamedTuple[])
    candidates = Vector{Float64}[]
    λ, rng = popsize, Xoshiro(seed)
    for attempt in 0:restarts
        left = budget - s.n
        left < 2λ && break
        start = attempt == 0 ? θ0 : clamp.(θ0 .+ σ0 .* randn(rng, length(θ0)), lower, upper)
        share = attempt == restarts ? left : left ÷ 2
        r = CES.minimize(s, start, σ0; lower, upper, popsize=λ, maxfevals=share, seed=rand(rng, UInt),
                         verbosity=0, stagnation=10^9, ftol=1e-30)
        push!(candidates, clamp.(CES.population_mean(r), lower, upper))
        λ *= 2
    end
    pick(θ) = let
        m = try build(θ) catch err; err isa ArgumentError ? nothing : rethrow() end
        m === nothing ? Inf : score(simulate_frames(m; T, seeds=base+50_000_000:base+50_000_000+4runs-1, spec.sample, dtscale), spec).f
    end
    best = length(candidates) == 1 ? only(candidates) : argmin(pick, candidates)
    return (; θ=best, candidates, evals=s.n, log=s.log)
end

# ── Validation and comparison ───────────────────────────────────────────────────────────────

"""Held-out seeds, the acceptance step, the acceptance rules from the plan."""
function validate(m, spec; seeds=9_000_001:9_000_064, T=2000.0, dtscale=0.01, tol=0.05)
    n = size(spec.P, 1)
    runs = simulate_frames(m; T, seeds, spec.sample, dtscale)
    failed = count(r -> r.failed, runs)
    meas = measure(runs, n)
    v = verdict(meas, spec.P, spec.τ; tol)
    return (; dwell=meas.dwell, sem=meas.sem, err=v.err, rms=sqrt(mean(log.(meas.dwell ./ spec.τ) .^ 2)), off=v.off,
              branch=v.branch, counts=meas.counts, failed, pass=v.pass && failed == 0,
              dwell_ok=all(abs.(v.err) .<= tol), branch_ok=all(last.(v.branch)))
end

"""Every model on the same held-out seeds; returns one named result per model."""
compare(models::Pair...; spec, kw...) = [name => validate(m, spec; kw...) for (name, m) in models]

"""
The generic pipeline: formula seed, correction passes, and CMA-ES on raw rates only if those don't
pass on seeds the search never sees. `stage` says which step produced the model.
"""
function fit(spec; passes=4, runs=16, T=125sum(spec.τ), budget=500, popsize=12, tol=0.05, polish=true)
    trace = correct(spec; passes, runs, T, dtscale=0.01)
    usable = filter(p -> p.model !== nothing, trace)
    isempty(usable) && throw(ArgumentError("the formula has no design: $(last(trace).note)"))
    chosen = argmin(p -> p.f, usable).model
    gate = (; seeds=8_000_001:8_000_032, T=2T, dtscale=0.01, tol)
    validate(chosen, spec; gate...).pass && return (; model=chosen, stage=:correction, evals=0, trace)
    seed = first(usable).model
    n = length(seed.α)
    lower, upper = [fill(log(0.05), n); fill(log(0.5), n)], [fill(log(200.0), n); fill(log(1e4), n)]
    θ0 = clamp.([log.(seed.α); log.(seed.ν)], lower .+ 1e-6, upper .- 1e-6)
    s = search(θ -> raw_model(seed, θ), spec, θ0, 0.3, lower, upper; budget, popsize, runs, T, dtscale=0.04)
    model = raw_model(seed, s.θ)
    polish && ((model, _) = correct_rates(model, spec; passes=2, runs=32, T=4T))
    return (; model, stage=:search, evals=s.evals, trace)
end

function show_result(name, r)
    @printf("%-30s dwell %s  err%% %s  rms %5.2f%%  off %5.2f%%  dwell %s branch %s %s\n", name,
        join((@sprintf("%.3f", d) for d in r.dwell), " "), join((@sprintf("%+.1f", 100e) for e in r.err), " "), 100r.rms, 100r.off,
        r.dwell_ok ? "ok" : "NO", r.branch_ok ? "ok" : "NO", r.pass ? "PASS" : "fail")
end
