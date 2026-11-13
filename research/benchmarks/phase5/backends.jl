# Phase 5 prototype: the two-population network (Ashwin & Postlethwaite 2016, appendix A) and a CTRNN
# backend (seeded partly from their 2021 construction), on the Phase 3 and 4 pipeline.
#
# Choices the user agreed on 2026-11-08:
#   TwoPopulation  one global speed, a signed ν per edge, one shared noise, no hidden states.
#   CTRNN          one cell per state, argmax decoder, all weights, biases and time constants fitted,
#                  searched from three regimes (2021 weights, ring oscillator, near-fold).
# The CTRNN also has one global gain: the 2021 weights only make sense at gain 1/ε = 20.
include(joinpath(@__DIR__, "..", "phase4", "backends.jl"))
using DelimitedFiles

"""CMA-ES with a time limit, which returns the population mean it had reached when time ran out."""
function timed_search(build, spec, θ0, σ0, lower, upper; budget, popsize, runs, T, dtscale=0.04, seed=1, base=1_000_000, maxtime)
    s = Search(build, spec, copy(θ0), runs, T, dtscale, base, 0, NamedTuple[])
    r = CES.minimize(s, θ0, σ0; lower, upper, popsize, maxfevals=budget, maxtime, seed=UInt(seed), verbosity=0, stagnation=10^9, ftol=1e-30)
    return (; θ=clamp.(CES.population_mean(r), lower, upper), evals=s.n)
end

# ── TwoPopulation ───────────────────────────────────────────────────────────────────────────

const TWO = (A=0.5, C=2.0, D=10.0, E=4.0, F=2.0, ν0=0.01, contract=1.5, step=2.5)

struct TwoPopModel{N}
    P::Matrix{Float64}
    visible::Int
    α::Vector{Float64}          # the step-size rule reads this: the fastest rate is about 10 s
    s::Float64
    B::NTuple{N,Float64}
    from::NTuple{N,Int}
    to::NTuple{N,Int}
    ε::Float64
end

function two_model(spec, edges, s, ν)
    N = length(edges)
    return TwoPopModel{N}(float(spec.P), size(spec.P, 1), [TWO.step * s], s, ntuple(k -> 1.5 - ν[k], N),
                          ntuple(k -> edges[k][1], N), ntuple(k -> edges[k][2], N), float(spec.ε₀))
end
"""θ = [log s; log(−ν)], the heteroclinic side, for the correction passes."""
two_log(spec, edges, θ) = two_model(spec, edges, exp(θ[1]), -exp.(θ[2:end]))
"""θ = [log s; asinh(ν/ν₀)], either side of the bifurcation, for the search."""
two_signed(spec, edges, θ) = two_model(spec, edges, exp(θ[1]), TWO.ν0 .* sinh.(θ[2:end]))

drift_two(z, q, t) = drift_two(z, q, q.size)
function drift_two(z::SVector{M}, q, ::Val{nv}) where {M,nv}
    ne = M - nv
    p2, y2 = sum(j -> z[j]^2, 1:nv), sum(k -> z[nv+k]^2, 1:ne)
    p4 = p2^2
    dp = ntuple(Val(nv)) do j
        out = sum(k -> q.from[k] == j ? z[nv+k]^2 * z[j] * z[q.to[k]] : 0.0, 1:ne)
        inn = sum(k -> q.to[k] == j ? z[nv+k]^2 * z[q.from[k]]^2 : 0.0, 1:ne)
        q.s * (z[j] * (TWO.F * (1 - p2) + TWO.D * (z[j]^2 * p2 - p4)) + TWO.E * (inn - out))
    end
    dy = ntuple(Val(ne)) do k
        y = z[nv+k]
        λ = TWO.A - q.B[k] * z[q.from[k]]^2 + TWO.C * (y2 - y^2)
        -q.s * y * ((y^2 - 1)^2 + λ)
    end
    return SVector{M}(dp..., dy...)
end
noise_flat(z, q, t) = q.ε * ones(typeof(z))

function read_two!(ig)
    q = ig.p
    ig.iter % q.every == 0 || return nothing
    z = ig.u
    if !(maximum(abs, z) < 100)
        q.V.failed = true
        terminate!(ig)
        return nothing
    end
    _, k = findmax(i -> abs(z[i]), 1:q.nv)
    if isempty(q.V.state) || last(q.V.state) != k
        push!(q.V.state, k)
        push!(q.V.t, ig.t)
    end
    return nothing
end
const TWO_FRAME = DiscreteCallback((u, t, ig) -> true, read_two!; save_positions=(false, false))

two_params(m::TwoPopModel) = (; nv=m.visible, size=Val(m.visible), m.s, m.B, m.from, m.to, m.ε)
two_start(m::TwoPopModel{N}) where {N} = SVector{m.visible + N}(ntuple(i -> i == 1 ? 1.0 : m.ε, m.visible + N))

function simulate_frames(m::TwoPopModel; T, seeds, sample, dtscale=0.01)
    every, dt = frame_steps(m, sample, dtscale)
    prob = SDEProblem(drift_two, noise_flat, two_start(m), (0.0, T), merge(two_params(m), (; every, V=Frames())))
    ens = EnsembleProblem(prob;
        prob_func=(pr, ctx) -> remake(pr; seed=UInt64(seeds[ctx.sim_id]), p=(; pr.p..., V=Frames())),
        output_func=(sol, ctx) -> ((state=sol.prob.p.V.state, t=sol.prob.p.V.t, failed=sol.prob.p.V.failed), false))
    return solve(ens, EM(), EnsembleThreads(); trajectories=length(seeds), merge(quiet(dt), (; callback=TWO_FRAME))...).u
end

function fit_two(spec; passes=4, runs=16, T=125sum(spec.τ), budget=500, popsize=12, seconds=120.0)
    edges = edge_list(spec.P)
    ne = length(edges)
    r = seed_rates(spec, edges)
    logθ0 = [log(r.α / TWO.contract); r.logs .+ log(TWO.contract)]
    llo, lhi = [log(0.05); fill(log(1e-4), ne)], [log(200.0); fill(log(5.0), ne)]
    blog = θ -> two_log(spec, edges, θ)
    trace, _ = correct_edges(blog, clamp.(logθ0, llo, lhi), spec, edges, llo, lhi; passes, runs, T)
    θc = argmin(p -> p.f, trace).θ
    to_signed(θ) = [θ[1]; asinh.(-exp.(θ[2:end]) ./ TWO.ν0)]
    done(stage, θs, evals) = (; model=two_signed(spec, edges, θs), θ=θs, stage, evals, seeded=r.seeded, dim=length(θs))
    validate(blog(θc), spec; seeds=8_000_001:8_000_032, T=2T, dtscale=0.01).pass && return done(:correction, to_signed(θc), 0)
    slo = [log(0.05); fill(asinh(-5.0 / TWO.ν0), ne)]
    shi = [log(200.0); fill(asinh(0.3 / TWO.ν0), ne)]
    s = timed_search(θ -> two_signed(spec, edges, θ), spec, clamp.(to_signed(θc), slo .+ 1e-6, shi .- 1e-6), 0.3, slo, shi; budget, popsize, runs, T, maxtime=seconds)
    if all(<(0), s.θ[2:end])    # still all heteroclinic, so the log-rate polish applies
        back = [s.θ[1]; log.(-TWO.ν0 .* sinh.(s.θ[2:end]))]
        _, θp = correct_edges(blog, clamp.(back, llo, lhi), spec, edges, llo, lhi; passes=2, runs=32, T=4T, base=200_000)
        return done(:search, to_signed(θp), s.evals)
    end
    return done(:search, s.θ, s.evals)
end

# ── CTRNN ───────────────────────────────────────────────────────────────────────────────────

const CTRNN_RATE_MAX = 200.0

struct CTRNNModel
    P::Matrix{Float64}
    visible::Int
    α::Vector{Float64}          # stiffest rate, for the step-size rule
    W::Matrix{Float64}
    b::Vector{Float64}
    τ::Vector{Float64}
    g::Float64
    ε::Float64
end

"""
θ = [vec(W); b; log τ; log g] over all N cells, so length(θ) = (N + 1)². The first size(P, 1) cells
are the states; any after them are hidden. A model stiffer than `cap` counts as infeasible.
"""
function ctrnn_model(spec, θ; cap=CTRNN_RATE_MAX)
    n = isqrt(length(θ)) - 1
    (n + 1)^2 == length(θ) || throw(DimensionMismatch("θ has length $(length(θ)), not (N + 1)²"))
    W, b, τ, g = reshape(θ[1:n^2], n, n), θ[n^2+1:n^2+n], exp.(θ[n^2+n+1:n^2+2n]), exp(θ[end])
    rate = max(1.0, g * maximum(abs, W) / 4) / minimum(τ)
    rate > cap && throw(ArgumentError("rate $rate is above the limit"))
    return CTRNNModel(float(spec.P), size(spec.P, 1), [rate], W, b, τ, g, float(spec.ε₀))
end

"""Start of the search in one regime, with the time constants scaled to the targets."""
function ctrnn_seed(spec, regime)
    P, τ, n = spec.P, spec.τ, size(spec.P, 1)
    if regime === :excitable        # the 2021 four-weight construction, δ = 0.4
        W = [i == j ? 1.0 : P[j, i] > 0 ? 0.3 : P[i, j] > 0 ? -0.7 : 0.0 for i in 1:n, j in 1:n]
        b, g, unit = fill(-0.5, n), 20.0, 7.0
    else                            # the ring of the user's SimpleModelCTRNN, on and off the fold
        p, m, unit = regime === :oscillator ? (-5.0, -14.0, 3.14) : (-6.9, -15.0, 5.0)
        W = [i == j ? 1.0 : P[j, i] > 0 ? p : m for i in 1:n, j in 1:n]
        b, g = zeros(n), 1.0
    end
    return [vec(W); b; log.(τ ./ unit); log(g)]
end

drift_ctrnn(y, q, t) = (-y .+ q.W * (1 ./ (1 .+ exp.(-q.g .* (y .+ q.b))))) ./ q.τ

function read_ctrnn!(ig)
    q = ig.p
    ig.iter % q.every == 0 || return nothing
    y = ig.u
    if !(maximum(abs, y) < 1000)
        q.V.failed = true
        terminate!(ig)
        return nothing
    end
    k = argmax(i -> q.g * (y[i] + q.b[i]), 1:q.nv)    # state cells only; hidden cells are never read
    if isempty(q.V.state) || last(q.V.state) != k
        push!(q.V.state, k)
        push!(q.V.t, ig.t)
    end
    return nothing
end
const CTRNN_FRAME = DiscreteCallback((u, t, ig) -> true, read_ctrnn!; save_positions=(false, false))

ctrnn_params(m::CTRNNModel) = let n = size(m.W, 1)
    (; W=SMatrix{n,n}(m.W), b=SVector{n}(m.b), τ=SVector{n}(m.τ), m.g, m.ε, nv=m.visible)
end
ctrnn_start(m::CTRNNModel) = let n = size(m.W, 1)
    SVector{n}(ntuple(i -> -m.b[i] + (i == 1 ? 4.0 : -4.0) / m.g, n))
end

function simulate_frames(m::CTRNNModel; T, seeds, sample, dtscale=0.01)
    every, dt = frame_steps(m, sample, dtscale)
    prob = SDEProblem(drift_ctrnn, noise_flat, ctrnn_start(m), (0.0, T), merge(ctrnn_params(m), (; every, V=Frames())))
    ens = EnsembleProblem(prob;
        prob_func=(pr, ctx) -> remake(pr; seed=UInt64(seeds[ctx.sim_id]), p=(; pr.p..., V=Frames())),
        output_func=(sol, ctx) -> ((state=sol.prob.p.V.state, t=sol.prob.p.V.t, failed=sol.prob.p.V.failed), false))
    return solve(ens, EM(), EnsembleThreads(); trajectories=length(seeds), merge(quiet(dt), (; callback=CTRNN_FRAME))...).u
end

"""Time-constant correction, τᵢ ← τᵢ·(target/measured). Stops if a state goes unvisited."""
function ctrnn_correct(spec, θ0; passes, runs, T, base=100_000, lower, upper)
    n = size(spec.P, 1)
    idx = n^2+n+1:n^2+2n
    θ = copy(θ0)
    trace = []
    for pass in 1:passes
        m = try ctrnn_model(spec, θ) catch err; err isa ArgumentError ? nothing : rethrow() end
        m === nothing && break
        r = simulate_frames(m; T, seeds=base+1000pass+1:base+1000pass+runs, spec.sample, dtscale=0.01)
        s = score(r, spec)
        push!(trace, (; θ=copy(θ), s.f))
        all(isfinite, s.dwell) || break
        θ[idx] .+= log.(spec.τ ./ s.dwell)
        θ .= clamp.(θ, lower, upper)
    end
    return trace, θ
end

function fit_ctrnn(spec, regime; passes=3, runs=16, T=125sum(spec.τ), budget=500, popsize=14, seconds=280.0)
    n = size(spec.P, 1)
    θ0 = ctrnn_seed(spec, regime)
    lower = [fill(-40.0, n^2); fill(-30.0, n); fill(log(1e-3), n); log(0.5)]
    upper = [fill(40.0, n^2); fill(30.0, n); fill(log(30.0), n); log(40.0)]
    trace, _ = ctrnn_correct(spec, clamp.(θ0, lower, upper); passes, runs, T, lower, upper)
    θc = isempty(trace) ? clamp.(θ0, lower, upper) : argmin(p -> p.f, trace).θ
    done(stage, θ, evals) = (; model=ctrnn_model(spec, θ), θ, stage, evals, seeded=!isempty(trace), dim=length(θ))
    validate(ctrnn_model(spec, θc), spec; seeds=8_000_001:8_000_032, T=2T, dtscale=0.01).pass && return done(:correction, θc, 0)
    W = θc[1:n^2]
    scale = [0.1 .* max.(abs.(W), 0.2maximum(abs, W)); 0.1 .* max.(abs.(θc[n^2+1:n^2+n]), 1.0); fill(0.3, n); 0.2]
    build(z) = ctrnn_model(spec, clamp.(θc .+ scale .* z, lower, upper))
    zlo, zhi = (lower .- θc) ./ scale, (upper .- θc) ./ scale
    s = timed_search(build, spec, zeros(length(θc)), 1.0, zlo, zhi; budget, popsize, runs, T, maxtime=seconds)
    θs = clamp.(θc .+ scale .* s.θ, lower, upper)
    _, θp = ctrnn_correct(spec, θs; passes=2, runs=32, T=4T, base=200_000, lower, upper)
    final = try ctrnn_model(spec, θp); θp catch err; err isa ArgumentError ? θs : rethrow() end
    return done(:search, final, s.evals)
end
