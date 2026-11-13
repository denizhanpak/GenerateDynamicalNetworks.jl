# Phase 4 prototype: the simplex and cylinder realisations of Ashwin & Postlethwaite (2013) as
# backends of the Phase 3 pipeline. It becomes `src/` once the user signs it off.
#
# Choices the user agreed on 2026-11-08:
#   SimplexLV   one global speed, a rate per edge, a contraction per state; one shared noise, so
#               branches come from the edge rates; no hidden states.
#   CylinderLV  one global speed, a rate per edge, k_α = k_ω = 2 and L_ω = 1.5625 fixed; a frame is
#               vertex k only if |p − k| < 0.25 and every |y| < 0.5, otherwise the last vertex holds.
include(joinpath(@__DIR__, "..", "phase3", "worm.jl"))
include(joinpath(@__DIR__, "..", "phase2", "cases.jl"))

edge_list(P) = [(i, j) for i in axes(P, 1) for j in axes(P, 2) if P[i, j] > 0]
successors(P, i) = findall(>(0), P[i, :])

"""
Shared speed and log edge rate for unit contraction, from the Phase 2 inversion with one noise value.
Branches: the paper's low-noise scaling gives a rate ratio of about p^(1/L). `seeded` is false if the
formula has no design, in which case a rough guess from τ is used and the search has to do the work.
"""
function seed_rates(spec, edges)
    n = size(spec.P, 1)
    L = max(log(sqrt(0.1) / spec.ε₀), 1.0)
    ε_eff = [effective_noise(fill(float(spec.ε₀), length(successors(spec.P, i)))) for i in 1:n]
    α, base, seeded = try
        a, ν = dwell_rates(:u, spec.τ, ε_eff, ones(n))
        a[1], -log.(ν), true
    catch err
        err isa ArgumentError || rethrow()
        s = 10 / minimum(spec.τ)
        s, log.(L ./ (s .* spec.τ)), false
    end
    logs = [base[i] + (log(spec.P[i, j]) - mean(log.(spec.P[i, successors(spec.P, i)]))) / L for (i, j) in edges]
    return (; α, logs, seeded)
end

# ── SimplexLV ───────────────────────────────────────────────────────────────────────────────

"""dxⱼ = s·xⱼ(1 − |x|² + Σᵢ aᵢⱼxᵢ²) + ε dW, written as a u-form model so all of Phase 3 applies."""
function simplex_model(spec, edges, θ)
    n, ne = size(spec.P, 1), length(edges)
    s, e, c = exp(θ[1]), exp.(θ[2:1+ne]), exp.(θ[2+ne:1+ne+n])
    a = [i == j ? 0.0 : -c[i] for i in 1:n, j in 1:n]
    for (k, (i, j)) in enumerate(edges)
        a[i, j] = e[k]
    end
    ρ = [s * (i == j ? 1.0 : 1.0 - a[j, i]) for i in 1:n, j in 1:n]
    return (; form=:u, P=float(spec.P), visible=n, κ=0.1, β=ones(n), α=fill(s, n), ε=fill(float(spec.ε₀), n), ρ)
end

function make_simplex(spec, edges)
    n, ne = size(spec.P, 1), length(edges)
    r = seed_rates(spec, edges)
    θ0 = [log(r.α); r.logs; zeros(n)]
    lower = [log(0.05); fill(log(1e-4), ne); fill(log(0.25), n)]
    upper = [log(200.0); fill(log(4.0), ne); fill(log(4.0), n)]
    return (; θ0, lower, upper, build=θ -> simplex_model(spec, edges, θ), seeded=r.seeded)
end

# ── CylinderLV ──────────────────────────────────────────────────────────────────────────────

const CYL_K, CYL_LOMEGA, CYL_CONTRACT = 2.0, 1.5625, 1.5625 + 9 / 16

struct CylinderModel{N}
    P::Matrix{Float64}
    visible::Int
    α::Vector{Float64}          # one entry, the speed: the step-size rule reads it
    Lα::NTuple{N,Float64}
    from::NTuple{N,Float64}
    to::NTuple{N,Float64}
    ε::Float64
end

"""Eigenvalue at vertex k along edge ℓ (the paper's e_{kℓ}), solved for L_α."""
cyl_Lα(e, i, j) = e + 9 / 16 + CYL_LOMEGA * sech(CYL_K * (i - j))^2

function cylinder_model(spec, edges, θ)
    N = length(edges)
    e = exp.(θ[2:1+N])
    Lα = ntuple(k -> cyl_Lα(e[k], edges[k]...), N)
    return CylinderModel{N}(float(spec.P), size(spec.P, 1), [exp(θ[1])], Lα,
                            ntuple(k -> float(edges[k][1]), N), ntuple(k -> float(edges[k][2]), N), float(spec.ε₀))
end

function make_cylinder(spec, edges)
    ne = length(edges)
    r = seed_rates(spec, edges)
    θ0 = [log(r.α / CYL_CONTRACT); r.logs .+ log(CYL_CONTRACT)]
    lower = [log(0.05); fill(log(1e-3), ne)]
    upper = [log(200.0); fill(log(8.0), ne)]
    return (; θ0, lower, upper, build=θ -> cylinder_model(spec, edges, θ), seeded=r.seeded)
end

function drift_cylinder(z::SVector{M}, m::NamedTuple, t) where {M}
    N = M - 1
    p, y2 = z[M], sum(i -> z[i]^2, 1:N)
    dy = ntuple(Val(N)) do j
        G = (z[j]^2 - 5 / 4)^2 - 1 - m.Lα[j] * sech(CYL_K * (p - m.from[j]))^2 + CYL_LOMEGA * sech(CYL_K * (p - m.to[j]))^2 + y2 - z[j]^2
        -m.s * z[j] * G
    end
    F = sum(j -> z[j]^2 * tanh(m.to[j] - p), 1:N)
    return SVector{M}(dy..., m.s * (-sinpi(2p) + F))
end
noise_cylinder(z, m, t) = m.ε * ones(typeof(z))

function read_cylinder!(ig)
    q = ig.p
    ig.iter % q.every == 0 || return nothing
    z, M = ig.u, length(ig.u)
    if !(maximum(abs, z) < 100)
        q.V.failed = true
        terminate!(ig)
        return nothing
    end
    k = round(Int, z[M])
    if 1 <= k <= q.visible && abs(z[M] - k) < 0.25 && maximum(i -> abs(z[i]), 1:M-1) < 0.5 && (isempty(q.V.state) || last(q.V.state) != k)
        push!(q.V.state, k)
        push!(q.V.t, ig.t)
    end
    return nothing
end
const CYL_FRAME = DiscreteCallback((u, t, ig) -> true, read_cylinder!; save_positions=(false, false))

"""Same contract as the Phase 3 `simulate_frames`, so `score`, `search` and `validate` work unchanged."""
function simulate_frames(m::CylinderModel{N}; T, seeds, sample, dtscale=0.01) where {N}
    every, dt = frame_steps(m, sample, dtscale)
    q = (; s=m.α[1], m.Lα, m.from, m.to, m.ε, m.visible, every, V=Frames())
    z0 = SVector{N + 1}(ntuple(k -> k <= N ? m.ε : 1.0, Val(N + 1)))
    prob = SDEProblem(drift_cylinder, noise_cylinder, z0, (0.0, T), q)
    ens = EnsembleProblem(prob;
        prob_func=(pr, ctx) -> remake(pr; seed=UInt64(seeds[ctx.sim_id]), p=(; pr.p..., V=Frames())),
        output_func=(sol, ctx) -> ((state=sol.prob.p.V.state, t=sol.prob.p.V.t, failed=sol.prob.p.V.failed), false))
    return solve(ens, EM(), EnsembleThreads(); trajectories=length(seeds), merge(quiet(dt), (; callback=CYL_FRAME))...).u
end

# ── Correction and the pipeline ─────────────────────────────────────────────────────────────

"""
Rates are moved by what was measured: every edge out of state i by log(mᵢ/τᵢ), and the edges of a
branch by the branch error over L, the sensitivity from the paper's scaling. Returns one row per
pass and the θ after the last update.
"""
function correct_edges(build, θ0, spec, edges, lower, upper; passes, runs, T, dtscale=0.01, base=100_000)
    n = size(spec.P, 1)
    L = max(log(sqrt(0.1) / spec.ε₀), 1.0)
    θ = copy(θ0)
    trace = []
    for pass in 1:passes
        r = simulate_frames(build(θ); T, seeds=base+1000pass+1:base+1000pass+runs, spec.sample, dtscale)
        s = score(r, spec)
        push!(trace, (; pass, θ=copy(θ), f=s.f, dwell=s.dwell, off=s.off))
        all(isfinite, s.dwell) || break
        C = measure(r, n).counts
        change = zeros(length(edges))
        for (k, (i, j)) in enumerate(edges)
            out = successors(spec.P, i)
            error_of(b) = log(spec.P[i, b]) - log(max(C[i, b], 1) / max(sum(C[i, out]), 1))
            change[k] = log(s.dwell[i] / spec.τ[i]) + (error_of(j) - mean(error_of, out)) / L
        end
        θ[2:1+length(edges)] .+= change
        θ .= clamp.(θ, lower, upper)
    end
    return trace, θ
end

struct OutOfTime <: Exception end

"""Formula seed, correction, held-out gate, then CMA-ES on the raw parameters and a short polish."""
function fit_backend(spec, make; passes=4, runs=16, T=125sum(spec.τ), budget=500, popsize=12, seconds=420.0)
    edges = edge_list(spec.P)
    b = make(spec, edges)
    trace, _ = correct_edges(b.build, clamp.(b.θ0, b.lower, b.upper), spec, edges, b.lower, b.upper; passes, runs, T)
    θc = argmin(p -> p.f, trace).θ
    done(stage, θ, evals) = (; model=b.build(θ), θ, stage, evals, seeded=b.seeded, dim=length(θ))
    validate(b.build(θc), spec; seeds=8_000_001:8_000_032, T=2T, dtscale=0.01).pass && return done(:correction, θc, 0)
    stop = time() + seconds
    capped(θ) = (time() > stop && throw(OutOfTime()); b.build(θ))
    s = try
        search(capped, spec, clamp.(θc, b.lower .+ 1e-6, b.upper .- 1e-6), 0.3, b.lower, b.upper; budget, popsize, runs, T, dtscale=0.04)
    catch err
        err isa OutOfTime || rethrow()
        return done(:out_of_time, θc, 0)
    end
    _, θp = correct_edges(b.build, s.θ, spec, edges, b.lower, b.upper; passes=2, runs=32, T=4T, base=200_000)
    return done(:search, θp, s.evals)
end
