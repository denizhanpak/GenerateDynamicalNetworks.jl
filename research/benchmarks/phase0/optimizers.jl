# Which optimiser? Same evaluation budget for every method, scored on held-out seeds at a finer dt.
include("common.jl")
import CMAEvolutionStrategy as CES
import Evolutionary as EV
import Metaheuristics as MH
import BlackBoxOptim as BBO
using DelimitedFiles

const SCALE = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1.0
const REPS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 6
const RING = SMatrix{4,4}([mod1(i + 1, 4) == j ? 1.0 : 0.0 for i in 1:4, j in 1:4])

"""Phase 1's measurement: the average of per-run means, plus penalties for missing or off-graph."""
function score(tallies, τ)
    n = length(τ)
    moves = sum(t.moves for t in tallies)
    total = sum(moves)
    off = total == 0 ? 1.0 : sum(moves[i, j] for i in 1:n, j in 1:n if RING[i, j] == 0; init=0) / total
    m = zeros(n)
    f = 0.0
    for i in 1:n
        means = [t.sum[i] / t.count[i] for t in tallies if t.count[i] > 0]
        if isempty(means)
            f += 25.0 + 10.0
            m[i] = NaN
        else
            m[i] = mean(means)
            f += min(log(m[i] / τ[i])^2, 25.0)
        end
    end
    return (; f=f + 5off, dwell=m, off)
end

# ── Problem A: refit the worm ring, 5 dimensions, from the design formula ────────────────────
const A = let m = ring_model(WORM.τ; ε=1e-3)
    (; name="lv_ring_5d", center=[log(m.α[1]); log.(m.ν)], half=fill(2.0, 5), budget=round(Int, 1500SCALE), ε=1e-3,
        runs=12, T=60.0, c=0.04, startable=true)
end
function a_model(u)
    θ = A.center .+ A.half .* u
    α, ν = exp(θ[1]), max.(exp.(θ[2:5]), 1.2)
    a = fill(α, 4)
    ρ = createcycle(a, ones(4), ν)
    return (; α=SVector{4}(a), ρ=SMatrix{4,4}(ρ), ε=SVector{4}(fill(A.ε, 4)))
end
function a_run(u, seeds; c=A.c, T=A.T)
    p = a_model(u)
    dt = c / p.α[1]
    x0 = SVector{4}([0.9, A.ε, A.ε, A.ε])
    return [em_tally(lv_drift, identity_readout, x0, p, p.ε, dt, round(Int, T / dt); seed=s) for s in seeds]
end
a_eval(u, seeds) = score(a_run(u, seeds), WORM.τ)
a_holdout(u) = score(on_threads(s -> only(a_run(u, [s]; c=0.01, T=200.0)), 1_000_001:1_000_048, 16), WORM.τ)

# ── Problem B: a 4-neuron CTRNN, 24 dimensions, from nothing ─────────────────────────────────
const B = (; name="ctrnn_24d", center=zeros(24), half=ones(24), budget=round(Int, 20000SCALE), runs=8, T=40.0, dt=0.005,
    noise=0.3, startable=false)
logistic(v) = 1 / (1 + exp(-v))
ct_drift(y, p) = (-y + p.W * logistic.(y + p.θ)) ./ p.τ
ct_read(y, p) = logistic.(y + p.θ)
function b_model(u)
    W = SMatrix{4,4}(16 .* u[1:16])
    θ = SVector{4}(16 .* u[17:20])
    τ = SVector{4}(exp.(log(0.05) .+ (u[21:24] .+ 1) ./ 2 .* (log(5.0) - log(0.05))))
    return (; W, θ, τ)
end
function b_run(u, seeds; dt=B.dt, T=B.T)
    p = b_model(u)
    σ = SVector{4}(fill(B.noise, 4))
    y0 = -p.θ
    return [em_tally(ct_drift, ct_read, y0, p, σ, dt, round(Int, T / dt); seed=s, lower=-Inf, stride=4) for s in seeds]
end
b_eval(u, seeds) = score(b_run(u, seeds), WORM.τ)
b_holdout(u) = score(on_threads(s -> only(b_run(u, [s]; dt=B.dt / 2, T=120.0)), 1_000_001:1_000_048, 16), WORM.τ)

# ── Bookkeeping ──────────────────────────────────────────────────────────────────────────────
mutable struct Counter{F}
    f::F
    limit::Int
    n::Int
    best::Float64
    bestx::Vector{Float64}
    trace::Vector{Tuple{Int,Float64}}
    fresh::Bool
    runs::Int
    base::Int
end
function (c::Counter)(u)
    c.n >= c.limit && return c.best + 1.0    # over budget: nothing further can count
    x = clamp.(collect(u), -1, 1)
    seeds = c.fresh ? (c.base + c.n * c.runs + 1:c.base + (c.n + 1) * c.runs) : (c.base + 1:c.base + c.runs)
    v = c.f(x, seeds).f
    c.n += 1
    if v < c.best
        c.best, c.bestx = v, x
        push!(c.trace, (c.n, v))
    end
    return v
end

function run_method(name, P, ev, rep)
    d = length(P.center)
    seed = 7919 * rep + hash(name) % 1000
    Random.seed!(seed)        # BlackBoxOptim ignores RngSeed and draws from the task's RNG
    rng = Xoshiro(seed)
    fresh = endswith(name, "fresh")
    c = Counter(ev, P.budget, 0, Inf, zeros(d), Tuple{Int,Float64}[], fresh, P.runs, 10_000 * rep)
    lo, hi = fill(-1.0, d), fill(1.0, d)
    start = startswith(name, "seeded") ? zeros(d) : rand(rng, d) .* 2 .- 1
    σ = startswith(name, "seeded") ? 0.1 : 0.4
    final = nothing
    if name == "random CES restarts" || name == "random CES restarts fresh"
        # Restarts from random points, doubling the population each time (IPOP).
        popsize, candidates = 4 + floor(Int, 3log(d)), Vector{Vector{Float64}}()
        while P.budget - c.n > 2popsize
            s0 = rand(rng, d) .* 2 .- 1
            r = CES.minimize(c, s0, σ; lower=lo, upper=hi, popsize, maxfevals=P.budget - c.n, seed=rand(rng, UInt),
                verbosity=0, ftol=1e-30)
            push!(candidates, clamp.(CES.population_mean(r), -1, 1))
            popsize *= 2
        end
        if fresh
            # Means are compared on seeds no run has used, with four times the usual runs.
            pick(x) = ev(x, 900_000 * rep .+ (1:4P.runs)).f
            final = argmin(pick, candidates)
        else
            final = c.bestx
        end
    elseif name == "random DE then CES fresh"
        # Global search on fixed seeds, then polish the best point on fresh noise.
        BBO.bboptimize(c; SearchRange=[(-1.0, 1.0) for _ in 1:d], NumDimensions=d,
            Method=:adaptive_de_rand_1_bin_radiuslimited, MaxFuncEvals=round(Int, 0.7P.budget), TraceMode=:silent)
        c.fresh = true
        r = CES.minimize(c, copy(c.bestx), 0.05; lower=lo, upper=hi, maxfevals=P.budget - c.n, seed=seed, verbosity=0,
            stagnation=10^9, ftol=1e-30)
        final = clamp.(CES.population_mean(r), -1, 1)
    elseif occursin("CES", name)
        nh = occursin("noise", name) ? CES.NoiseHandling(d) : nothing
        r = CES.minimize(c, start, σ; lower=lo, upper=hi, noise_handling=nh, maxfevals=P.budget, seed=seed,
            verbosity=0, stagnation=10^9, ftol=1e-30)
        final = fresh ? clamp.(CES.population_mean(r), -1, 1) : c.bestx
    elseif occursin("EvoCMA", name)
        λ = d <= 8 ? 8 : 12
        EV.optimize(c, EV.BoxConstraints(lo, hi), start, EV.CMAES(μ=λ ÷ 2, λ=λ, sigma0=σ),
            EV.Options(iterations=P.budget ÷ λ, rng=Xoshiro(seed), abstol=1e-12, reltol=0.0, successive_f_tol=10^7))
        final = c.bestx
    elseif occursin("BBO", name)
        method = occursin("xNES", name) ? :xnes : :adaptive_de_rand_1_bin_radiuslimited
        BBO.bboptimize(c; SearchRange=[(-1.0, 1.0) for _ in 1:d], NumDimensions=d, Method=method,
            MaxFuncEvals=P.budget, TraceMode=:silent)
        final = c.bestx
    elseif occursin("SHADE", name)
        o = MH.Options(f_calls_limit=P.budget, seed=seed, iterations=10^7, time_limit=Inf, verbose=false, f_tol=0.0, x_tol=0.0)
        MH.optimize(c, MH.boxconstraints(lb=lo, ub=hi), MH.SHADE(; options=o))
        final = c.bestx
    end
    return (; name, rep, evals=c.n, best=c.best, final, trace=c.trace)
end

methods_a = ["seeded CES", "seeded CES noise", "seeded CES fresh", "seeded CES noise fresh", "seeded EvoCMA", "random CES", "random BBO DE", "random BBO xNES", "random SHADE"]
methods_b = ["random CES", "random CES noise", "random EvoCMA", "random BBO DE", "random BBO xNES", "random SHADE"]

function campaign(P, ev, holdout, methods; tag="")
    jobs = [(m, r) for m in methods for r in 1:REPS]
    results = Vector{Any}(undef, length(jobs))
    next = Threads.Atomic{Int}(1)
    @sync for _ in 1:15
        Threads.@spawn while true
            j = Threads.atomic_add!(next, 1)
            j > length(jobs) && break
            m, r = jobs[j]
            try
                t = @elapsed v = run_method(m, P, ev, r)
                results[j] = merge(v, (; seconds=t))
            catch e
                results[j] = (; name=m, rep=r, error=first(replace(sprint(showerror, e), "\n" => " "), 240))
            end
        end
    end
    rows = []
    open("$(P.name)$(tag)_trace.csv", "w") do tr
        println(tr, "method,rep,evals,best")
        for r in results
            if haskey(r, :error)
                println("ERROR ", r.name, " rep ", r.rep, " ", r.error)
                continue
            end
            for (n, b) in r.trace; println(tr, "$(r.name),$(r.rep),$n,$b"); end
            h = holdout(r.final)
            worst = maximum(abs.(h.dwell ./ WORM.τ .- 1))
            push!(rows, (r.name, r.rep, r.evals, r.best, h.f, worst, h.off, r.seconds))
        end
    end
    open("$(P.name)$(tag)_runs.csv", "w") do io
        println(io, "method,rep,evals,search_f,heldout_f,worst_dwell_error,off_graph,seconds")
        for r in rows; println(io, join(r, ",")); end
    end
    println("\n", P.name, ", budget ", P.budget, ", ", REPS, " repeats")
    @printf("%-24s %6s %10s %10s %10s %8s %8s\n", "method", "evals", "search f", "heldout f", "worst err", "success", "off")
    for m in methods
        s = [r for r in rows if r[1] == m]
        isempty(s) && continue
        ok = count(r -> isfinite(r[6]) && r[6] <= 0.05 && r[7] <= 0.01, s)
        @printf("%-24s %6d %10.4g %10.4g %9.1f%% %5d/%d %7.1f%%\n", m, median(getindex.(s, 3)), median(getindex.(s, 4)),
            median(getindex.(s, 5)), 100median(replace(getindex.(s, 6), NaN => Inf)), ok, length(s), 100median(getindex.(s, 7)))
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
h0 = a_holdout(zeros(5))
println("Design formula, before any search: dwell error ", round.(100 .* (h0.dwell ./ WORM.τ .- 1); digits=1), "% f=", round(h0.f; digits=4))
@time campaign(A, a_eval, a_holdout, methods_a)
flush(stdout)
@time campaign(B, b_eval, b_holdout, methods_b)

# What does each library itself cost per evaluation? Noise keeps them from stopping on convergence.
println("\nLibrary overhead, noisy 5-D sphere, about 20000 evaluations")
const CALLS = Ref(0)
sphere(x) = (CALLS[] += 1; sum(abs2, x) + 0.01randn())
lo, hi = fill(-1.0, 5), fill(1.0, 5)
overhead = [
    ("CMAEvolutionStrategy", () -> CES.minimize(sphere, fill(0.5, 5), 0.3; lower=lo, upper=hi, maxfevals=20000, verbosity=0, ftol=1e-30, stagnation=10^9, seed=1)),
    ("Evolutionary CMAES", () -> EV.optimize(sphere, EV.BoxConstraints(lo, hi), fill(0.5, 5), EV.CMAES(μ=4, λ=8, sigma0=0.3), EV.Options(iterations=2500, abstol=1e-30, reltol=0.0, successive_f_tol=10^7))),
    ("BlackBoxOptim DE", () -> BBO.bboptimize(sphere; SearchRange=[(-1.0, 1.0) for _ in 1:5], NumDimensions=5, MaxFuncEvals=20000, TraceMode=:silent)),
    ("Metaheuristics SHADE", () -> MH.optimize(sphere, MH.boxconstraints(lb=lo, ub=hi), MH.SHADE(; options=MH.Options(f_calls_limit=20000, f_tol=0.0, x_tol=0.0, iterations=10^7)))),
]
open("overhead.csv", "w") do io
    println(io, "library,seconds,calls")
    for (n, f) in overhead
        f()
        CALLS[] = 0
        t = @elapsed f()
        @printf("%-22s %6.2f s  %6d calls  %5.1f µs per evaluation\n", n, t, CALLS[], 1e6t / CALLS[])
        println(io, "$n,$t,$(CALLS[])")
    end
end

# Does generation-level threading actually help?
println("\nThreading inside a generation, Problem A, 16 per generation, 96 evaluations")
for mt in (false, true)
    t = @elapsed CES.minimize(u -> a_eval(clamp.(u, -1, 1), 1:12).f, zeros(5), 0.1; lower=fill(-1.0, 5), upper=fill(1.0, 5),
        popsize=16, maxfevals=96, verbosity=0, multi_threading=mt, seed=3)
    @printf("multi_threading=%-5s %6.2f s\n", mt, t)
end
end
