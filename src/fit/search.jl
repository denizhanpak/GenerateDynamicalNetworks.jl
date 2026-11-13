# CMA-ES on fresh noise (the prototype's Phase 3 search). Every objective call simulates with seeds
# nobody has used, so the search can't overfit one set of noise; candidates are compared on more runs.

"""The search objective: `score` of fresh runs of `build(θ)`; a point `build` rejects costs 100 plus its distance from the start."""
mutable struct Search{F,S}
    build::F
    spec::S
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
    v = score(simulate(m; T=s.T, seeds, sample=s.spec.sample, dtscale=s.dtscale), s.spec).f
    push!(s.log, (; n=s.n, f=v))
    return v
end

"""
    cma_search(build, spec, θ0, σ0, lower, upper; budget, restarts=0, popsize=8, runs=16, T=1000, dtscale=0.04,
               seed=1, base=1_000_000) -> (; θ, candidates, evals, log)

CMA-ES (CMAEvolutionStrategy.jl) on `θ ↦ score(simulate(build(θ)))` within the box `[lower, upper]`,
restarting with twice the population while `budget` evaluations last. Each restart's population mean is
a candidate; with several, the one scoring best on four times the runs, on unused seeds, wins.
"""
function cma_search(build, spec, θ0, σ0, lower, upper; budget, restarts=0, popsize=8, runs=16, T=1000.0, dtscale=0.04,
                    seed=1, base=1_000_000)
    s = Search(build, spec, copy(θ0), runs, float(T), float(dtscale), base, 0, NamedTuple[])
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
        m === nothing ? Inf : score(simulate(m; T, seeds=base+50_000_000:base+50_000_000+4runs-1, sample=spec.sample, dtscale), spec).f
    end
    best = length(candidates) == 1 ? only(candidates) : argmin(pick, candidates)
    return (; θ=best, candidates, evals=s.n, log=s.log)
end

"""
    timed_search(build, spec, θ0, σ0, lower, upper; budget, popsize, runs, T, dtscale=0.04, seed=1,
                 base=1_000_000, maxtime) -> (; θ, evals)

One CMA-ES run that stops after `budget` evaluations or `maxtime` seconds and returns the population
mean it reached. With a finite `maxtime` that binds, the result depends on the machine's speed.
"""
function timed_search(build, spec, θ0, σ0, lower, upper; budget, popsize, runs, T, dtscale=0.04, seed=1, base=1_000_000, maxtime)
    s = Search(build, spec, copy(θ0), runs, float(T), float(dtscale), base, 0, NamedTuple[])
    r = CES.minimize(s, θ0, σ0; lower, upper, popsize, maxfevals=budget, maxtime, seed=UInt(seed), verbosity=0, stagnation=10^9, ftol=1e-30)
    return (; θ=clamp.(CES.population_mean(r), lower, upper), evals=s.n)
end
