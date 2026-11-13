# CMA-ES on the worm from the formula seed, in one search space, then validation on held-out seeds.
# Run from notebooks/ with
#   julia --project=. -t 8 ../benchmarks/phase3/worm_search.jl request|raw budget [T] [tag]
include(joinpath(@__DIR__, "worm.jl"))
using DelimitedFiles

const OUT = joinpath(@__DIR__, "results")
mkpath(OUT)

space = ARGS[1]
budget = parse(Int, ARGS[2])
T = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 500.0
tag = length(ARGS) >= 4 ? ARGS[4] : space

seed = design(WORM_P, WORM_SPEC.τ; form=:u, ε₀=0.1)
if space == "request"
    build = θ -> request_model(WORM_SPEC, θ)
    θ0 = log.(WORM_SPEC.τ)
    σ0, lower, upper, popsize = 0.3, θ0 .- 3, θ0 .+ 3, 8
else
    build = θ -> raw_model(seed, θ)
    θ0 = [log.(seed.α); log.(seed.ν)]
    σ0, lower, upper, popsize = 0.3, [fill(log(0.05), 4); fill(log(0.5), 4)], [fill(log(200.0), 4); fill(log(1e4), 4)], 12
end

t = @elapsed s = search(build, WORM_SPEC, θ0, σ0, lower, upper; budget, popsize, T, runs=16, dtscale=0.04)
@printf("%s: %d evaluations, %.0f s, θ = %s\n", space, s.evals, t, join(round.(s.θ; digits=4), ", "))
m = build(s.θ)
println("α = ", round.(m.α; sigdigits=5), "  ν = ", round.(m.ν; sigdigits=5))
t = @elapsed r = validate(m, WORM_SPEC; T=1000.0)
show_result("$space, held-out", r)
println("   counts ", r.counts, "   validation ", round(t; digits=1), " s")

writedlm(joinpath(OUT, "worm_$(tag)_theta.csv"), s.θ, ',')
writedlm(joinpath(OUT, "worm_$(tag)_trace.csv"), [getfield.(s.log, :n) getfield.(s.log, :f)], ',')
