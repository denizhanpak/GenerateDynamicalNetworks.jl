# One independent repeat of the worm search: new CMA-ES seed and new noise, from the same formula
# seed, then the ν polish, scored on the held-out runs used for every other worm model. Run from
# notebooks/ with
#   julia --project=. -t 8 ../benchmarks/phase3/worm_repeat.jl rep [budget] [αmax]
# rep 1 is the original run in worm_compare.csv, so repeats start at 2.
include(joinpath(@__DIR__, "worm.jl"))
using DelimitedFiles

const OUT = joinpath(@__DIR__, "results")
rep = parse(Int, ARGS[1])
budget = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 800
αmax = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 200.0

seed = design(WORM_P, WORM_SPEC.τ; form=:u, ε₀=0.1)
build = θ -> raw_model(seed, θ)
θ0 = [log.(seed.α); log.(seed.ν)]
lower, upper = [fill(log(0.05), 4); fill(log(0.5), 4)], [fill(log(αmax), 4); fill(log(1e4), 4)]

t = @elapsed begin
    s = search(build, WORM_SPEC, θ0, 0.3, lower, upper; budget, popsize=12, T=500.0, runs=16, dtscale=0.04,
               seed=rep, base=10_000_000rep + 1_000_000)
    found = build(s.θ)
    polished, _ = correct_rates(found, WORM_SPEC; passes=3, base=200_000 + 100_000rep)
end
kw = (; seeds=9_100_001:9_100_064, T=4000.0, dtscale=0.01)
before = validate(found, WORM_SPEC; kw...)
after = validate(polished, WORM_SPEC; kw...)
show_result("repeat $rep, search", before)
show_result("repeat $rep, polished", after)
println("   α = ", round.(polished.α; sigdigits=4), "  ν = ", round.(polished.ν; sigdigits=4), "  ", round(Int, t), " s")

writedlm(joinpath(OUT, "worm_repeat_$(rep)_theta.csv"), [log.(polished.α); log.(polished.ν)], ',')
file = joinpath(OUT, "worm_repeats.csv")
open(file, "a") do io
    filesize(file) == 0 && println(io, "rep,search_rms,search_off,search_pass,rms,off,pass,e1,e2,e3,e4,a1,a2,a3,a4,n1,n2,n3,n4,evals,seconds")
    println(io, join([rep; before.rms; before.off; before.pass; after.rms; after.off; after.pass; after.err; polished.α; polished.ν; s.evals; t], ","))
end
