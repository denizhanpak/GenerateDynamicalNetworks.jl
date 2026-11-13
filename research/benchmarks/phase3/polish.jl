# Polish the raw-space CMA-ES result with long-run ν corrections, then compare it with the user's
# fit on the same held-out seeds. Run from notebooks/ with
#   julia --project=. -t 8 ../benchmarks/phase3/polish.jl
include(joinpath(@__DIR__, "worm.jl"))
using DelimitedFiles

const OUT = joinpath(@__DIR__, "results")
seed = design(WORM_P, WORM_SPEC.τ; form=:u, ε₀=0.1)
found = raw_model(seed, vec(readdlm(joinpath(OUT, "worm_raw_theta.csv"), ',')))

t = @elapsed polished, trace = correct_rates(found, WORM_SPEC; passes=3)
for p in trace
    @printf("pass %d ν %s dwell %s off %.4f f %.3g\n", p.pass, join((@sprintf("%.4g", x) for x in p.ν), " "),
        join((@sprintf("%.4f", x) for x in p.dwell), " "), p.off, p.f)
end
println("polish ", round(t; digits=1), " s")
writedlm(joinpath(OUT, "worm_raw_polished_theta.csv"), [log.(polished.α); log.(polished.ν)], ',')

kw = (; seeds=9_100_001:9_100_064, T=4000.0, dtscale=0.01)
for (name, r) in compare("user fit" => USER_FIT, "CMA-ES" => found, "CMA-ES + polish" => polished; spec=WORM_SPEC, kw...)
    show_result(name, r)
    println("   sem% ", round.(100 .* r.sem ./ r.dwell; digits=2), "  off-graph transitions ", sum(r.counts[c] for c in CartesianIndices(WORM_P) if WORM_P[c] == 0))
end
