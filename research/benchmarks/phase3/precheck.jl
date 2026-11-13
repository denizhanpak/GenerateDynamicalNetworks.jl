# Quick look before any search. Run from notebooks/ with
#   julia --project=. -t 8 ../benchmarks/phase3/precheck.jl
# (1) the user's worm fit under the new decoder, (2) the formula design at noise 0.1, (3) correction.
include(joinpath(@__DIR__, "worm.jl"))

t = @elapsed r = validate(USER_FIT, WORM_SPEC; seeds=1:64, dtscale=0.01)
show_result("user fit, seeds 1:64", r)
println("   counts ", r.counts, "   ", round(t; digits=1), " s")

trace = correct(WORM_SPEC; passes=6, runs=16, T=1000.0, dtscale=0.01)
for p in trace
    @printf("pass %d request %s dwell %s off %.3f f %.4g %s\n", p.pass, join((@sprintf("%.3f", x) for x in p.request), " "),
        join((@sprintf("%.3f", x) for x in p.dwell), " "), p.off, p.f, p.note)
end
