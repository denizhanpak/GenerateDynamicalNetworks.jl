# Checks for the 2026-11-13 review fixes: rough_rates (now in synchrony.jl) equals every stored contraction, and
# fit_hop's no-design fallback (seed with no design → clear error; final update with none → keep the last pass).
# Run from notebooks/ with  julia --project=. -t 7 ../benchmarks/phase7/test_fallback.jl   (about a minute)
include(joinpath(@__DIR__, "hopfield.jl"))
using Test
# 1. rough_rates moved, values unchanged: contraction matches every saved Phase 7 fit
include(joinpath(@__DIR__, "..", "phase5", "twoway_cases.jl"))
fitrows = readlines(joinpath(@__DIR__, "results", "fits.csv"))[2:end]
csplit(l) = split(l, r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)")
saved = Dict((strip(csplit(l)[1], '"'), parse(Float64, csplit(l)[2])) => parse(Float64, csplit(l)[end]) for l in fitrows)
worm = worm_data_case()
for (name, P, τ, ε) in [[(n, P, τ, ε) for ε in (1e-3, 0.1) for (n, P, τ) in NAMED]; [(worm[1], worm[2], worm[3], 0.1)]]
    _, c = rough_rates((; P=float(P), τ, ε₀=ε))
    @test c ≈ saved[(name, ε)]
end
println("rough_rates matches all ", length(saved), " saved contractions")
# 2. failure injection: make the k-th model build come back with no design
calls, poison = Ref(0), Ref(0)
spec = (; P=float(ring(3)), τ=[0.1, 0.3, 1.0], form=:u, ε₀=0.1, sample=0.01)
L = hop_layout(spec)
function hop_model(s::typeof(spec), L, out; kw...)
    m, w = invoke(hop_model, Tuple{Any,Any,Any}, s, L, out; kw...)
    calls[] += 1
    calls[] == poison[] && fill!(m.J, NaN)
    return m, w
end
ck = tempname() * ".jls"
# (a) the seed itself has no design, nothing saved yet → a clear error, not a file error
rm(ck; force=true); rm(ck * ".seed"; force=true); calls[] = 0; poison[] = 1
err = try fit_hop(spec, L; checkpoint=ck, passes=1); nothing catch e; e end
@test err isa ErrorException && occursin("seed rates have no pairwise design", err.msg)
println("(a) ", err.msg)
# (b) one pass runs, then the final update has no design → keep pass 1's simulated rates
rm(ck; force=true); rm(ck * ".seed"; force=true); calls[] = 0; poison[] = 2
f = fit_hop(spec, L; checkpoint=ck, passes=1)
@test f.stopped == 2 && f.out == f.trace[1].out && all(isfinite, f.model.J)
println("(b) stopped=", f.stopped, ", kept pass 1 rates ", round.(f.out; sigdigits=4))
# (c) normal path untouched
rm(ck; force=true); rm(ck * ".seed"; force=true); calls[] = 0; poison[] = 0
g = fit_hop(spec, L; checkpoint=ck, passes=1)
@test g.stopped === nothing && maximum(abs, g.out) <= 2g.contract
println("(c) stopped=nothing, out ", round.(g.out; sigdigits=4))
