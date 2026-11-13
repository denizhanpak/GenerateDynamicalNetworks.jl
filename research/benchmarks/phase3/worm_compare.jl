# Saves the tables the notebook reads: every worm model on the same long held-out runs, and the
# correction passes in dwell-request space. Run from notebooks/ with
#   julia --project=. -t 8 ../benchmarks/phase3/worm_compare.jl
include(joinpath(@__DIR__, "worm.jl"))
using DelimitedFiles

const OUT = joinpath(@__DIR__, "results")
θ(name) = vec(readdlm(joinpath(OUT, "worm_$(name)_theta.csv"), ','))
seed = design(WORM_P, WORM_SPEC.τ; form=:u, ε₀=0.1)

kw = (; seeds=9_100_001:9_100_064, T=4000.0, dtscale=0.01)
results = compare("formula seed" => seed, "CMA-ES on dwell requests" => request_model(WORM_SPEC, θ("request")),
    "your fit" => USER_FIT, "CMA-ES on α and ν" => raw_model(seed, θ("raw")),
    "CMA-ES on α and ν + polish" => raw_model(seed, θ("raw_polished")); spec=WORM_SPEC, kw...)
open(joinpath(OUT, "worm_compare.csv"), "w") do io
    println(io, "model,d1,d2,d3,d4,e1,e2,e3,e4,sem1,sem2,sem3,sem4,rms,off,pass")
    for (name, r) in results
        show_result(name, r)
        println(io, join(["\"$name\""; r.dwell; r.err; r.sem; r.rms; r.off; r.pass], ","))
    end
end

trace = correct(WORM_SPEC; passes=6, runs=16, T=1000.0, dtscale=0.01)
open(joinpath(OUT, "worm_correction.csv"), "w") do io
    println(io, "pass,r1,r2,r3,r4,d1,d2,d3,d4,off,note")
    for p in trace
        println(io, join([p.pass; p.request; p.dwell; p.off; "\"" * replace(first(p.note, 90), "," => ";", "\"" => "") * "\""], ","))
    end
end
