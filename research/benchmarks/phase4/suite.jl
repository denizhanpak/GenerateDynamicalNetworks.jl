# The Phase 4 test suite: both backends on the six Phase 2 graphs at two noise levels (24 fits).
# Finished fits are skipped, so running this again carries on. Run from notebooks/ with
#   julia --project=. -t 16 ../benchmarks/phase4/suite.jl [start_new_until_seconds] [seconds_per_search]
# New fits stop starting after the first number of seconds, so a batch stays under the time cap.
include(joinpath(@__DIR__, "backends.jl"))
using DelimitedFiles

const OUT = joinpath(@__DIR__, "results")
mkpath(OUT)
const CASES_CSV = joinpath(OUT, "suite.csv")
const EDGES_CSV = joinpath(OUT, "suite_edges.csv")
const HEADER = "case,noise,backend,stage,seeded,dim,worst_err,rms,off,dwell_ok,branches_ok,repeat_branches_ok,failed,pass,evals,seconds"

function append(file, header, row)
    open(file, "a") do io
        filesize(file) == 0 && println(io, header)
        println(io, join(row, ","))
    end
end

key_of(name, ε₀, backend) = "$name|$ε₀|$backend"
finished() = isfile(CASES_CSV) ? Set(let c = split(l, r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)"); key_of(strip(c[1], '"'), parse(Float64, c[2]), c[3]) end for l in readlines(CASES_CSV)[2:end]) : Set{String}()

limit = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 150.0
per_search = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 300.0
clock = time()
todo = [(n, ε₀, b) for ε₀ in (1e-3, 0.1) for b in (:simplex, :cylinder) for n in first.(NAMED)]
makers = Dict(:simplex => make_simplex, :cylinder => make_cylinder)

for (name, ε₀, backend) in todo
    key_of(name, ε₀, backend) in finished() && continue
    if time() - clock > limit
        println("batch time used; rerun to continue")
        break
    end
    _, P, τ = only(c for c in NAMED if c[1] == name)
    spec = (; P, τ, form=:u, ε₀, sample=name == "worm ring" ? WORM_SPEC.sample : minimum(τ) / 10)
    t = @elapsed f = fit_backend(spec, makers[backend]; seconds=per_search)
    r = validate(f.model, spec; T=250sum(τ))
    again = validate(f.model, spec; T=250sum(τ), seeds=9_200_001:9_200_064)
    show_result("$backend $name ε=$ε₀ [$(f.stage)]", r)
    for (b, c) in zip(r.branch, again.branch)
        @printf("    branch %d→%d designed %.2f measured %.3f %s, repeat %.3f %s\n", b[1], b[2], b[3], b[4], b[5] ? "ok" : "outside CI", c[4], c[5] ? "ok" : "outside CI")
        append(EDGES_CSV, "case,noise,backend,from,to,designed,measured,within_ci,repeat_measured", ("\"$name\"", ε₀, backend, b..., c[4]))
    end
    @printf("    dim %d, seeded %s, evaluations %d, failed runs %d, %.0f s\n", f.dim, f.seeded, f.evals, r.failed, t)
    flush(stdout)
    writedlm(joinpath(OUT, "theta_$(backend)_$(replace(name, r"[^A-Za-z0-9]+" => "_"))_$ε₀.csv"), f.θ, ',')
    append(CASES_CSV, HEADER, ("\"$name\"", ε₀, backend, f.stage, f.seeded, f.dim, maximum(abs, r.err), r.rms, r.off, r.dwell_ok, r.branch_ok, again.branch_ok, r.failed, r.pass, f.evals, t))
end
println("finished: ", length(finished()), " of ", length(todo))
