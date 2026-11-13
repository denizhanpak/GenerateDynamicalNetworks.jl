# The Phase 5 suite: two-population on the six Phase 2 graphs at two noise levels, CTRNN searched from
# three regimes on the worm ring and foraging at noise 0.1, and the CTRNN edge design on all six graphs
# at 0.1 (24 fits). Finished fits are skipped. Run from notebooks/ with
#   julia --project=. -t 8 ../benchmarks/phase5/suite.jl [start_new_until_seconds] [two_search_seconds] [ctrnn_search_seconds]
include(joinpath(@__DIR__, "ctrnn_design.jl"))      # includes backends.jl

const OUT = joinpath(@__DIR__, "results")
mkpath(OUT)
const CASES_CSV = joinpath(OUT, "suite.csv")
const EDGES_CSV = joinpath(OUT, "suite_edges.csv")
const HEADER = "case,noise,backend,regime,stage,seeded,dim,worst_err,rms,off,dwell_ok,branches_ok,repeat_branches_ok,failed,pass,evals,seconds,gate_f"

function append(file, header, row)
    open(file, "a") do io
        filesize(file) == 0 && println(io, header)
        println(io, join(row, ","))
    end
end

key_of(name, ε₀, backend, regime) = "$name|$ε₀|$backend|$regime"
function finished()
    isfile(CASES_CSV) || return Set{String}()
    Set(let c = split(l, r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)")
        key_of(strip(c[1], '"'), parse(Float64, c[2]), c[3], c[4])
    end for l in readlines(CASES_CSV)[2:end])
end

limit = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 380.0
two_seconds = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 120.0
ctrnn_seconds = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 280.0
clock = time()
todo = [[(n, ε₀, :two, :default) for ε₀ in (1e-3, 0.1) for n in first.(NAMED)];
        [(n, 0.1, :ctrnn, r) for n in ("worm ring", "foraging") for r in (:excitable, :oscillator, :fold)];
        [(n, 0.1, :ctrnn, :design) for n in first.(NAMED)]]

"""A design fit the pilot (`ctrnn_design.jl`) already made: the same procedure, read back with its fit time."""
function pilot_design(name, spec)
    file = joinpath(OUT, "theta_ctrnn_design_$(replace(name, r"[^A-Za-z0-9]+" => "_"))_$(spec.ε₀).csv")
    log_file = joinpath(OUT, "ctrnn_design.csv")
    (isfile(file) && isfile(log_file)) || return nothing
    row = findlast(l -> startswith(l, "\"$name\",$(spec.ε₀),"), readlines(log_file))
    row === nothing && return nothing
    θ = vec(readdlm(file, ','))
    f = (; model=ctrnn_model(spec, θ; cap=Inf), θ, stage=:correction, evals=0, seeded=true, dim=count(>(0), spec.P))
    return f, parse(Float64, last(split(readlines(log_file)[row], ',')))
end

for (name, ε₀, backend, regime) in todo
    key_of(name, ε₀, backend, regime) in finished() && continue
    if time() - clock > limit
        println("batch time used; rerun to continue")
        break
    end
    _, P, τ = only(c for c in NAMED if c[1] == name)
    spec = (; P, τ, form=:u, ε₀, sample=name == "worm ring" ? WORM_SPEC.sample : minimum(τ) / 10)
    pilot = regime === :design ? pilot_design(name, spec) : nothing
    if pilot !== nothing
        f, t = pilot
    else
        t = @elapsed f = backend === :two ? fit_two(spec; seconds=two_seconds) :
                         regime === :design ? fit_ctrnn_design(spec) : fit_ctrnn(spec, regime; seconds=ctrnn_seconds)
    end
    gate = score(simulate_frames(f.model; T=250sum(τ), seeds=8_000_001:8_000_032, spec.sample, dtscale=0.01), spec).f
    r = validate(f.model, spec; T=250sum(τ))
    again = validate(f.model, spec; T=250sum(τ), seeds=9_200_001:9_200_064)
    show_result("$backend/$regime $name ε=$ε₀ [$(f.stage)]", r)
    for (b, c) in zip(r.branch, again.branch)
        @printf("    branch %d→%d designed %.2f measured %.3f %s, repeat %.3f %s\n", b[1], b[2], b[3], b[4], b[5] ? "ok" : "outside CI", c[4], c[5] ? "ok" : "outside CI")
        append(EDGES_CSV, "case,noise,backend,regime,from,to,designed,measured,within_ci,repeat_measured", ("\"$name\"", ε₀, backend, regime, b..., c[4]))
    end
    @printf("    dim %d, evaluations %d, failed runs %d, held-out score %.4f, %.0f s\n", f.dim, f.evals, r.failed, gate, t)
    flush(stdout)
    writedlm(joinpath(OUT, "theta_$(backend)_$(regime)_$(replace(name, r"[^A-Za-z0-9]+" => "_"))_$ε₀.csv"), f.θ, ',')
    append(CASES_CSV, HEADER, ("\"$name\"", ε₀, backend, regime, f.stage, f.seeded, f.dim, maximum(abs, r.err), r.rms, r.off, r.dwell_ok, r.branch_ok, again.branch_ok, r.failed, r.pass, f.evals, t, gate))
end
println("finished: ", length(finished()), " of ", length(todo))
