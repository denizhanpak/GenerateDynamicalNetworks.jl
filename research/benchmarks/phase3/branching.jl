# The generic pipeline on one graph with branches or hidden states, under sampled argmax. Run from
# notebooks/ with
#   julia --project=. -t 8 ../benchmarks/phase3/branching.jl "foraging" 0.1 [budget]
# Each case appends one row to results/branching.csv and one row per branch to branching_edges.csv.
# Branches are measured on two independent held-out seed sets, to separate chance from bias.
include(joinpath(@__DIR__, "worm.jl"))
include(joinpath(@__DIR__, "..", "phase2", "cases.jl"))

const OUT = joinpath(@__DIR__, "results")
mkpath(OUT)
append(file, header, row) = open(joinpath(OUT, file), "a") do io
    filesize(joinpath(OUT, file)) == 0 && println(io, header)
    println(io, join(row, ","))
end
const CASES = "case,noise,stage,hidden,worst_err,rms,off,branches_ok,pass,evals,seconds"

name, ε₀ = ARGS[1], parse(Float64, ARGS[2])
budget = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 500
_, P, τ = only(c for c in NAMED if c[1] == name)
spec = (; P, τ, form=:u, ε₀, sample=minimum(τ) / 10)

try
    t = @elapsed f = fit(spec; budget)
    r = validate(f.model, spec; T=250sum(τ))
    again = validate(f.model, spec; T=250sum(τ), seeds=9_200_001:9_200_064)
    show_result("$name ε=$ε₀ [$(f.stage)]", r)
    hidden = size(f.model.P, 1) - size(P, 1)
    for (b, c) in zip(r.branch, again.branch)
        @printf("    branch %d→%d designed %.2f measured %.3f %s, repeat %.3f %s\n", b[1], b[2], b[3], b[4], b[5] ? "ok" : "outside CI", c[4], c[5] ? "ok" : "outside CI")
        append("branching_edges.csv", "case,noise,from,to,designed,measured,within_ci,repeat_measured", ("\"$name\"", ε₀, b..., c[4]))
    end
    @printf("    hidden %d, evaluations %d, %.0f s\n", hidden, f.evals, t)
    append("branching.csv", CASES, ("\"$name\"", ε₀, f.stage, hidden, maximum(abs, r.err), r.rms, r.off, r.branch_ok, r.pass, f.evals, t))
catch err
    err isa ArgumentError || rethrow()
    msg = first(sprint(showerror, err), 140)
    println("$name ε=$ε₀ NO DESIGN: ", msg)
    append("branching.csv", CASES, ("\"$name\"", ε₀, "no design", 0, NaN, NaN, NaN, false, false, 0, 0.0))
end
