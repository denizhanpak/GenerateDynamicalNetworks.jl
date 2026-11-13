# Phase 2 sweeps: x form against u form, on named graphs at three noise levels and on random
# repaired graphs. Run from notebooks/ with
#   julia --project=. -t 16 ../benchmarks/phase2/sweeps.jl named|random [noises, e.g. 1e-5,1e-3] [tag]
# Each case gets three passes: the first design, one correction (request τ + (τ − measured)),
# and acceptance of the corrected design on held-out seeds at half the step.
include(joinpath(@__DIR__, "horchler.jl"))
using DelimitedFiles, Printf

const OUT = joinpath(@__DIR__, "results")
mkpath(OUT)

include(joinpath(@__DIR__, "cases.jl"))

case_rows, state_rows, branch_rows = Any[], Any[], Any[]

function run_pass(case, form, ε₀, pass, P, τ, request; seeds, dtscale, cycles)
    n = size(P, 1)
    m = design(P, request; form, ε₀)
    T = cycles * sum(τ)
    dt = dtscale / maximum(m.α)
    t = @elapsed runs = simulate(m; T, seeds, dt)
    meas = measure(runs, n)
    v = verdict(meas, P, τ)
    steps = length(seeds) * T / dt
    for i in 1:n
        push!(state_rows, (case, form, ε₀, pass, i, τ[i], request[i], meas.dwell[i], meas.sem[i], length(meas.samples[i])))
    end
    for (i, j, p, f, ok) in v.branch
        push!(branch_rows, (case, form, ε₀, pass, i, j, p, f, sum(meas.counts[i, :]), ok))
    end
    worst = all(isfinite, v.err) ? maximum(abs, v.err) : Inf    # a state never visited counts as infinitely wrong
    push!(case_rows, (case, form, ε₀, pass, size(m.P, 1) - n, m.α[1], worst, all(last.(v.branch)), v.off, v.pass,
                      t, 1e9t * Threads.nthreads() / steps, ""))
    @printf("%-20s %s %.0e %-9s hidden %d  α %6.1f  worst %5.1f%%  branches %s  off %.3f  %s  %.1f s\n",
        case, form, ε₀, pass, size(m.P, 1) - n, m.α[1], 100worst, all(last.(v.branch)) ? "ok " : "BAD", v.off,
        v.pass ? "PASS" : "fail", t)
    return meas
end

function run_case(case, P, τ, form, ε₀; cycles)
    try
        first = run_pass(case, form, ε₀, "first", P, τ, τ; seeds=1:16, dtscale=0.01, cycles)
        request = τ .+ (τ .- first.dwell)
        any(!isfinite, request) && error("a state was never visited")
        run_pass(case, form, ε₀, "corrected", P, τ, request; seeds=17:32, dtscale=0.01, cycles)
        run_pass(case, form, ε₀, "accept", P, τ, request; seeds=101:116, dtscale=0.005, cycles)
    catch err
        msg = replace(sprint(showerror, err), r"\s+" => " ", "," => ";")
        push!(case_rows, (case, form, ε₀, "no design", 0, NaN, NaN, false, NaN, false, NaN, NaN, first(msg, 160)))
        @printf("%-20s %s %.0e  NO DESIGN: %s\n", case, form, ε₀, first(msg, 120))
    end
end

function save(tag)
    q(x) = x isa AbstractString || x isa Symbol ? "\"$(x)\"" : string(x)
    for (name, header, rows) in [
            ("cases", "case,form,noise,pass,hidden,alpha,worst_err,branches_ok,off_graph,accepted,seconds,ns_per_step,error", case_rows),
            ("states", "case,form,noise,pass,state,target,request,dwell,sem,visits", state_rows),
            ("branches", "case,form,noise,pass,from,to,designed,measured,departures,within_ci", branch_rows)]
        open(joinpath(OUT, "$(tag)_$(name).csv"), "w") do io
            println(io, header)
            foreach(r -> println(io, join(q.(r), ",")), rows)
        end
    end
end

group = isempty(ARGS) ? "named" : ARGS[1]
println("threads ", Threads.nthreads(), ", group ", group)
noises = length(ARGS) >= 2 ? parse.(Float64, split(ARGS[2], ",")) : [1e-5, 1e-3, 1e-2]
tag = length(ARGS) >= 3 ? ARGS[3] : group
if group == "named"
    for (case, P, τ) in NAMED, ε₀ in noises, form in (:x, :u)
        run_case(case, P, τ, form, ε₀; cycles=100)
    end
elseif group == "random"
    rng = Xoshiro(2026)
    for g in 1:20
        P = random_spec_matrix(rng, 6, 0.3)
        τ = round.(rand(rng, Uniform(0.5, 3.0), 6); digits=2)
        writedlm(joinpath(OUT, "random_graph_$g.csv"), [P τ], ',')
        for form in (:x, :u)
            run_case("random $g", P, τ, form, 1e-5; cycles=100)
        end
    end
end
save(tag)
