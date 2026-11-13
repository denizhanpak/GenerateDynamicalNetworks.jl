# The Phase 6 suite: the synchrony-space backend on the six fixture graphs at noise 1e-3 and 0.1, and on
# the real worm graph (Phase 5's worm_data.jl) at 0.1. Per case: a corner layout, the fit, the acceptance
# (held-out sets 1 and 2, and set 1 at half the step) and the class. Every stage resumes across batches:
# the fit saves after each 16-run chunk, the acceptance runs are kept as they finish, and the time spent
# on a case is added up over batches. Exit code 3 means "rerun to continue". Run from notebooks/ with
#   julia --project=. -t 7 ../benchmarks/phase6/suite.jl [start_new_until_seconds] [batch_seconds]
include(joinpath(@__DIR__, "synchrony.jl"))
include(joinpath(@__DIR__, "..", "phase5", "twoway_cases.jl"))     # the worm graph measured from the tracks

# With GDN_SEED1D=1 the fit starts from the 1-D first-passage seed (seed1d.jl, option B) and results go
# to results/seed1d/.
const SEED1D = get(ENV, "GDN_SEED1D", "") == "1"
SEED1D && include(joinpath(@__DIR__, "seed1d.jl"))
const RESULTS6 = SEED1D ? joinpath(@__DIR__, "results", "seed1d") : joinpath(@__DIR__, "results")
mkpath(RESULTS6)
const SUITE = joinpath(RESULTS6, "suite.csv")
san6(s) = replace(s, r"[^A-Za-z0-9]+" => "_")

function add_rows(file, header, rows)
    open(file, "a") do io
        filesize(file) == 0 && println(io, header)
        foreach(r -> println(io, join(r, ",")), rows)
    end
end
done_cases() = isfile(SUITE) ? Set(let c = split(l, r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)"); (strip(c[1], '"'), parse(Float64, c[2])) end
                                   for l in readlines(SUITE)[2:end]) : Set{Tuple{String,Float64}}()

"""
`validate` on the same seeds, simulated 16 runs at a time with finished chunks kept in `store` (saved
to `file`). Returns `nothing` when the next chunk would end after `deadline`; runs are independent per
seed, so the result equals `validate`'s.
"""
function chunked_validate(m, spec, key, store, file; seeds, T, dtscale=0.01, deadline, tol=0.05)
    done = get!(store, key, Any[])
    per_unit = NaN
    for chunk in Iterators.partition(seeds[length(done)+1:end], 16)
        time() + (isnan(per_unit) ? 0.0 : 1.2per_unit * length(chunk) / dtscale) > deadline && return nothing
        started = time()
        append!(done, simulate_frames(m; T, seeds=collect(chunk), spec.sample, dtscale))
        per_unit = (time() - started) / (length(chunk) / dtscale)
        serialize(file, store)
    end
    runs = [x for x in done]
    n = size(spec.P, 1)
    failed = count(r -> r.failed, runs)
    meas = measure(runs, n)
    v = verdict(meas, spec.P, spec.τ; tol)
    return (; dwell=meas.dwell, err=v.err, rms=sqrt(mean(log.(meas.dwell ./ spec.τ) .^ 2)), off=v.off, branch=v.branch,
              pass=v.pass && failed == 0, dwell_ok=all(abs.(v.err) .<= tol), branch_ok=all(last.(v.branch)))
end

worm = worm_data_case()
cases = [[(name, P, τ, ε) for ε in (1e-3, 0.1) for (name, P, τ) in NAMED]; [(worm[1], worm[2], worm[3], 0.1)]]
only_case = get(ENV, "GDN_ONLY", "")                     # run just one case, by name, when set
isempty(only_case) || filter!(c -> c[1] == only_case, cases)
limit = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 240.0
clock = time()
deadline = clock + (length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 480.0)
for (name, P, τ, ε) in cases
    (name, ε) in done_cases() && continue
    key = "$(san6(name))_$ε"
    ckpt, runs_file, fit_file, time_file = (joinpath(RESULTS6, "$(p)_$key$(x)") for (p, x) in
        (("ckpt", ".jls"), ("runs", ".jls"), ("fit", ".jls"), ("elapsed", ".txt")))
    started = isfile(ckpt) || isfile(runs_file) || isfile(fit_file)
    if !started && time() - clock > limit
        println("batch time used; rerun to continue")
        break
    end
    t0 = time()
    spent() = (isfile(time_file) ? parse(Float64, read(time_file, String)) : 0.0) + time() - t0
    stop() = (write(time_file, string(spent())); println("  batch time used; rerun to continue"); exit(3))
    spec = (; P=float(P), τ, form=:u, ε₀=ε, sample=startswith(name, "worm") ? WORM_SPEC.sample : minimum(τ) / 10)
    L = sync_layout(spec.P)
    println("\n$name at noise $ε: $(L.M) pairs, $(transits(L)) transit corners, states on ",
            [corner_name(v, L.M) for v in L.corner_of])
    flush(stdout)
    if !isfile(fit_file)
        f = try
            if SEED1D
                # start the race solve from the rule-of-thumb rates: the mean-only start sticks at the
                # clamp on the worm ring at 1e-3 and gives the same seed in every other case
                seed, c1 = seed_out_race(spec; start=first(rough_rates(spec)))
                fit_sync(spec, L; checkpoint=ckpt, deadline, seed, response=race_response_full(c1, spec.ε₀))
            else
                fit_sync(spec, L; checkpoint=ckpt, deadline)
            end
        catch err
            err isa OutOfTime || rethrow()
            stop()
        end
        serialize(fit_file, (; layout=L, f.out, f.fast, f.contract, f.arrive, f.floor, spec, fit_seconds=spent()))
        rm(ckpt; force=true)
    end
    fit = deserialize(fit_file)
    m = sync_model(spec, L, fit.out; fit.fast, fit.contract, fit.arrive, fit.floor)
    T = 250sum(τ)
    store = isfile(runs_file) ? deserialize(runs_file) : Dict{Symbol,Any}()
    r = chunked_validate(m, spec, :set1, store, runs_file; seeds=9_000_001:9_000_064, T, deadline)
    r === nothing && stop()
    again = chunked_validate(m, spec, :set2, store, runs_file; seeds=9_200_001:9_200_064, T, deadline)
    again === nothing && stop()
    fine = chunked_validate(m, spec, :half, store, runs_file; seeds=9_000_001:9_000_064, T, dtscale=0.005, deadline)
    fine === nothing && stop()
    show_result("sync $name ε=$ε, set 1", r)
    show_result("sync $name ε=$ε, set 2", again)
    show_result("sync $name ε=$ε, half step", fine)
    for (b, c, h) in zip(r.branch, again.branch, fine.branch)
        @printf("    branch %d→%d designed %.2f measured %.3f %s, set 2 %.3f %s, half step %.3f %s\n", b[1], b[2], b[3],
                b[4], b[5] ? "ok" : "outside CI", c[4], c[5] ? "ok" : "outside CI", h[4], h[5] ? "ok" : "outside CI")
    end
    class, outcomes = classify(m; spec, Tnoisy=40sum(τ), Tfree=150sum(τ))
    pass = r.pass && again.pass && fine.pass
    total = spent()
    @printf("  class %s (%s), out-rates %s, pass on all three: %s, fit %.0f s, case %.0f s\n", class, outcomes,
            round.(fit.out; sigdigits=3), pass, fit.fit_seconds, total)
    flush(stdout)
    q = "\"$name\""
    add_rows(joinpath(RESULTS6, "suite_states.csv"), "case,noise,state,target,set1,set2,half_step",
             [(q, ε, i, τ[i], r.dwell[i], again.dwell[i], fine.dwell[i]) for i in eachindex(τ)])
    add_rows(joinpath(RESULTS6, "suite_edges.csv"), "case,noise,from,to,designed,set1,set1_ok,set2,set2_ok,half_step,half_step_ok",
             [(q, ε, b[1], b[2], b[3], b[4], b[5], c[4], c[5], h[4], h[5]) for (b, c, h) in zip(r.branch, again.branch, fine.branch)])
    add_rows(joinpath(RESULTS6, "fits.csv"), "case,noise,edge,out,fast,contract,arrive,floor",
             [(q, ε, e, fit.out[e], fit.fast, fit.contract, fit.arrive, fit.floor) for e in eachindex(fit.out)])
    add_rows(SUITE, "case,noise,pairs,oscillators,transits,worst_err,rms,off,dwell_ok,branches_ok,set2_pass,half_pass,pass,class,outcomes,seconds",
             [(q, ε, L.M, 2L.M, transits(L), maximum(abs, r.err), r.rms, r.off, r.dwell_ok, r.branch_ok, again.pass, fine.pass, pass,
               class, "\"$outcomes\"", fit.fit_seconds)])
    rm(runs_file; force=true)
    rm(time_file; force=true)
end
println("finished: ", length(done_cases()), " of ", length(cases))
