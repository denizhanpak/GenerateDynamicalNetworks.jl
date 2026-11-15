# The benchmark suite: every backend, with the package's default options, fitted on its cases (common.jl) and
# checked by the acceptance rule (`accept`: held-out seed sets 1 and 2, and set 1 at half the step) and the
# classifier. Results are appended to results/cases.csv, states.csv and edges.csv; finished cases are skipped,
# so the suite runs in batches. From the repository root:
#
#   julia --project=benchmarks -t 7 benchmarks/suite.jl [backend ...] [--only=case] [--until=s] [--deadline=s]
#
# With no backend named, all of them. --only runs one case (by name, every noise level). --until stops starting
# new cases after s seconds; --deadline stops the fits that can resume (CTRNN, synchrony, Hopfield) after s seconds,
# saving a checkpoint the next batch picks up. Exit code 3 means cases are left: run the same command again.
#
# Columns
#   cases.csv   backend, case, noise, states, edges, worst_err, rms, off, dwell_ok, branches_ok, failed,
#               set1_pass, set2_pass, half_pass, pass, class, fit_seconds, check_seconds, threads, commit, date, note
#               (worst_err … branches_ok and failed are set 1's; fit_seconds is summed over batches)
#   states.csv  backend, case, noise, state, target, set1, set1_sem, set2, half_step   (mean dwells)
#   edges.csv   backend, case, noise, from, to, designed, set1, set1_ok, set2, set2_ok, half_step, half_step_ok
include(joinpath(@__DIR__, "common.jl"))

const CASES_CSV, STATES_CSV, EDGES_CSV = joinpath.(RESULTS, ("cases.csv", "states.csv", "edges.csv"))
const CASES_HEADER = ["backend", "case", "noise", "states", "edges", "worst_err", "rms", "off", "dwell_ok", "branches_ok",
                      "failed", "set1_pass", "set2_pass", "half_pass", "pass", "class", "fit_seconds", "check_seconds",
                      "threads", "commit", "date", "note"]
const STATES_HEADER = ["backend", "case", "noise", "state", "target", "set1", "set1_sem", "set2", "half_step"]
const EDGES_HEADER = ["backend", "case", "noise", "from", "to", "designed", "set1", "set1_ok", "set2", "set2_ok",
                      "half_step", "half_step_ok"]

commit() = try readchomp(`git -C $(@__DIR__) rev-parse --short HEAD`) catch; "unknown" end

"""Fit and check one case. Returns false if the batch ran out of time partway through a resumable fit."""
function run_case(key, name, ε, deadline)
    b = backend_entry(key)
    spec = case_spec(name, ε)
    mkpath(CHECKPOINTS)
    stem = joinpath(CHECKPOINTS, file_key(key, name, ε))
    before = isfile(stem * ".elapsed") ? parse(Float64, read(stem * ".elapsed", String)) : 0.0
    t0 = time()
    model = try
        b.resumable ? fit(spec, b.backend; checkpoint=stem * ".jls", deadline) : fit(spec, b.backend)
    catch err
        if err isa OutOfTime
            write(stem * ".elapsed", string(before + time() - t0))
            println("  $key $name ε=$ε: batch time used; it resumes next batch")
            return false
        end
        err isa ArgumentError || rethrow()
        # no model (e.g. the design has no solution): one row saying so
        note = sprint(showerror, err)
        append_row(CASES_CSV, CASES_HEADER, (key, name, ε, nstates(spec), length(edge_list(spec)), NaN, NaN, NaN, false,
                                             false, 0, false, false, false, false, :none, before + time() - t0, 0.0,
                                             Threads.nthreads(), commit(), now(), note))
        println("  $key $name ε=$ε: no model ($note)")
        return true
    end
    fit_seconds = before + time() - t0
    t1 = time()
    a = accept(model, spec)
    k = classify(model, spec)
    check_seconds = time() - t1
    s1, s2, h = a.set1, a.set2, a.half_step
    append_row(CASES_CSV, CASES_HEADER, (key, name, ε, nstates(spec), length(edge_list(spec)), maximum(abs, s1.err), s1.rms,
                                         s1.off, s1.dwell_ok, s1.branch_ok, s1.failed, s1.pass, s2.pass, h.pass, a.pass,
                                         k.class, fit_seconds, check_seconds, Threads.nthreads(), commit(), now(), ""))
    for i in 1:nstates(spec)
        append_row(STATES_CSV, STATES_HEADER, (key, name, ε, i, spec.τ[i], s1.dwell[i], s1.sem[i], s2.dwell[i], h.dwell[i]))
    end
    for (e1, e2, eh) in zip(s1.branch, s2.branch, h.branch)
        append_row(EDGES_CSV, EDGES_HEADER, (key, name, ε, e1.from, e1.to, e1.designed, e1.measured, e1.ok, e2.measured, e2.ok,
                                             eh.measured, eh.ok))
    end
    rm(stem * ".jls"; force=true)
    rm(stem * ".elapsed"; force=true)
    @printf("  %s %s ε=%g: %s (set 1 %s, set 2 %s, half step %s), worst error %.1f%%, %s, fit %.0f s, checks %.0f s\n",
            key, name, ε, a.pass ? "PASS" : "fail", s1.pass, s2.pass, h.pass, 100maximum(abs, s1.err), k.class, fit_seconds,
            check_seconds)
    return true
end

function main(args)
    opt(name) = (i = findfirst(startswith("--$name="), args); i === nothing ? nothing : split(args[i], "="; limit=2)[2])
    until = something(tryparse(Float64, something(opt("until"), "Inf")))
    deadline = time() + something(tryparse(Float64, something(opt("deadline"), "Inf")))
    only = opt("only")
    keys = filter(a -> !startswith(a, "--"), args)
    isempty(keys) && (keys = BACKEND_NAMES)
    for k in keys
        k in BACKEND_NAMES || error("unknown backend $(repr(k)); the backends are $(join(BACKEND_NAMES, ", "))")
    end
    done = Set(case_key(r[1], r[2], parse(Float64, r[3])) for r in read_rows(CASES_CSV))
    todo = [(k, name, ε) for k in keys for (name, ε) in backend_entry(k).cases
            if !(case_key(k, name, ε) in done) && (only === nothing || name == only)]
    println("$(length(todo)) cases to run on $(Threads.nthreads()) threads")
    start = time()
    for (k, name, ε) in todo
        if time() - start > until
            println("batch time used; run again to continue")
            return 3
        end
        run_case(k, name, ε, deadline) || return 3
        flush(stdout)
    end
    println("finished")
    return 0
end

exit(main(ARGS))
