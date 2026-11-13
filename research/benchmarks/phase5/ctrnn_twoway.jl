# Two-way edges through relay cells (see ctrnn_design.jl): fit and check on four made-up graphs, and a
# regression check that the relay changes leave the earlier CTRNN fits exactly as they were.
# Run from notebooks/ with
#   julia --project=. -t 8 ../benchmarks/phase5/ctrnn_twoway.jl fit <case>
#   julia --project=. -t 8 ../benchmarks/phase5/ctrnn_twoway.jl check <case>     (the acceptance)
#   julia --project=. -t 8 ../benchmarks/phase5/ctrnn_twoway.jl extra <case>     (third seed set, CV, class, relay use)
#   julia --project=. -t 8 ../benchmarks/phase5/ctrnn_twoway.jl regress
include(joinpath(@__DIR__, "classes.jl"))          # the classifier, and backends.jl through it
include(joinpath(@__DIR__, "ctrnn_design.jl"))

include(joinpath(@__DIR__, "twoway_cases.jl"))
theta_file(name) = joinpath(OUT, "theta_ctrnn_twoway_$(san(name))_0.1.csv")
"""Rows for one case, replacing any that case already had, so a rerun doesn't duplicate it."""
function replace_rows(file, header, name, rows)
    keep = isfile(file) ? filter(l -> !startswith(l, "\"$name\","), readlines(file)[2:end]) : String[]
    open(file, "w") do io
        println(io, header)
        foreach(l -> println(io, l), keep)
        foreach(r -> println(io, join(r, ",")), rows)
    end
end
function append_row(file, header, row)
    open(file, "a") do io
        filesize(file) == 0 && println(io, header)
        println(io, join(row, ","))
    end
end

"""
One long run saved every half frame. For each relay (j → i, cell k), counts the decoded j → i switches
and how many had k above threshold within the 3τₖ before the switch, and how often k crossed threshold.
"""
function relay_check(m, spec, layout; seed=7, T=100sum(spec.τ))
    every, dt = frame_steps(m, spec.sample, 0.01)
    sol = solve(SDEProblem(drift_ctrnn, noise_flat, ctrnn_start(m), (0.0, T), ctrnn_params(m); seed=UInt64(seed)), EM();
                dt, adaptive=false, saveat=spec.sample / 2, maxiters=typemax(Int))
    state = [argmax(i -> y[i], 1:layout.n) for y in sol.u]
    map(enumerate(layout.rel)) do (r, (j, i))
        k = layout.n + r
        window = 3layout.τc[k]
        switches = [f for f in 2:length(state) if state[f-1] == j && state[f] == i]
        preceded = count(f -> any(g -> sol.u[g][k] > 0, findall(t -> sol.t[f] - window <= t <= sol.t[f], sol.t)), switches)
        ups = [sol.t[f] for f in 2:length(sol.u) if sol.u[f-1][k] <= 0 < sol.u[f][k]]
        episodes = count(e -> e == 1 || ups[e] - ups[e-1] > window, eachindex(ups))   # crossings 3τₖ apart count once
        (; edge="$j→$i", switches=length(switches), preceded, episodes)
    end
end

"""
`validate` on the same seeds, but simulated 16 runs at a time with finished chunks kept in `store`
(a Dict saved to `file`), so a long check can span batches. Returns `nothing` when the next chunk
would end after `deadline`; the runs are independent per seed, so the result equals `validate`'s.
"""
function chunked_validate(m, spec, key, store, file; seeds, T, dtscale=0.01, deadline, tol=0.05)
    done = get!(store, key, Any[])
    for chunk in Iterators.partition(seeds[length(done)+1:end], 16)
        est = get(store, :per_unit, NaN) * length(chunk) / dtscale
        time() + (isnan(est) ? 0.0 : 1.2est) > deadline && return nothing
        started = time()
        append!(done, simulate_frames(m; T, seeds=collect(chunk), spec.sample, dtscale))
        store[:per_unit] = (time() - started) / (length(chunk) / dtscale)
        serialize(file, store)
    end
    runs = [x for x in done]
    n = size(spec.P, 1)
    failed = count(r -> r.failed, runs)
    meas = measure(runs, n)
    v = verdict(meas, spec.P, spec.τ; tol)
    return (; dwell=meas.dwell, sem=meas.sem, err=v.err, rms=sqrt(mean(log.(meas.dwell ./ spec.τ) .^ 2)), off=v.off,
              branch=v.branch, counts=meas.counts, failed, pass=v.pass && failed == 0,
              dwell_ok=all(abs.(v.err) .<= tol), branch_ok=all(last.(v.branch)))
end
store_file(name) = joinpath(OUT, "twoway_runs_$(san(name)).jls")
load_store(name) = let st = isfile(store_file(name)) ? deserialize(store_file(name)) : Dict{Any,Any}()
    st[:per_unit] = NaN          # timing is measured afresh in each batch
    st
end
out_of_time() = (println("  batch time used; rerun to continue"); exit(3))
need(x) = x === nothing ? out_of_time() : x

mode = ARGS[1]
if mode == "fit"
    name = ARGS[2]
    spec = twoway_spec(name)
    layout = design_layout(spec)
    println(name, ": ", layout.n, " states, relays ", layout.rel, ", cell τ ", round.(layout.τc; sigdigits=3))
    ckpt = joinpath(OUT, "twoway_checkpoint_$(san(name)).jls")
    t = @elapsed f = try
        fit_ctrnn_design(spec; checkpoint=ckpt, deadline=time() + 480)
    catch err
        err isa OutOfTime || rethrow()
        println("  batch time used; rerun to continue")
        exit(3)
    end
    rm(ckpt; force=true)
    rm(store_file(name); force=true)          # check runs of an earlier fit no longer apply
    writedlm(theta_file(name), f.θ, ',')
    @printf("  last batch %.0f s, step-rule rate %.0f\n", t, f.model.α[1])
    append_row(joinpath(OUT, "ctrnn_twoway_fits.csv"), "case,cells,relays,rate,seconds", ("\"$name\"", length(layout.τc), length(layout.rel), f.model.α[1], t))
elseif mode == "check"          # the acceptance: held-out sets 1 and 2, and set 1 at half the step
    name = ARGS[2]
    spec = twoway_spec(name)
    m = ctrnn_model(spec, vec(readdlm(theta_file(name), ',')); cap=Inf)
    T = 250sum(spec.τ)
    store, deadline = load_store(name), time() + 480
    r = need(chunked_validate(m, spec, :set1, store, store_file(name); seeds=9_000_001:9_000_064, T, deadline))
    again = need(chunked_validate(m, spec, :set2, store, store_file(name); seeds=9_200_001:9_200_064, T, deadline))
    fine = need(chunked_validate(m, spec, :half, store, store_file(name); seeds=9_000_001:9_000_064, T, dtscale=0.005, deadline))
    show_result("$name, held-out set 1", r)
    show_result("$name, held-out set 2", again)
    show_result("$name, set 1 at half the step", fine)
    for (b, c, f) in zip(r.branch, again.branch, fine.branch)
        @printf("    branch %d→%d designed %.2f measured %.3f %s, set 2 %.3f %s, half step %.3f %s\n", b[1], b[2], b[3],
                b[4], b[5] ? "ok" : "outside CI", c[4], c[5] ? "ok" : "outside CI", f[4], f[5] ? "ok" : "outside CI")
    end
    q = "\"$name\""
    replace_rows(joinpath(OUT, "ctrnn_twoway.csv"), "case,worst_err,rms,off,dwell_ok,branches_ok,repeat_pass,fine_pass,pass", name,
                 [(q, maximum(abs, r.err), r.rms, r.off, r.dwell_ok, r.branch_ok, again.pass, fine.pass, r.pass && again.pass && fine.pass)])
    replace_rows(joinpath(OUT, "ctrnn_twoway_states.csv"), "case,state,target,set1,set2,half_step", name,
                 [(q, i, spec.τ[i], r.dwell[i], again.dwell[i], fine.dwell[i]) for i in eachindex(spec.τ)])
    replace_rows(joinpath(OUT, "ctrnn_twoway_branches.csv"), "case,from,to,designed,set1,set1_ok,set2,set2_ok,half_step,half_step_ok", name,
                 [(q, b[1], b[2], b[3], b[4], b[5], c[4], c[5], f[4], f[5]) for (b, c, f) in zip(r.branch, again.branch, fine.branch)])
    println("  pass on all three: ", r.pass && again.pass && fine.pass)
elseif mode == "extra"          # information beyond the acceptance: a third seed set, bout CV, class, relay use
    name = ARGS[2]
    spec = twoway_spec(name)
    layout = design_layout(spec)
    m = ctrnn_model(spec, vec(readdlm(theta_file(name), ',')); cap=Inf)
    T = 250sum(spec.τ)
    store, deadline = load_store(name), time() + 420
    third = need(chunked_validate(m, spec, :third, store, store_file(name); seeds=9_400_001:9_400_064, T, deadline))
    println("  third held-out set: dwell errors ", round.(100 .* third.err; digits=1), "%, off-graph ", round(100third.off; digits=2),
            "%, branches ", join((@sprintf("%d→%d %.3f %s", b[1], b[2], b[4], b[5] ? "ok" : "outside CI") for b in third.branch), ", "))
    cv = [std(x) / mean(x) for x in measure(simulate_frames(m; T, seeds=9_000_001:9_000_016, spec.sample, dtscale=0.01), layout.n).samples]
    println("  bout CV ", round.(cv; digits=2))
    class, outcomes = classify(m; spec, Tnoisy=40sum(spec.τ), Tfree=150sum(spec.τ))
    println("  class ", class, ": ", outcomes)
    relays = relay_check(m, spec, layout)
    for x in relays
        println("  relay for $(x.edge): $(x.switches) switches, $(x.preceded) with the relay on just before")
    end
    q = "\"$name\""
    replace_rows(joinpath(OUT, "ctrnn_twoway_extra.csv"), "case,third_pass,third_branches_ok,class,relay_switches,relay_preceded", name,
                 [(q, third.pass, third.branch_ok, class, sum(x -> x.switches, relays), sum(x -> x.preceded, relays))])
    replace_rows(joinpath(OUT, "ctrnn_twoway_cv.csv"), "case,state,cv", name, [(q, i, cv[i]) for i in eachindex(cv)])
elseif mode == "regress"
    # The suite's numbers for held-out set 1 against the same check under the relay-aware code.
    rows = [split(l, r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)") for l in readlines(joinpath(OUT, "suite.csv"))[2:end]]
    for c in filter(c -> c[3] == "ctrnn" && c[4] in ("design", "excitable"), rows)
        local name, regime = strip(c[1], '"'), c[4]
        local spec = fit_spec(name, 0.1)
        local m = ctrnn_model(spec, vec(readdlm(joinpath(OUT, "theta_ctrnn_$(regime)_$(san(name))_0.1.csv"), ',')); cap=Inf)
        v = validate(m, spec; T=250sum(spec.τ))
        same = v.rms == parse(Float64, c[9]) && v.off == parse(Float64, c[10])
        @printf("  %-22s %-9s rms %.6f (suite %.6f)  off %.4f (suite %.4f)  %s\n", name, regime, v.rms, parse(Float64, c[9]),
                v.off, parse(Float64, c[10]), same ? "identical" : "DIFFERENT")
        flush(stdout)
    end
end
