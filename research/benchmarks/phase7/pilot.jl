# The Phase 7 pilot: the 3-state ring with dwells 1, 3 and 10 on the Hopfield network, at one noise
# level. A check that every vertex's Jacobian has the designed rates, then the fit, the acceptance
# (held-out sets 1 and 2, and set 1 at half the step), a third set, bout CV and class. Checkpointed:
# exit code 3 means "rerun to continue". Run from notebooks/ with
#   julia --project=. -t 7 ../benchmarks/phase7/pilot.jl <noise> [batch_seconds]
include(joinpath(@__DIR__, "hopfield.jl"))

const OUT = joinpath(@__DIR__, "results")
mkpath(OUT)
noise = parse(Float64, ARGS[1])
deadline = time() + (length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 480.0)
spec = (; P=float(ring(3)), τ=[1.0, 3.0, 10.0], form=:u, ε₀=noise, sample=0.1)
L = hop_layout(spec)
println("3-ring at noise $noise: $(L.M) neurons, states on ", [corner_name(v, L.M) for v in L.corner_of],
        ", paths ", [join(corner_name.(p, L.M), "→") for p in L.paths])

let (rough, c) = rough_rates(spec), (m, worst) = hop_model(spec, L, rough; fast=c, contract=c), q = hop_params(m)
    gap = 0.0
    for v in 1:2^m.n
        ψ = SVector{m.n}(ntuple(j -> corner_bit(v, j) == 1 ? Float64(π) : 0.0, m.n))
        ev = sort(real.(eigvals(jac(drift_hop, q, ψ))))
        gap = max(gap, maximum(abs, ev .- sort([vertex_rate(m.J, m.b, m.k, v, j) for j in 1:m.n])), maximum(abs, drift_hop(ψ, q, 0.0)))
    end
    @printf("vertex check: design violation %.1e, largest gap between Jacobian eigenvalues and design rates %.1e\n", worst, gap)
end

ckpt, fit_file, runs_file = (joinpath(OUT, "pilot_$(p)_$noise.jls") for p in ("ckpt", "fit", "runs"))
stop() = (println("  batch time used; rerun to continue"); exit(3))
if !isfile(fit_file)
    t0 = time()
    f = try
        fit_hop(spec, L; checkpoint=ckpt, deadline)
    catch err
        err isa OutOfTime || rethrow()
        stop()
    end
    serialize(fit_file, (; f.out, f.fast, f.contract, f.seed, f.worst, f.stopped, seconds=time() - t0))
    rm(ckpt; force=true); rm(ckpt * ".seed"; force=true)
end
fitted = deserialize(fit_file)
get(fitted, :stopped, nothing) === nothing ||
    println("fit stopped: no pairwise design for the rates after pass $(fitted.stopped - 1); checking pass $(fitted.stopped - 1)")
m, worst = hop_model(spec, L, fitted.out; fast=fitted.fast, contract=fitted.contract)
@printf("fitted out-rates %s, race seed %s, transit %.3g, contraction %.3g, design violation %.1e\n",
        round.(fitted.out; sigdigits=4), fitted.seed === nothing ? "–" : round.(fitted.seed; sigdigits=4), fitted.fast, fitted.contract, worst)
T = 250sum(spec.τ)
store = isfile(runs_file) ? deserialize(runs_file) : Dict{Symbol,Any}()
function chunked(key, seeds, dtscale)
    done = get!(store, key, Any[])
    for chunk in Iterators.partition(seeds[length(done)+1:end], 16)
        time() > deadline - 60 && (serialize(runs_file, store); stop())
        append!(done, simulate_frames(m; T, seeds=collect(chunk), spec.sample, dtscale))
        serialize(runs_file, store)
    end
    runs = [x for x in done]
    meas = measure(runs, 3)
    v = verdict(meas, spec.P, spec.τ; tol=0.05)
    return (; dwell=meas.dwell, err=v.err, rms=sqrt(mean(log.(meas.dwell ./ spec.τ) .^ 2)), off=v.off, branch=v.branch,
              pass=v.pass && !any(r -> r.failed, runs), samples=meas.samples,
              dwell_ok=all(abs.(v.err) .<= 0.05), branch_ok=all(last.(v.branch)))
end
r = chunked(:set1, 9_000_001:9_000_064, 0.01)
again = chunked(:set2, 9_200_001:9_200_064, 0.01)
fine = chunked(:half, 9_000_001:9_000_064, 0.005)
third = chunked(:third, 9_400_001:9_400_064, 0.01)
show_result("hopfield pilot ε=$noise, held-out set 1", r)
show_result("hopfield pilot ε=$noise, held-out set 2", again)
show_result("hopfield pilot ε=$noise, set 1 at half the step", fine)
show_result("hopfield pilot ε=$noise, third set (information)", third)
cv = [std(x) / mean(x) for x in r.samples]
println("  bout CV ", round.(cv; digits=2))
class, outcomes = classify(m; spec, Tnoisy=40sum(spec.τ), Tfree=150sum(spec.τ))
println("  class ", class, ": ", outcomes)
println("  pass on all three: ", r.pass && again.pass && fine.pass)

function put(file, header, rows)
    keep = isfile(file) ? filter(l -> !startswith(l, "$noise,"), readlines(file)[2:end]) : String[]
    open(file, "w") do io
        println(io, header)
        foreach(l -> println(io, l), keep)
        foreach(x -> println(io, join(x, ",")), rows)
    end
end
put(joinpath(OUT, "pilot.csv"), "noise,neurons,worst_err,rms,off,set1_pass,set2_pass,half_pass,third_pass,class,seconds",
    [(noise, L.M, maximum(abs, r.err), r.rms, r.off, r.pass, again.pass, fine.pass, third.pass, class, fitted.seconds)])
put(joinpath(OUT, "pilot_states.csv"), "noise,state,target,set1,set2,half_step,third,cv",
    [(noise, i, spec.τ[i], r.dwell[i], again.dwell[i], fine.dwell[i], third.dwell[i], cv[i]) for i in 1:3])
put(joinpath(OUT, "pilot_out.csv"), "noise,edge,out,seed,fast,contract",
    [(noise, e, fitted.out[e], fitted.seed === nothing ? NaN : fitted.seed[e], fitted.fast, fitted.contract) for e in eachindex(fitted.out)])
rm(runs_file; force=true)
