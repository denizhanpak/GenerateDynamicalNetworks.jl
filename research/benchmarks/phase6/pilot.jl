# The Phase 6 pilot: a 3-state ring with dwells 1, 3 and 10 on 4 phase oscillators (2 pairs), at one
# noise level. First a check that every corner's Jacobian has exactly the designed rates, then the fit,
# the acceptance (held-out sets 1 and 2, and set 1 at half the step), a third set, bout CV and class.
# Run from notebooks/ with
#   julia --project=. -t 7 ../benchmarks/phase6/pilot.jl <noise>
include(joinpath(@__DIR__, "synchrony.jl"))

const OUT = joinpath(@__DIR__, "results")
mkpath(OUT)
noise = parse(Float64, ARGS[1])
spec = (; P=float(ring(3)), τ=[1.0, 3.0, 10.0], form=:u, ε₀=noise, sample=0.1)
L = ring3_layout(spec)
println("3-ring at noise $noise: states on corners ", [corner_name(v, L.M) for v in L.corner_of],
        ", paths ", [join(corner_name.(p, L.M), "→") for p in L.paths])

# Every corner is an equilibrium whose Jacobian has eigenvalues Λ[:, v] and one zero per pair (its common phase).
let m = sync_model(spec, L, [1.0, 2.0, 3.0]; fast=5.0, contract=7.0), q = sync_params(m)
    worst = 0.0
    for v in 1:2^L.M
        θ = SVector{2L.M}(ntuple(i -> iseven(i) && corner_bit(v, i ÷ 2) == 1 ? π : 0.0, 2L.M))
        ev = sort(real.(eigvals(jac(drift_sync, q, θ))))
        want = sort([m.Λ[:, v]; zeros(L.M)])
        worst = max(worst, maximum(abs, ev .- want), maximum(abs, drift_sync(θ, q, 0.0)))
    end
    @printf("corner check: largest gap between Jacobian eigenvalues and design, or drift at a corner, %.2e\n", worst)
end

t = @elapsed f = fit_sync_pilot(spec, L)
m = f.model
writedlm(joinpath(OUT, "pilot_out_$noise.csv"), f.out, ',')
@printf("fitted out-rates %s, transit %.3g, contraction %.3g, %.0f s\n", round.(f.out; sigdigits=4), f.fast, f.contract, t)
T = 250sum(spec.τ)
r = validate(m, spec; T)
again = validate(m, spec; T, seeds=9_200_001:9_200_064)
fine = validate(m, spec; T, dtscale=0.005)
third = validate(m, spec; T, seeds=9_400_001:9_400_064)
show_result("pilot ε=$noise, held-out set 1", r)
show_result("pilot ε=$noise, held-out set 2", again)
show_result("pilot ε=$noise, set 1 at half the step", fine)
show_result("pilot ε=$noise, third set (information)", third)
cv = [std(x) / mean(x) for x in measure(simulate_frames(m; T, seeds=9_000_001:9_000_016, spec.sample, dtscale=0.01), 3).samples]
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
put(joinpath(OUT, "pilot.csv"), "noise,worst_err,rms,off,set1_pass,set2_pass,half_pass,third_pass,class,seconds",
    [(noise, maximum(abs, r.err), r.rms, r.off, r.pass, again.pass, fine.pass, third.pass, class, t)])
put(joinpath(OUT, "pilot_states.csv"), "noise,state,target,set1,set2,half_step,third,cv",
    [(noise, i, spec.τ[i], r.dwell[i], again.dwell[i], fine.dwell[i], third.dwell[i], cv[i]) for i in 1:3])
