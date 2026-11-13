# StochasticDiffEq set up for search: nothing saved, clamp and dwell read in a callback,
# runs in parallel through EnsembleProblem. Compared with the hand-written loop in common.jl.
include("common.jl")
using StochasticDiffEq, DelimitedFiles
const ε = 1e-3
m = ring_model(WORM.τ; ε); p = lv_static(m)
dt0 = 0.01 / m.α[1]; x0 = SVector{4}([0.9, ε, ε, ε]); T = 800.0; nsteps = round(Int, T / dt0)
f(u, q, t) = lv_drift(u, q)
g(u, q, t) = q.ε
quiet = (dt=dt0, adaptive=false, save_everystep=false, save_start=false, save_end=false, dense=false)

"""Dwell read as the run goes: the `latch(0.5)` rule, first visit dropped."""
mutable struct Latch; s::Int; n::Int; e::Float64; sum::Vector{Float64}; count::Vector{Int}; end
Latch() = Latch(0, 0, 0.0, zeros(4), zeros(Int, 4))
function see!(L, u, t)
    v, i = findmax(u)
    if v > 0.5 && i != L.s
        L.n += 1
        L.n >= 3 && (L.sum[L.s] += t - L.e; L.count[L.s] += 1)
        L.s, L.e = i, t
    end
end
clamp!(ig) = (ig.u = max.(ig.u, 0.0); nothing)
clamp_cb = DiscreteCallback((u, t, ig) -> true, clamp!; save_positions=(false, false))
function dwell_run(seed)
    L = Latch()
    cb = DiscreteCallback((u, t, ig) -> true, ig -> (clamp!(ig); see!(L, ig.u, ig.t)); save_positions=(false, false))
    solve(SDEProblem(f, g, x0, (0.0, T), p), EM(); callback=cb, seed, quiet...)
    L.sum ./ L.count
end

rows = Any[]
function record(name, fn)
    fn(); GC.gc(); r = @timed fn()
    d = r.value isa Vector ? r.value : fill(NaN, 4)
    @printf("%-34s %6.1f ns/step %8.1f MB  dwell %s\n", name, 1e9r.time / nsteps, r.bytes / 1e6, join(round.(d; digits=3), " "))
    push!(rows, (name, 1e9r.time / nsteps, r.bytes / 1e6, d...))
end
record("hand-written loop", () -> (t = em_tally(lv_drift, identity_readout, x0, p, p.ε, dt0, nsteps; seed=1); t.sum ./ t.count))
record("EM, no clamp, no reading", () -> solve(SDEProblem(f, g, x0, (0.0, T), p), EM(); seed=1, quiet...))
record("EM, clamp callback", () -> solve(SDEProblem(f, g, x0, (0.0, T), p), EM(); callback=clamp_cb, seed=1, quiet...))
record("EM, clamp and dwell callback", () -> dwell_run(1))
open("library_integrator.csv", "w") do io
    println(io, "name,ns_per_step,mb,d1,d2,d3,d4")
    for r in rows; println(io, "\"", r[1], "\",", join(r[2:end], ",")); end
end

# Threads: 64 runs of T = 200 through EnsembleProblem.
short = SDEProblem(f, g, x0, (0.0, 200.0), p)
ens = EnsembleProblem(short; prob_func=(pr, ctx) -> remake(pr; seed=UInt64(ctx.sim_id)), output_func=(sol, ctx) -> (sol.u[end], false))
go(alg) = solve(ens, EM(), alg; trajectories=64, callback=clamp_cb, quiet..., save_end=true)
go(EnsembleSerial()); serial = @elapsed go(EnsembleSerial())
go(EnsembleThreads()); threaded = @elapsed go(EnsembleThreads())
@printf("EnsembleThreads on %d threads: %.2f s against %.2f s serial, %.1f×\n", Threads.nthreads(), threaded, serial, serial / threaded)
writedlm("library_ensemble.csv", ["threads" "serial_s" "threaded_s"; Threads.nthreads() serial threaded], ',')
