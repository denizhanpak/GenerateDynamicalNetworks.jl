# Which simulator? All run the same worm ring at noise 1e-3 and read dwell with the same rule.
include("common.jl")
using StochasticDiffEq, DelimitedFiles
const ε = 1e-3
m = ring_model(WORM.τ; ε)
p = lv_static(m)
dt0 = 0.01 / m.α[1]
x0 = SVector{4}([0.9, ε, ε, ε])
T = 800.0
nsteps = round(Int, T / dt0)
println("threads=", Threads.nthreads(), " α=", round(m.α[1]; digits=2), " dt=", dt0, " steps=", nsteps)

rows = Any[]
function record(name, f; note="")
    f()                                     # compile
    GC.gc()
    r = @timed f()
    d = r.value
    @printf("%-24s %7.2f s %8.1f ns/step %9.1f MB  dwell %s  %s\n", name, r.time, 1e9r.time / nsteps, r.bytes / 1e6,
        join(round.(d; digits=3), " "), note)
    push!(rows, (name, r.time, 1e9r.time / nsteps, r.bytes / 1e6, d...))
    return d
end
offline(sol_t, X) = begin
    state, t = Int[], Float64[]
    for k in axes(X, 2)
        v, i = findmax(view(X, :, k))
        if v > 0.5 && (isempty(state) || last(state) != i)
            push!(state, i); push!(t, sol_t[k])
        end
    end
    s, e = state[2:end], t[2:end]
    from, d = s[1:end-1], diff(e)
    [mean(d[from .== i]) for i in 1:4]
end

# 1. Baseline: the SHCTools path Phase 1 used, saving a trajectory and decoding it afterwards.
record("lv_integrate + decode", () -> begin
    every = max(1, round(Int, minimum(m.τ) / 200 / dt0))
    sol = lv_integrate(0.0:dt0:T, Vector(x0), m.ρ, m.α, m.ε; every, seed=1)
    offline(sol.t, Array(sol))
end)

# 2. StochasticDiffEq, static arrays, clamping and tallying in a callback, saving nothing.
record("StochasticDiffEq SVector", () -> begin
    f(u, q, t) = lv_drift(u, q)
    g(u, q, t) = q.ε
    tally = Tally(4)
    st = Ref((0, 0, 0.0))
    function affect!(ig)
        ig.u = max.(ig.u, 0.0)
        v, i = findmax(ig.u)
        s, n, e = st[]
        if v > 0.5 && i != s
            n += 1
            n >= 3 && (tally.sum[s] += ig.t - e; tally.count[s] += 1)
            st[] = (i, n, ig.t)
        end
        SciMLBase.u_modified!(ig, false)
    end
    cb = DiscreteCallback((u, t, ig) -> true, affect!; save_positions=(false, false))
    prob = SDEProblem(f, g, x0, (0.0, T), p)
    solve(prob, EM(); dt=dt0, adaptive=false, save_everystep=false, save_start=false, save_end=false,
        callback=cb, seed=1, dense=false)
    tally.sum ./ tally.count
end)

# 3. The generic hand-written kernel.
record("bare loop, read each step", () -> begin
    t = em_tally(lv_drift, identity_readout, x0, p, p.ε, dt0, nsteps; seed=1)
    t.sum ./ t.count
end)
record("bare loop, read every 10", () -> begin
    t = em_tally(lv_drift, identity_readout, x0, p, p.ε, dt0, nsteps; seed=1, stride=10)
    t.sum ./ t.count
end)

# Do they agree? Same rule, many more visits than the timing runs.
println("\nAgreement, 16 seeds of T=800 each")
ref = pooled(on_threads(s -> em_tally(lv_drift, identity_readout, x0, p, p.ε, dt0, nsteps; seed=s), 1:16, 16))
println("bare loop      ", round.(ref.mean; digits=4), " ± ", round.(ref.sem; digits=4), " n=", ref.n)
base = on_threads(1:16, 16) do s
    every = max(1, round(Int, minimum(m.τ) / 200 / dt0))
    sol = lv_integrate(0.0:dt0:T, Vector(x0), m.ρ, m.α, m.ε; every, seed=100 + s)
    offline(sol.t, Array(sol))
end
b = reduce(hcat, base)
println("lv_integrate   ", round.(vec(mean(b; dims=2)); digits=4), " ± ", round.(vec(std(b; dims=2)) ./ 4; digits=4))
println("difference/se  ", round.((vec(mean(b; dims=2)) .- ref.mean) ./ sqrt.(ref.sem .^ 2 .+ (vec(std(b; dims=2)) ./ 4) .^ 2); digits=2))

# Thread scaling of the kernel on a fixed amount of work.
println("\nThread scaling, 64 runs of T=100")
scaling = map([1, 2, 4, 8, 12, 16]) do nt
    n100 = round(Int, 100 / dt0)
    on_threads(s -> em_tally(lv_drift, identity_readout, x0, p, p.ε, dt0, n100; seed=s), 1:64, nt)
    t = @elapsed on_threads(s -> em_tally(lv_drift, identity_readout, x0, p, p.ε, dt0, n100; seed=s), 1:64, nt)
    (nt, t)
end
for (nt, t) in scaling
    @printf("threads %2d  %6.2f s  speedup %.2f\n", nt, t, first(scaling)[2] / t)
end

# How coarse can dt go? 64 runs of T=250, so about 2000 cycles at every dt.
println("\nStep size")
dts = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16]
sweep = map(dts) do c
    dt = c / m.α[1]
    n = round(Int, 250 / dt)
    t = @elapsed r = pooled(on_threads(s -> em_tally(lv_drift, identity_readout, x0, p, p.ε, dt, n; seed=s), 1:64, 16))
    (; c, dt, t, mean=r.mean, sem=r.sem, n=r.n)
end
fine = first(sweep)
for r in sweep
    @printf("dt=%.4f/α (%.2e)  %6.2f s  dwell %s  shift%% %s  visits %d\n", r.c, r.dt, r.t, join(round.(r.mean; digits=4), " "),
        join(round.(100 .* (r.mean ./ fine.mean .- 1); digits=2), " "), minimum(r.n))
end

open("simulators.csv", "w") do io
    println(io, "name,seconds,ns_per_step,mb,d1,d2,d3,d4")
    for r in rows; println(io, "\"", r[1], "\",", join(r[2:end], ",")); end
end
open("thread_scaling.csv", "w") do io
    println(io, "threads,seconds")
    for r in scaling; println(io, join(r, ",")); end
end
open("step_size.csv", "w") do io
    println(io, "c,dt,seconds,d1,d2,d3,d4,s1,s2,s3,s4")
    for r in sweep; println(io, join([r.c, r.dt, r.t, r.mean..., r.sem...], ",")); end
end
