# Can DiffEqGPU's kernel run this on the CPU, including the clamp at zero?
include("common.jl")
using DiffEqGPU, StochasticDiffEq, KernelAbstractions
m = ring_model(WORM.τ; ε=1e-3); p = lv_static(m)
dt = 0.01 / m.α[1]; x0 = SVector{4}([0.9, 1e-3, 1e-3, 1e-3])
f(u, q, t) = u .* (q.α - q.ρ * u)
g(u, q, t) = q.ε
prob = SDEProblem(f, g, x0, (0.0, 20.0), p)
ens = EnsembleProblem(prob)
for (name, kw) in (("no clamp", (;)),
    ("clamp callback", (; callback=DiscreteCallback((u, t, i) -> true, i -> (i.u = max.(i.u, 0.0); nothing); save_positions=(false, false)))))
    try
        solve(ens, GPUEM(), EnsembleGPUKernel(CPU()); trajectories=16, dt, adaptive=false, save_everystep=false, kw...)
        t = @elapsed sol = solve(ens, GPUEM(), EnsembleGPUKernel(CPU()); trajectories=16, dt, adaptive=false, save_everystep=false, kw...)
        steps = 16 * round(Int, 20.0 / dt)
        println(name, ": ok, ", round(1e9t / steps; digits=1), " ns/step across threads, min final x ", round(minimum(minimum(s.u[end]) for s in sol.u); sigdigits=3))
    catch e
        println(name, ": FAILED ", first(replace(sprint(showerror, e), "\n" => " "), 300))
    end
end
