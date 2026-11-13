include("common.jl")
using StochasticDiffEq
const ε = 1e-3
m = ring_model(WORM.τ; ε); p = lv_static(m)
dt0 = 0.01 / m.α[1]; x0 = SVector{4}([0.9, ε, ε, ε])

# 1. Path check: same seed, does the bare loop reproduce lv_integrate exactly?
println("PATH CHECK")
n = 20000
sol = lv_integrate(0.0:dt0:n*dt0, Vector(x0), m.ρ, m.α, m.ε; every=1, seed=1)
rng = Xoshiro(1); x = x0; s = p.ε .* sqrt(dt0)
states = [x]
for k in 1:n
    global x = max.(x + dt0 * lv_drift(x, p) + s .* randn(rng, SVector{4,Float64}), 0.0)
    push!(states, x)
end
for k in (1, 2, 3, 100, min(length(sol.u), 20000))
    println("step ", k, "  lv ", round.(sol.u[k]; sigdigits=7), "  bare ", round.(Vector(states[k]); sigdigits=7))
end
println("saved points ", length(sol.u), "; max abs diff over the first 20: ", maximum(maximum(abs.(sol.u[k] .- states[k])) for k in 1:20))
d(s; T=800.0) = em_tally(lv_drift, identity_readout, x0, p, p.ε, dt0, round(Int, T / dt0); seed=s)
t = d(1); println("bare dwell seed 1 to 6 digits: ", round.(t.sum ./ t.count; sigdigits=6))

# 2. Step size at other noise levels: 4x as many runs at each, since the shifts are small.
println("\nSTEP SIZE, 256 runs of T=250")
open("step_size_noise.csv", "w") do io
    println(io, "noise,c,seconds,d1,d2,d3,d4,s1,s2,s3,s4")
    for e in (1e-5, 1e-3, 1e-2)
        q = ring_model(WORM.τ; ε=e); ps = lv_static(q); y0 = SVector{4}([0.9, e, e, e])
        rows = map([0.0025, 0.01, 0.02, 0.04, 0.08, 0.16]) do c
            dt = c / q.α[1]; nn = round(Int, 250 / dt)
            tm = @elapsed r = pooled(on_threads(k -> em_tally(lv_drift, identity_readout, y0, ps, ps.ε, dt, nn; seed=k), 1:256, 16))
            (; c, tm, r)
        end
        f = first(rows).r
        for r in rows
            z = (r.r.mean .- f.mean) ./ sqrt.(r.r.sem .^ 2 .+ f.sem .^ 2)
            @printf("ε=%.0e dt=%.4f/α  %5.1f s  shift%% %s  z %s\n", e, r.c, r.tm,
                join(lpad.(round.(100 .* (r.r.mean ./ f.mean .- 1); digits=2), 6), " "), join(lpad.(round.(z; digits=1), 5), " "))
            println(io, join([e, r.c, r.tm, r.r.mean..., r.r.sem...], ","))
        end
    end
end

# 3. Size: static kernel against an in-place kernel on plain arrays, on rings of N states.
function em_plain(α, ρ, ε, x0, dt, nsteps; seed)
    rng = Xoshiro(seed); N = length(x0); x = copy(x0); dx = similar(x); z = similar(x); s = ε .* sqrt(dt); state = 0
    @inbounds for k in 1:nsteps
        mul!(dx, ρ, x)
        randn!(rng, z)
        @simd for i in 1:N
            x[i] = max(x[i] + dt * x[i] * (α[i] - dx[i]) + s[i] * z[i], 0.0)
        end
        v, i = findmax(x)
        v > 0.5 && i != state && (state = i)
    end
    return state
end
println("\nSIZE ns/step")
open("network_size.csv", "w") do io
    println(io, "N,static_ns,plain_ns")
    for N in (4, 8, 16, 32, 64)
        τ = fill(1.0, N); q = ring_model(τ; ε=1e-3); dt = 0.01 / q.α[1]; nn = 400_000
        pl() = em_plain(q.α, q.ρ, q.ε, [0.9; fill(1e-3, N - 1)], dt, nn; seed=1)
        pl(); tp = @elapsed pl()
        ts = NaN
        if N <= 32
            ps = lv_static(q); y0 = SVector{N}([0.9; fill(1e-3, N - 1)])
            st() = em_tally(lv_drift, identity_readout, y0, ps, ps.ε, dt, nn; seed=1)
            comp = @elapsed st()
            ts = @elapsed st()
            @printf("N=%3d static %8.1f plain %8.1f   (first call %.1f s)\n", N, 1e9ts / nn, 1e9tp / nn, comp)
        else
            @printf("N=%3d static   (not run) plain %8.1f\n", N, 1e9tp / nn)
        end
        println(io, "$N,$(1e9ts / nn),$(1e9tp / nn)")
    end
end
