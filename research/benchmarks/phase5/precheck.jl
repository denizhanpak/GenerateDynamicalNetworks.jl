# Phase 5 pre-check, before any design choice.
#   1. Two-population network (Ashwin & Postlethwaite 2016, appendix A): the 3-ring at ν = ∓0.01.
#   2. CTRNN excitable network (Ashwin & Postlethwaite 2021): the default weights on a 4-ring.
#   3. What a classifier would measure on models whose class is already known.
# Run from notebooks/ with
#   julia --project=. -t 16 ../benchmarks/phase5/precheck.jl
include(joinpath(@__DIR__, "..", "phase3", "worm.jl"))
using DelimitedFiles

const EDGE3 = [(1, 2), (2, 3), (3, 1)]

# ── Two populations: p per vertex, y per edge. G(y, λ) = −y[(y² − 1)² + λ] ────────────────────

function drift_two(z::SVector{M}, q, t) where {M}
    nv, ne = q.nv, M - q.nv
    p2, y2 = sum(j -> z[j]^2, 1:nv), sum(k -> z[nv+k]^2, 1:ne)
    p4 = p2^2
    dp = ntuple(Val(nv)) do j
        out = sum(k -> q.from[k] == j ? z[nv+k]^2 * z[j] * z[q.to[k]] : 0.0, 1:ne)
        inn = sum(k -> q.to[k] == j ? z[nv+k]^2 * z[q.from[k]]^2 : 0.0, 1:ne)
        z[j] * (q.F * (1 - p2) + q.D * (z[j]^2 * p2 - p4)) + q.E * (inn - out)
    end
    dy = ntuple(Val(ne)) do k
        y = z[nv+k]
        λ = q.A - q.B * z[q.from[k]]^2 + q.C * (y2 - y^2)
        -y * ((y^2 - 1)^2 + λ)
    end
    return SVector{M}(dp..., dy...)
end
noise_two(z::SVector{M}, q, t) where {M} = SVector{M}(ntuple(i -> i <= q.nv ? q.ηp : q.ηy, Val(M)))

two_params(ν; ηp=1e-3, ηy=1e-3) = (; nv=3, from=first.(EDGE3), to=last.(EDGE3), A=0.5, B=1.5 - ν, C=2.0, D=10.0, E=4.0, F=2.0, ηp, ηy)
two_start(ν) = SVector{6}(1.0, 0, 0, 1e-3, 1e-3, 1e-3)

# ── CTRNN excitable network: ẏᵢ = −yᵢ + Σⱼ wᵢⱼ φ(yⱼ) + σ dW, φ(y) = 1/(1 + exp(−(y − θ)/ε)) ─────

function ctrnn2021(P; ε=0.05, θ=0.5, ws=1.0, wm=-0.7, wp=0.3, wt=0.0)
    n = size(P, 1)
    W = [i == j ? ws : P[j, i] > 0 ? wp : P[i, j] > 0 ? wm : wt for i in 1:n, j in 1:n]
    return (; W=SMatrix{n,n}(W), g=1 / ε, b=fill(-θ, n) |> SVector{n}, τ=ones(n) |> SVector{n})
end
drift_ctrnn(y, q, t) = (-y .+ q.W * (1 ./ (1 .+ exp.(-q.g .* (y .+ q.b))))) ./ q.τ
noise_flat(z, q, t) = q.η * ones(typeof(z))

# ── Generic measurements ────────────────────────────────────────────────────────────────────

function run_path(f, g, q, u0, T, dt, saveat, seed)
    sol = solve(SDEProblem(f, g, u0, (0.0, T), q; seed=UInt64(seed)), EM(); dt, adaptive=false, saveat, maxiters=typemax(Int))
    return sol.t, sol.u
end
zero_noise(z, q, t) = zero(z)

function bouts_of(t, u, decode)
    s, st = Int[], Float64[]
    for (a, b) in zip(t, u)
        k = decode(b)
        (isempty(s) || last(s) != k) && (push!(s, k); push!(st, a))
    end
    return s, diff([st; t[end]])
end

function jac(f, q, u)
    n = length(u)
    h = 1e-6
    reduce(hcat, [Vector(f(u .+ h * unit(n, i, u), q, 0.0) .- f(u .- h * unit(n, i, u), q, 0.0)) / 2h for i in 1:n])
end
unit(n, i, u) = typeof(u)(ntuple(k -> k == i ? 1.0 : 0.0, n))

"""Newton from many points along the run; equilibria found, with how many unstable directions each has."""
function equilibria_near(f, q, u)
    found = Vector{Float64}[]
    unstable = Int[]
    for start in u[round.(Int, range(1, length(u); length=min(300, length(u))))]
        x = start
        ok = false
        for _ in 1:60
            F = f(x, q, 0.0)
            maximum(abs, F) < 1e-10 && (ok = true; break)
            J = jac(f, q, x)
            d = try J \ Vector(F) catch; break end
            x = x .- typeof(x)(Tuple(clamp.(d, -1.0, 1.0)))
            maximum(abs, x) < 1e3 || break
        end
        ok || continue
        any(e -> maximum(abs, e .- Vector(x)) < 1e-5, found) && continue
        push!(found, Vector(x))
        push!(unstable, count(>(1e-7), real.(eigvals(jac(f, q, x)))))
    end
    return found, unstable
end

function report(name, f, q, u0, decode; T, dt, known)
    t, u = run_path(f, zero_noise, q, u0, T, dt, max(dt, T / 40_000), 1)
    s, d = bouts_of(t, u, decode)
    speed = [norm(f(x, q, 0.0)) for x in u]
    tail = speed[end÷2:end]
    found, unstable = equilibria_near(f, q, u)
    late = max(1, length(d) - 8):length(d)
    nearest = isempty(found) ? NaN : minimum(minimum(maximum(abs, Vector(x) .- e) for e in found) for x in u[end÷2:end][argmin(tail):argmin(tail)])
    @printf("%-34s known %-12s bouts %4d  last durations %s\n", name, known, length(s), join((@sprintf("%.3g", x) for x in d[late]), " "))
    @printf("%-34s final speed %.2e  median/min speed (2nd half) %.3g  equilibria %d (unstable dims %s)  slowest point to nearest equilibrium %.3g\n",
            "", speed[end], median(tail) / minimum(tail), length(found), sort(unstable), nearest)
    flush(stdout)
end

# ── Run ─────────────────────────────────────────────────────────────────────────────────────

println("── 1. Two-population 3-ring, noise on y (ηp = 1e-3), argmax of |p|, 16 runs of 20000")
decode_two(z) = argmax(abs.(z[SVector(1, 2, 3)]))
for ν in (-0.01, 0.01), ηy in (1e-2, 1e-3, 1e-4)
    q = two_params(ν; ηy)
    ens = EnsembleProblem(SDEProblem(drift_two, noise_two, two_start(ν), (0.0, 20_000.0), q);
        prob_func=(pr, ctx) -> remake(pr; seed=UInt64(ctx.sim_id)), output_func=(sol, ctx) -> (sol, false))
    t = @elapsed sims = solve(ens, EM(), EnsembleThreads(); trajectories=16, dt=0.01, adaptive=false, saveat=1.0, maxiters=typemax(Int))
    rs = [bouts_of(s.t, s.u, decode_two) for s in sims.u]
    d = vcat([x[2][2:end-1] for x in rs if length(x[2]) > 2]...)
    trans = sum(length(x[1]) - 1 for x in rs)
    @printf("  ν %+.2f  ηy %.0e  %.0f s  transitions %6d  mean residence %s  CV %s\n", ν, ηy, t, trans, isempty(d) ? "–" : @sprintf("%.1f", mean(d)), isempty(d) ? "–" : @sprintf("%.2f", std(d) / mean(d)))
    flush(stdout)
end

println("\n── 2. CTRNN excitable network, 2021 default weights, 4-ring, argmax of y, 16 runs of 5000")
const RING4 = Float64[0 1 0 0; 0 0 1 0; 0 0 0 1; 1 0 0 0]
base = ctrnn2021(RING4)
y0 = SVector{4}(1.0, 0.3, 0.0, -0.7)
for η in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)
    q = merge(base, (; η))
    ens = EnsembleProblem(SDEProblem(drift_ctrnn, noise_flat, y0, (0.0, 5000.0), q);
        prob_func=(pr, ctx) -> remake(pr; seed=UInt64(ctx.sim_id)), output_func=(sol, ctx) -> (sol, false))
    sims = solve(ens, EM(), EnsembleThreads(); trajectories=16, dt=0.01, adaptive=false, saveat=0.1, maxiters=typemax(Int))
    rs = [bouts_of(s.t, s.u, y -> argmax(y)) for s in sims.u]
    C = zeros(Int, 4, 4)
    for r in rs, k in 1:length(r[1])-1
        C[r[1][k], r[1][k+1]] += 1
    end
    d = vcat([x[2][2:end-1] for x in rs if length(x[2]) > 2]...)
    @printf("  noise %.2f  transitions %6d  on the graph %s  mean residence %s\n", η, sum(C), sum(C) == 0 ? "–" : @sprintf("%.3f", sum(C .* (RING4 .> 0)) / sum(C)), isempty(d) ? "–" : @sprintf("%.1f", mean(d)))
    flush(stdout)
end

println("\n── 3. What a classifier sees on models of known class (noise-free runs)")
states2(y, net) = (on = y .+ net.b .> 0; on[1] ? (on[2] ? 4 : 1) : (on[2] ? 3 : 2))
genome = vec(readdlm(joinpath(@__DIR__, "..", "..", "data", "CTRNN2NeuronOpt.txt"))[9, 1:8])
two_node(g) = (; W=SMatrix{2,2}(reshape(g[1:4], 2, 2)), g=1.0, b=SVector{2}(g[5:6]), τ=exp.(SVector{2}(g[7:8])))
net2 = two_node(genome)
beer = (; W=SMatrix{2,2}(4.5, -1.0, 1.0, 4.5), g=1.0, b=SVector{2}(-2.75, -1.75), τ=SVector{2}(fill(sum(WORM_SPEC.τ) / 29, 2)...))
ring3(a, p, n) = SMatrix{3,3}([i == j ? a : j == mod1(i - 1, 3) ? p : n for i in 1:3, j in 1:3])
rps(p, n) = (; W=ring3(1.0, p, n), g=1.0, b=SVector{3}(0.0, 0, 0), τ=SVector{3}(1.0, 1, 1))
theta(z, q, t) = SVector{2}(z[1] * (1 - z[1]^2 - z[2]^2) - z[2] * (1 - z[1] + q.r), z[2] * (1 - z[1]^2 - z[2]^2) + z[1] * (1 - z[1] + q.r))
θ_of = vec(readdlm(joinpath(@__DIR__, "results", "..", "..", "phase4", "results", "theta_simplex_worm_ring_0.001.csv"), ','))
α, e, c = exp(θ_of[1]), exp.(θ_of[2:5]), exp.(θ_of[6:9])
a = [i == j ? 0.0 : -c[i] for i in 1:4, j in 1:4]
for (k, (i, j)) in enumerate([(1, 2), (2, 3), (3, 4), (4, 1)])
    a[i, j] = e[k]
end
simplex = (; α=SVector{4}(fill(α, 4)...), ρ=SMatrix{4,4}([α * (i == j ? 1.0 : 1.0 - a[j, i]) for i in 1:4, j in 1:4]))
argm(z) = argmax(abs.(z))
user = (; α=SVector{4}(USER_FIT.α), ρ=SMatrix{4,4}(USER_FIT.ρ))
u0(n) = SVector{n}(ntuple(i -> i == 1 ? 0.95 : 1e-3, n))
report("your worm fit (u form)", drift_u, user, u0(4), argm; T=1500.0, dt=0.02 / maximum(user.α), known="heteroclinic")
report("simplex worm, noise-1e-3 fit", drift_u, simplex, u0(4), argm; T=300.0, dt=0.02 / α, known="heteroclinic")
report("two-population, ν = -0.01", drift_two, two_params(-0.01), two_start(-0.01), decode_two; T=8000.0, dt=0.01, known="heteroclinic")
report("two-population, ν = +0.01", drift_two, two_params(0.01), two_start(0.01), decode_two; T=8000.0, dt=0.01, known="excitable")
report("CTRNN 2021, 4-ring", drift_ctrnn, base, y0, argm; T=2000.0, dt=0.01, known="excitable")
report("your 2-node CTRNN (row 9)", drift_ctrnn, net2, SVector{2}(log(0.9 / 0.1), log(0.1 / 0.9)) .- net2.b, y -> states2(y, net2); T=300.0, dt=0.002, known="limit cycle")
report("Beer oscillator", drift_ctrnn, beer, SVector{2}(log(0.9 / 0.1), log(0.1 / 0.9)) .- beer.b, y -> states2(y, beer); T=100.0, dt=0.001, known="limit cycle")
report("your ring3 (-5, -14) haunted", drift_ctrnn, rps(-5.0, -14.0), SVector{3}(0.9, -4.0, -8.0), argm; T=400.0, dt=0.005, known="ghost")
report("your ring3 (-6.9, -15)", drift_ctrnn, rps(-6.9, -15.0), SVector{3}(0.9, -4.0, -8.0), argm; T=400.0, dt=0.005, known="mixed")
report("theta neuron, r = 1e-3", theta, (; r=1e-3), SVector{2}(0.0, 1.0), z -> z[1] > 0.5 ? 1 : 2; T=2000.0, dt=0.005, known="ghost")
report("theta neuron, r = 1.0", theta, (; r=1.0), SVector{2}(0.0, 1.0), z -> z[1] > 0.5 ? 1 : 2; T=200.0, dt=0.005, known="limit cycle")
