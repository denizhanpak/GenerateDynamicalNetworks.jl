# Phase 4 pre-check, before any design choice: are the two constructions of Ashwin & Postlethwaite
# (2013) implemented correctly? Three checks against things the paper or algebra fixes.
#   1. The shared-α Horchler design is a time-rescaled simplex network (algebra, checked on ρ).
#   2. Simplex: the decision graph of section 5.1, branch proportions against their Figure 12.
#   3. Cylinder: the Petersen graph with their parameters (9), and which edges actually occur.
# Run from notebooks/ with
#   julia --project=. -t 8 ../benchmarks/phase4/precheck.jl
include(joinpath(@__DIR__, "..", "phase3", "worm.jl"))
include(joinpath(@__DIR__, "..", "phase2", "cases.jl"))

# ── Simplex: dxⱼ = xⱼ(1 − |x|² + Σᵢ aᵢⱼ xᵢ²) + ζ dW, which is the u form with α = 1, ρⱼᵢ = 1 − aᵢⱼ ──

"""The simplex equation as a u-form model. `a[i, j]` is the eigenvalue at ξᵢ in the xⱼ direction."""
function simplex_model(P, a; ζ, s=1.0)
    n = size(a, 1)
    ρ = [s * (i == j ? 1.0 : 1.0 - a[j, i]) for i in 1:n, j in 1:n]
    return (; form=:u, P=float(P), visible=n, κ=0.1, β=ones(n), α=fill(s, n), ε=fill(float(ζ), n), ρ)
end

println("── 1. Horchler with shared α against simplex with speed α")
for (name, P, τ) in NAMED
    m = design(P, τ; form=:u, ε₀=1e-3, hidden=true)
    n = length(m.α)
    allequal(m.α) || (println("  $name: α not shared, skipped"); continue)
    a = [i == j ? 0.0 : m.P[i, j] > 0 ? 1 / m.ν[i] : -1.0 for i in 1:n, j in 1:n]
    println("  ", rpad(name, 22), "n = $n  max |ρ difference| = ", maximum(abs, simplex_model(m.P, a; ζ=1e-3, s=m.α[1]).ρ .- m.ρ))
end

println("\n── 2. Simplex, decision graph, section 5.1 parameters")
const DECISION = let A = zeros(8, 8)
    A[1, 2] = 2.0
    A[2, 3], A[2, 4] = 1.99, 1.85
    A[3, 5], A[3, 6] = 1.99, 1.85
    A[4, 7], A[4, 8] = 1.99, 1.85
    A[[5, 6, 7, 8], 1] .= 1.98
    A
end
decision_a = [i == j ? 0.0 : DECISION[i, j] > 0 ? DECISION[i, j] : -2.0 for i in 1:8, j in 1:8]
figure12 = Dict(1e-5 => -0.59, 1e-8 => -0.80)    # read by eye off their Figure 12, log₁₀ proportion
for ζ in (1e-5, 1e-8)
    t = @elapsed runs = simulate_frames(simplex_model(DECISION, decision_a; ζ); T=10_000.0, seeds=1:32, sample=0.05, dtscale=0.01)
    meas = measure(runs, 8)
    C = meas.counts
    props = [C[2, 4] / sum(C[2, :]), C[3, 6] / sum(C[3, :]), C[4, 8] / sum(C[4, :])]
    @printf("  ζ %.0e  %.0f s  visits to ξ₁ %d  failed %d  off-graph %.4f  mean dwell %s\n", ζ, t, sum(C[1, :]), count(r -> r.failed, runs),
            off_graph(meas, DECISION), join((@sprintf("%.2f", d) for d in meas.dwell), " "))
    @printf("     log₁₀ proportion: ξ₄ %.3f  ξ₆ %.3f  ξ₈ %.3f   (their Figure 12, read by eye: about %.2f)\n",
            log10(props[1]), log10(props[2]), log10(props[3]), figure12[ζ])
end

# ── Cylinder: dyⱼ = −yⱼGⱼ(y, p) + ζ dW, dp = −sin 2πp + F(y, p) + ζ dW ──────────────────────────

struct Edges{N}
    from::NTuple{N,Float64}
    to::NTuple{N,Float64}
end

function drift_cylinder(z, q, t)
    N = length(z) - 1
    p = z[N+1]
    y2 = sum(i -> z[i]^2, 1:N)
    bell(L, k, d) = L * sech(k * d)^2
    dy = ntuple(N) do j
        G = (z[j]^2 - 5 / 4)^2 - 1 - bell(q.Lα, q.kα, p - q.E.from[j]) + bell(q.Lω, q.kω, p - q.E.to[j]) + q.K * (y2 - z[j]^2)
        -z[j] * G
    end
    F = sum(j -> z[j]^2 * tanh(q.E.to[j] - p), 1:N)
    return SVector{N + 1}(dy..., -sinpi(2p) + F)
end
noise_cylinder(z, q, t) = q.ζ * ones(typeof(z))

"""A frame is vertex k only if p is within 0.25 of k and every |y| is under 0.5; otherwise the last vertex holds."""
function read_cylinder!(ig)
    ig.iter % ig.p.every == 0 || return nothing
    z, N = ig.u, length(ig.u) - 1
    if !(maximum(abs, z) < 100)
        ig.p.V.failed = true
        terminate!(ig)
        return nothing
    end
    k = round(Int, z[N+1])
    if abs(z[N+1] - k) < 0.25 && maximum(i -> abs(z[i]), 1:N) < 0.5 && (isempty(ig.p.V.state) || last(ig.p.V.state) != k)
        push!(ig.p.V.state, k)
        push!(ig.p.V.t, ig.t)
    end
    return nothing
end

function run_cylinder(A; Lα, Lω, kα, kω, ζ, T, seeds, K=1.0, sample=0.05, dt=0.01)
    edges = [(i, j) for i in axes(A, 1) for j in axes(A, 2) if A[i, j] > 0]
    N = length(edges)
    E = Edges{N}(ntuple(k -> float(edges[k][1]), N), ntuple(k -> float(edges[k][2]), N))
    z0 = SVector{N + 1}(ntuple(k -> k <= N ? ζ : 1.0, N + 1))
    q = (; E, Lα, Lω, kα, kω, K, ζ, every=round(Int, sample / dt), V=Frames())
    prob = SDEProblem(drift_cylinder, noise_cylinder, z0, (0.0, T), q)
    ens = EnsembleProblem(prob;
        prob_func=(pr, ctx) -> remake(pr; seed=UInt64(seeds[ctx.sim_id]), p=(; pr.p..., V=Frames())),
        output_func=(sol, ctx) -> ((state=sol.prob.p.V.state, t=sol.prob.p.V.t, failed=sol.prob.p.V.failed), false))
    cb = DiscreteCallback((u, t, ig) -> true, read_cylinder!; save_positions=(false, false))
    return solve(ens, EM(), EnsembleThreads(); trajectories=length(seeds), merge(quiet(dt), (; callback=cb))...).u
end

println("\n── 3. Cylinder, Petersen graph (undirected, so 30 directed edges and 31 dimensions)")
const PETERSEN = let und = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10), (6, 8), (8, 10), (10, 7), (7, 9), (9, 6)]
    A = zeros(10, 10)
    for (a, b) in und
        A[a, b] = A[b, a] = 1.0
    end
    A
end
eigen_expand(d; Lα, Lω, kω) = -9 / 16 + Lα - Lω * sech(kω * d)^2
for (label, kω) in (("paper, kω = 0.4705", 0.4705), ("narrower, kω = 2.017", 2.017))
    params = (; Lα=1.4376, Lω=1.5625, kα=2.017, kω)
    println("  $label: expanding eigenvalue for an edge of length 1, 2, 4, 5 along p: ",
            [round(eigen_expand(d; Lα=params.Lα, Lω=params.Lω, kω); digits=3) for d in (1, 2, 4, 5)])
    t = @elapsed runs = run_cylinder(PETERSEN; params..., ζ=1e-6, T=3000.0, seeds=1:16)
    meas = measure(runs, 10)
    C = meas.counts
    used = count(i -> PETERSEN[i] > 0 && C[i] > 0, eachindex(C))
    @printf("    %.0f s  failed %d  transitions %d  off-graph %.4f  edges used %d of %d  mean dwell %s\n", t, count(r -> r.failed, runs), sum(C),
            off_graph(meas, PETERSEN), used, count(>(0), PETERSEN), join((@sprintf("%.1f", d) for d in meas.dwell), " "))
    println("    unused edges: ", [(i, j) for i in 1:10 for j in 1:10 if PETERSEN[i, j] > 0 && C[i, j] == 0])
end
