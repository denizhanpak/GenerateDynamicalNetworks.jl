# The classifier, tested on models of known class and then applied to every fit from Phases 4 and 5.
# Rule agreed with the user on 2026-11-08: noise-free runs start from points along a noisy run.
#   settles on a stable equilibrium                → excitable
#   stalls at a saddle, or cycles that keep growing → heteroclinic
#   repeating cycle, peak/slowest speed < 100      → limit cycle
#   repeating cycle, peak/slowest speed ≥ 100      → ghost
#   starts that disagree, or nothing recognisable  → unknown
# Run from notebooks/ with
#   julia --project=. -t 16 ../benchmarks/phase5/classes.jl [stop_after_seconds]
include(joinpath(@__DIR__, "backends.jl"))
import CTRNNs

const OUT = joinpath(@__DIR__, "results")
const CLASSES = joinpath(OUT, "classes.csv")
const GHOST_RATIO = 100.0

zero_noise(z, q, t) = zero(z)
noise_eta(z, q, t) = q.η * ones(typeof(z))

function jac(f, q, u)
    h = 1e-6
    bump(i) = typeof(u)(ntuple(k -> k == i ? h : 0.0, length(u)))
    return reduce(hcat, [Vector(f(u .+ bump(i), q, 0.0) .- f(u .- bump(i), q, 0.0)) / 2h for i in eachindex(u)])
end

function find_cycle(s, d)
    n = length(s)
    for p in 1:n÷3
        all(s[n-i] == s[n-i-p] for i in 0:2p-1) || continue
        return p, sum(d[n-p+1:n]), sum(d[n-2p+1:n-p])
    end
    return nothing
end

"""One noise-free run: :stable, :saddle, :growing, :periodic (with its speed ratio) or :irregular."""
function outcome(f, q, z, decode; T, dt)
    sol = solve(SDEProblem(f, zero_noise, z, (0.0, T), q; seed=UInt64(1)), EM(); dt, adaptive=false, saveat=T / 40_000, maxiters=typemax(Int))
    v = [norm(f(x, q, 0.0)) for x in sol.u]
    if v[end] < 1e-7 * maximum(v)
        return count(>(1e-7), real.(eigvals(jac(f, q, sol.u[end])))) == 0 ? (:stable, NaN) : (:saddle, NaN)
    end
    s, st = Int[], Float64[]
    for (a, x) in zip(sol.t, sol.u)
        k = decode(x)
        k > 0 && (isempty(s) || last(s) != k) && (push!(s, k); push!(st, a))
    end
    length(s) < 8 && return (:irregular, NaN)
    d = diff(st)
    c = find_cycle(s[2:end-1], d[2:end])
    c === nothing && return (:irregular, NaN)
    _, last_cycle, before = c
    last_cycle / before > 1.2 && return (:growing, NaN)
    tail = v[end÷2:end]
    return (:periodic, maximum(tail) / minimum(tail))
end

function classify(f, q, z0, decode; noise, Tnoisy, Tfree, dt, starts=5)
    qn = merge(q, (; η=noise))
    sol = solve(SDEProblem(f, noise_eta, z0, (0.0, Tnoisy), qn; seed=UInt64(11)), EM(); dt, adaptive=false, saveat=Tnoisy / 2000, maxiters=typemax(Int))
    points = [[z0]; sol.u[round.(Int, range(length(sol.u) ÷ 2, length(sol.u); length=starts - 1))]]
    results = [outcome(f, qn, p, decode; T=Tfree, dt) for p in points]
    labels = map(results) do (o, r)
        o === :stable ? :excitable : o in (:saddle, :growing) ? :heteroclinic : o === :irregular ? :unknown : r >= GHOST_RATIO ? :ghost : :limit_cycle
    end
    class = allequal(labels) ? first(labels) : :unknown
    return class, join(("$(o)" * (isnan(r) ? "" : "($(round(r; sigdigits=3)))") for (o, r) in results), " ")
end

# ── Each model type as a vector field ───────────────────────────────────────────────────────

function classify(m::NamedTuple; spec, kw...)
    n = length(m.α)
    q = (; α=SVector{n}(m.α), ρ=SMatrix{n,n}(m.ρ))
    z0 = SVector{n}(ntuple(i -> i == 1 ? 0.95 : 1e-3, n))
    return classify(drift_u, q, z0, z -> argmax(abs.(z)); noise=spec.ε₀, dt=0.02 / maximum(m.α), kw...)
end
function classify(m::CylinderModel{N}; spec, kw...) where {N}
    q = (; s=m.α[1], m.Lα, m.from, m.to)
    decode(z) = (k = round(Int, z[end]); 1 <= k <= m.visible && abs(z[end] - k) < 0.25 && maximum(i -> abs(z[i]), 1:N) < 0.5 ? k : 0)
    return classify(drift_cylinder, q, SVector{N + 1}(ntuple(k -> k <= N ? 1e-3 : 1.0, N + 1)), decode; noise=spec.ε₀, dt=0.02 / m.α[1], kw...)
end
function classify(m::TwoPopModel; spec, kw...)
    return classify(drift_two, two_params(m), two_start(m), z -> findmax(i -> abs(z[i]), 1:m.visible)[2]; noise=spec.ε₀, dt=0.02 / m.α[1], kw...)
end
function classify(m::CTRNNModel; spec, kw...)
    return classify(drift_ctrnn, ctrnn_params(m), ctrnn_start(m), y -> argmax(i -> m.g * (y[i] + m.b[i]), 1:m.visible); noise=spec.ε₀, dt=0.02 / m.α[1], kw...)
end

# ── Known classes ───────────────────────────────────────────────────────────────────────────

function record(group, name, known, expected, class, outcomes)
    open(CLASSES, "a") do io
        filesize(CLASSES) == 0 && println(io, "group,name,known,expected,class,outcomes")
        println(io, join((group, "\"$name\"", known, expected, class, "\"$outcomes\""), ","))
    end
    @printf("%-9s %-46s expected %-13s got %-13s %s\n", group, name, expected, class, outcomes)
    flush(stdout)
end

already(group, name) = isfile(CLASSES) && any(l -> startswith(l, "$group,\"$name\""), readlines(CLASSES))

theta3(z, q, t) = SVector{2}(z[1] * (1 - z[1]^2 - z[2]^2) - z[2] * (1 - z[1] + q.r), z[2] * (1 - z[1]^2 - z[2]^2) + z[1] * (1 - z[1] + q.r))
function net(W, b, τ, g=1.0)
    n = length(b)
    return (; W=SMatrix{n,n}(W), b=SVector{n}(b), τ=SVector{n}(τ), g)
end
ring3(p, m) = [i == j ? 1.0 : j == mod1(i - 1, 3) ? p : m for i in 1:3, j in 1:3]
quadrant(b) = y -> (on = y .+ b .> 0; on[1] ? (on[2] ? 4 : 1) : (on[2] ? 3 : 2))

function fixtures()
    ring4 = Float64[0 1 0 0; 0 0 1 0; 0 0 0 1; 1 0 0 0]
    g9 = vec(readdlm(joinpath(@__DIR__, "..", "..", "data", "CTRNN2NeuronOpt.txt"))[9, 1:8])
    two9 = net(reshape(g9[1:4], 2, 2), g9[5:6], exp.(g9[7:8]))
    beer = net([4.5 1.0; -1.0 4.5], [-2.75, -1.75], fill(0.273, 2))
    spec0 = (; ε₀=1e-3)
    W2021 = [i == j ? 1.0 : ring4[j, i] > 0 ? 0.3 : ring4[i, j] > 0 ? -0.7 : 0.0 for i in 1:4, j in 1:4]
    start2(b) = SVector{2}(log(9.0), -log(9.0)) .- b
    tp(ν) = TwoPopModel{3}(Float64[0 1 0; 0 0 1; 1 0 0], 3, [2.5], 1.0, ntuple(_ -> 1.5 - ν, 3), (1, 2, 3), (2, 3, 1), 1e-3)
    return [
        ("your worm fit", "heteroclinic", "heteroclinic", () -> classify(USER_FIT; spec=spec0, Tnoisy=100.0, Tfree=1500.0)),
        ("two-population, ν = −0.01", "heteroclinic", "heteroclinic", () -> classify(tp(-0.01); spec=spec0, Tnoisy=4000.0, Tfree=8000.0)),
        ("two-population, ν = +0.01", "excitable", "excitable", () -> classify(tp(0.01); spec=spec0, Tnoisy=2000.0, Tfree=4000.0)),
        ("CTRNN 2021, 4-ring", "excitable", "excitable", () -> classify(drift_ctrnn, net(W2021, fill(-0.5, 4), ones(4), 20.0), SVector{4}(1.0, 0.3, 0.0, -0.7), y -> argmax(y); noise=0.02, Tnoisy=200.0, Tfree=2000.0, dt=0.01)),
        ("your 2-node CTRNN (row 9)", "limit cycle (notes)", "ghost", () -> classify(drift_ctrnn, two9, start2(two9.b), quadrant(two9.b); noise=1e-3, Tnoisy=60.0, Tfree=300.0, dt=0.002)),
        ("Beer oscillator", "limit cycle", "limit_cycle", () -> classify(drift_ctrnn, beer, start2(beer.b), quadrant(beer.b); noise=1e-3, Tnoisy=20.0, Tfree=100.0, dt=0.001)),
        ("your ring3 (−5, −14)", "haunted (notes)", "limit_cycle", () -> classify(drift_ctrnn, net(ring3(-5.0, -14.0), zeros(3), ones(3)), SVector{3}(0.9, -4.0, -8.0), y -> argmax(y); noise=1e-3, Tnoisy=100.0, Tfree=400.0, dt=0.005)),
        ("your ring3 (−6.9, −15)", "mixed (notes)", "unknown", () -> classify(drift_ctrnn, net(ring3(-6.9, -15.0), zeros(3), ones(3)), SVector{3}(0.9, -4.0, -8.0), y -> argmax(y); noise=0.1, Tnoisy=400.0, Tfree=400.0, dt=0.005)),
        ("theta neuron, r = 1e-3", "ghost", "ghost", () -> classify(theta3, (; r=1e-3), SVector{2}(0.0, 1.0), z -> z[1] > 0.5 ? 1 : 2; noise=1e-4, Tnoisy=400.0, Tfree=2000.0, dt=0.005)),
        ("theta neuron, r = 1", "limit cycle", "limit_cycle", () -> classify(theta3, (; r=1.0), SVector{2}(0.0, 1.0), z -> z[1] > 0.5 ? 1 : 2; noise=1e-4, Tnoisy=50.0, Tfree=200.0, dt=0.005)),
    ]
end

# ── Fits ────────────────────────────────────────────────────────────────────────────────────

function fit_spec(name, ε₀)
    _, P, τ = only(c for c in NAMED if c[1] == name)
    return (; P, τ, form=:u, ε₀, sample=name == "worm ring" ? WORM_SPEC.sample : minimum(τ) / 10)
end
san(n) = replace(n, r"[^A-Za-z0-9]+" => "_")

function fits()
    out = []
    for ε₀ in (1e-3, 0.1), (name, _, _) in NAMED, b in (:simplex, :cylinder)
        spec = fit_spec(name, ε₀)
        make = b === :simplex ? make_simplex : make_cylinder
        file = joinpath(@__DIR__, "..", "phase4", "results", "theta_$(b)_$(san(name))_$ε₀.csv")
        isfile(file) && push!(out, ("$b $name $ε₀", spec, () -> make(spec, edge_list(spec.P)).build(vec(readdlm(file, ',')))))
    end
    for ε₀ in (1e-3, 0.1), (name, _, _) in NAMED
        spec = fit_spec(name, ε₀)
        file = joinpath(OUT, "theta_two_default_$(san(name))_$ε₀.csv")
        isfile(file) && push!(out, ("two $name $ε₀", spec, () -> two_signed(spec, edge_list(spec.P), vec(readdlm(file, ',')))))
    end
    for (name, _, _) in NAMED, r in (:excitable, :oscillator, :fold, :design)
        spec = fit_spec(name, 0.1)
        file = joinpath(OUT, "theta_ctrnn_$(r)_$(san(name))_0.1.csv")
        # A saved fit is read without the search's stiffness cap: the design regime doesn't use it.
        isfile(file) && push!(out, ("ctrnn/$r $name 0.1", spec, () -> ctrnn_model(spec, vec(readdlm(file, ',')); cap=Inf)))
    end
    return out
end

# The classification run itself, when this file is run rather than included.
if abspath(PROGRAM_FILE) == @__FILE__
limit = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 500.0
clock = time()
for (name, known, expected, run) in fixtures()
    already("known", name) && continue
    time() - clock > limit && (println("time used; rerun to continue"); exit(0))
    class, o = run()
    record("known", name, known, expected, class, o)
end
for (name, spec, make) in fits()
    already("fit", name) && continue
    time() - clock > limit && (println("time used; rerun to continue"); exit(0))
    class, o = classify(make(); spec, Tnoisy=40sum(spec.τ), Tfree=150sum(spec.τ))
    record("fit", name, "–", "–", class, o)
end
println("all classified")
end
