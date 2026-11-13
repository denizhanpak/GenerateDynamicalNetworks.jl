# Differential tests: the package's core layer against the research prototype on the same seeds.
# Every prototype model type is wrapped in the package's backend interface; the package's simulator,
# measurement, acceptance checks and classifier must reproduce the prototype's results exactly.
#
# Local only: it needs SHCTools.jl and CTRNNs.jl checked out beside this repository (see
# research/README.md) and the research results. From the repository root:
#   julia --project=test/differential -t 7 test/differential/run.jl
using Test, DelimitedFiles, Random
const PROTO = joinpath(@__DIR__, "..", "..", "research", "benchmarks")
include(joinpath(PROTO, "phase7", "hopfield.jl"))          # the prototype chain, phases 2–7
include(joinpath(PROTO, "phase5", "twoway_cases.jl"))
import GenerateDynamicalNetworks as G
import GenerateDynamicalNetworks: drift_function, noise_function, parameters, initial_state, speed, frame_state, in_bounds,
                                  project, class_state, class_start

struct Wrapped{T} <: G.AbstractNetworkModel
    m::T
end
speed(w::Wrapped) = maximum(w.m.α)

# Horchler / simplex (u or x form), as in phase2/horchler.jl and phase3/optimize.jl
const W_LV = Wrapped{<:NamedTuple}
drift_function(w::W_LV) = w.m.form === :x ? drift_x : drift_u
noise_function(::W_LV) = diffusion
parameters(w::W_LV) = sde_problem(w.m; T=1.0).p
initial_state(w::W_LV) = sde_problem(w.m; T=1.0).u0
frame_state(w::W_LV, q, u) = (i = findmax(abs, u)[2]; i <= w.m.visible ? i : 0)
project(w::W_LV, u) = w.m.form === :x ? max.(u, 0.0) : nothing
class_state(::W_LV, q, u) = argmax(abs.(u))
class_start(w::W_LV) = (n = length(w.m.α); SVector{n}(ntuple(i -> i == 1 ? 0.95 : 1e-3, n)))

# Cylinder (phase4/backends.jl)
const W_CYL = Wrapped{<:CylinderModel}
drift_function(::W_CYL) = drift_cylinder
noise_function(::W_CYL) = noise_cylinder
parameters(w::W_CYL) = (m = w.m; (; s=m.α[1], m.Lα, m.from, m.to, m.ε, m.visible))
initial_state(w::W_CYL) = (N = length(w.m.Lα); SVector{N + 1}(ntuple(k -> k <= N ? w.m.ε : 1.0, N + 1)))
function frame_state(w::W_CYL, q, z)
    M = length(z)
    k = round(Int, z[M])
    return 1 <= k <= w.m.visible && abs(z[M] - k) < 0.25 && maximum(i -> abs(z[i]), 1:M-1) < 0.5 ? k : 0
end
class_start(w::W_CYL) = (N = length(w.m.Lα); SVector{N + 1}(ntuple(k -> k <= N ? 1e-3 : 1.0, N + 1)))

# Two-population (phase5/backends.jl)
const W_TWO = Wrapped{<:TwoPopModel}
drift_function(::W_TWO) = drift_two
noise_function(::W_TWO) = noise_flat
parameters(w::W_TWO) = two_params(w.m)
initial_state(w::W_TWO) = two_start(w.m)
frame_state(::W_TWO, q, z) = findmax(i -> abs(z[i]), 1:q.nv)[2]

# CTRNN (phase5/backends.jl)
const W_NET = Wrapped{CTRNNModel}
drift_function(::W_NET) = drift_ctrnn
noise_function(::W_NET) = noise_flat
parameters(w::W_NET) = ctrnn_params(w.m)
initial_state(w::W_NET) = ctrnn_start(w.m)
in_bounds(::W_NET, y) = maximum(abs, y) < 1000
frame_state(::W_NET, q, y) = argmax(i -> q.g * (y[i] + q.b[i]), 1:q.nv)

# Synchrony oscillators (phase6/synchrony.jl) and Hopfield (phase7/hopfield.jl)
const W_SYN = Wrapped{SyncModel}
drift_function(::W_SYN) = drift_sync
noise_function(::W_SYN) = noise_flat
parameters(w::W_SYN) = sync_params(w.m)
initial_state(w::W_SYN) = sync_start(w.m)
in_bounds(::W_SYN, θ) = all(isfinite, θ)
w_sync_corner(θ, M) = 1 + sum((cos(θ[2σ] - θ[2σ-1]) < 0) << (σ - 1) for σ in 1:M)
frame_state(w::W_SYN, q, θ) = q.node_of[w_sync_corner(θ, length(θ) ÷ 2)]
class_state(w::W_SYN, q, θ) = (v = w_sync_corner(θ, w.m.M); w.m.node_of[v] == 0 ? -v : w.m.node_of[v])

const W_HOP = Wrapped{HopModel}
drift_function(::W_HOP) = drift_hop
noise_function(::W_HOP) = noise_flat
parameters(w::W_HOP) = hop_params(w.m)
initial_state(w::W_HOP) = hop_start(w.m)
in_bounds(::W_HOP, ψ) = all(isfinite, ψ)
w_hop_corner(ψ) = 1 + sum((cos(ψ[j]) < 0) << (j - 1) for j in eachindex(ψ))
frame_state(::W_HOP, q, ψ) = q.node_of[w_hop_corner(ψ)]
class_state(w::W_HOP, q, ψ) = (v = w_hop_corner(ψ); w.m.node_of[v] == 0 ? -v : w.m.node_of[v])

# ── The models: one per backend, from the prototype's design or stored fits ─────────────────────

pspec(name, ε) = (c = only(x for x in [NAMED; TWOWAY] if x[1] == name);
                  (; P=float(c[2]), τ=c[3], form=:u, ε₀=ε, sample=startswith(name, "worm") ? WORM_SPEC.sample : minimum(c[3]) / 10))
csplit(l) = split(l, r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)")
function stored_rows(file, name, ε)
    rows = [csplit(l) for l in readlines(file)[2:end]]
    return [r for r in rows if strip(r[1], '"') == name && parse(Float64, r[2]) == ε]
end

function models()
    out = Any[]
    ring3 = (; P=float(ring(3)), τ=[1.0, 3.0, 10.0], form=:u, ε₀=0.1, sample=0.1)
    push!(out, ("Horchler u, 3-ring", ring3, design(ring3.P, ring3.τ; form=:u, ε₀=0.1)))
    ring3x = merge(ring3, (; form=:x, ε₀=1e-3))
    push!(out, ("Horchler x, 3-ring", ring3x, design(ring3x.P, ring3x.τ; form=:x, ε₀=1e-3)))
    stored = fits()
    for (label, want) in (("simplex", "simplex foraging 0.1"), ("cylinder", "cylinder Δ-clique 0.1"), ("two-population", "two foraging 0.1"),
                          ("CTRNN design", "ctrnn/design foraging 0.1"))
        k = findfirst(f -> f[1] == want, stored)
        k === nothing && (@warn "no stored fit \"$want\""; continue)
        name, spec, make = stored[k]
        push!(out, ("$label, $(split(name, ' ')[2]) 0.1", spec, make()))
    end
    spec = pspec("foraging", 0.1)
    rows = stored_rows(joinpath(PROTO, "phase6", "results", "seed1d", "fits.csv"), "foraging", 0.1)
    out6, (fast, contract, arrive, floor) = [parse(Float64, r[4]) for r in rows], parse.(Float64, rows[1][5:8])
    push!(out, ("synchrony, foraging 0.1", spec, sync_model(spec, sync_layout(spec.P), out6; fast, contract, arrive, floor)))
    rows = stored_rows(joinpath(PROTO, "phase7", "results", "fits.csv"), "foraging", 0.1)
    out7, fast, contract = [parse(Float64, r[4]) for r in rows], parse(Float64, rows[1][6]), parse(Float64, rows[1][7])
    push!(out, ("Hopfield, foraging 0.1", spec, first(hop_model(spec, hop_layout(spec), out7; fast, contract))))
    return out
end

gspec(s) = G.TransitionSpec(s.P, s.τ; noise=s.ε₀, sample=s.sample)
branches(v) = [(b.from, b.to, b.designed, b.measured, b.ok) for b in v.branch]

@testset "package core layer == prototype" begin
    @testset "examples" begin
        for (name, P, τ) in [NAMED; TWOWAY]
            name in G.example_names() || continue
            p, g = pspec(name, 0.1), G.example(name; noise=0.1)
            @test g.P == p.P && g.τ == p.τ && g.sample == p.sample
        end
        @test all(s -> G.random_network(Xoshiro(s), 6, 0.35) == random_spec_matrix(Xoshiro(s), 6, 0.35), 1:30)
    end
    for (label, spec, m) in models()
        @testset "$label" begin
            w, gs, n = Wrapped(m), gspec(spec), size(spec.P, 1)
            T, seeds = 20sum(spec.τ), 1:8
            rp = simulate_frames(m; T, seeds, spec.sample)
            rg = G.simulate(w, gs; T, seeds)
            @test all(k -> rp[k].state == rg[k].state && rp[k].t == rg[k].t && rp[k].failed == rg[k].failed, eachindex(rp))
            @test sum(r -> length(r.state), rg) > 8                    # the runs actually move
            mp, mg = measure(rp, n), G.measure(rg, n)
            @test mp.counts == mg.counts && isequal(mp.dwell, mg.dwell) && isequal(mp.sem, mg.sem)
            vp, vg = verdict(mp, spec.P, spec.τ), G.verdict(mg, spec.P, spec.τ)
            @test isequal(vp.err, vg.err) && vp.off == vg.off && vp.pass == vg.pass && vp.branch == branches(vg)
            @test isequal(score(rp, spec).f, G.score(rg, gs).f)
            held = 9_000_001:9_000_008
            cp, cg = validate(m, spec; seeds=held, T, dtscale=0.005), G.validate(w, gs; seeds=held, T, dtscale=0.005)
            @test isequal(cp.dwell, cg.dwell) && isequal(cp.err, cg.err) && isequal(cp.rms, cg.rms) && cp.off == cg.off &&
                  cp.failed == cg.failed && cp.pass == cg.pass && cp.branch == branches(cg)
            Tnoisy, Tfree = 10sum(spec.τ), 30sum(spec.τ)
            # the prototype's classify(::NamedTuple) always used the u-form drift (it only met u-form models), so an
            # x-form model is compared through its generic form with the x-form drift
            kp = m isa NamedTuple && m.form === :x ?
                 classify(drift_x, parameters(w), class_start(w), u -> argmax(abs.(u)); noise=spec.ε₀, Tnoisy, Tfree, dt=0.02 / maximum(m.α)) :
                 classify(m; spec, Tnoisy, Tfree)
            kg = G.classify(w, gs; Tnoisy, Tfree)
            @test kp[1] === kg.class && kp[2] == kg.outcomes
            println("  $label: $(sum(r -> length(r.state), rg)) visits, class $(kg.class)")
        end
    end
end
