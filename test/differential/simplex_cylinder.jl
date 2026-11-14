# Differential tests for the simplex and cylinder backends against the prototype (phase 4). Local only, like run.jl.
# From the repository root:
#   julia --project=test/differential -t 7 test/differential/simplex_cylinder.jl
# The stored suite at the end reruns all 24 phase 4 fits (about half an hour on a quiet machine); an argument
# such as 0.1 reruns only that noise level.
using Test, DelimitedFiles
const PROTO = joinpath(@__DIR__, "..", "..", "research", "benchmarks")
include(joinpath(PROTO, "phase5", "classes.jl"))           # phases 2–5, with phase 4's backends and classify
include(joinpath(PROTO, "phase5", "twoway_cases.jl"))
import GenerateDynamicalNetworks as G

pspec(name, ε) = (c = only(x for x in [NAMED; TWOWAY] if x[1] == name);
                  (; P=float(c[2]), τ=c[3], form=:u, ε₀=ε, sample=startswith(name, "worm") ? WORM_SPEC.sample : minimum(c[3]) / 10))
gspec(s) = G.TransitionSpec(s.P, s.τ; noise=s.ε₀, sample=s.sample)
same(p::NamedTuple, g::G.LVModel) = p.form === g.form && p.P == g.P && p.visible == g.visible && p.κ == g.κ && p.β == g.β &&
                                    p.α == g.α && p.ε == g.ε && p.ρ == g.ρ
same(p::CylinderModel, g::G.CylinderModel) = p.P == g.P && p.visible == g.visible && p.α == [g.s] && p.Lα == g.Lα &&
                                             p.from == g.from && p.to == g.to && p.ε == g.ε
runs_equal(a, b) = length(a) == length(b) && all(k -> a[k].state == b[k].state && a[k].t == b[k].t && a[k].failed == b[k].failed, eachindex(a))
gstart(b, gs) = b === :simplex ? G.simplex_start(gs) : G.cylinder_start(gs, 1.5625)
gbuild(b, gs) = b === :simplex ? θ -> G.simplex_model(gs, θ) : θ -> G.CylinderModel(gs, θ)
pmake(b, ps) = (b === :simplex ? make_simplex : make_cylinder)(ps, edge_list(ps.P))
san(n) = replace(n, r"[^A-Za-z0-9]+" => "_")

@testset "simplex and cylinder backends == prototype" begin
    @testset "start, box and model: $b, $name, ε=$ε" for b in (:simplex, :cylinder), name in G.example_names(), ε in (1e-3, 0.1)
        ps = pspec(name, ε)
        gs = gspec(ps)
        p, g = pmake(b, ps), gstart(b, gs)
        @test p.θ0 == g.θ0 && p.lower == g.lower && p.upper == g.upper && p.seeded == g.seeded
        θ = clamp.(p.θ0 .+ 0.2 .* randn(Xoshiro(7), length(p.θ0)), p.lower, p.upper)
        mp, mg = p.build(θ), gbuild(b, gs)(θ)
        @test same(mp, mg)
        b === :cylinder || continue
        q = G.parameters(mg)
        M = length(mg.Lα) + 1
        zs = [SVector{M}([0.3 .* randn(Xoshiro(k), M - 1); 1 + 2rand(Xoshiro(k + 100))]) for k in 1:5]
        @test all(z -> drift_cylinder(z, q, 0.0) == G.drift_function(mg)(z, q, 0.0), zs)
        @test G.speed(mg) == mp.α[1]
    end
    @testset "correction passes: $b, $name" for b in (:simplex, :cylinder), (name, ε) in (("foraging", 0.1), ("Δ-clique", 1e-3))
        ps = pspec(name, ε)
        gs = gspec(ps)
        p = pmake(b, ps)
        edges = edge_list(ps.P)
        T = 20sum(ps.τ)
        θ0 = clamp.(p.θ0, p.lower, p.upper)
        tp, θp = correct_edges(p.build, θ0, ps, edges, p.lower, p.upper; passes=3, runs=8, T)
        tg, θg = G.correct_edges(gbuild(b, gs), θ0, gs, edges, p.lower, p.upper; passes=3, runs=8, T)
        @test θp == θg && [r.f for r in tp] == [r.f for r in tg] && θp != θ0
    end
    @testset "short fit: $b, foraging, ε=0.1" for b in (:simplex, :cylinder)
        ps = pspec("foraging", 0.1)
        gs = gspec(ps)
        kw = (; passes=2, runs=8, T=20sum(ps.τ), budget=24, popsize=6)
        fp = fit_backend(ps, b === :simplex ? make_simplex : make_cylinder; kw..., seconds=1e9)
        println("  short $b fit stage: ", fp.stage)
        mg = b === :simplex ? G.fit(gs, G.Simplex(; kw..., seconds=Inf)) : G.fit(gs, G.Cylinder(; kw..., seconds=Inf))
        @test same(fp.model, mg)
    end
    # The stored phase 4 suite (results/suite.csv and theta_*.csv): every stored θ rebuilt, simulated and classified,
    # then the full default fit (search unlimited by time) giving the same model and the same held-out numbers.
    rows = readdlm(joinpath(PROTO, "phase4", "results", "suite.csv"), ','; header=true)[1]
    @testset "stored θ: $(r[3]) $(r[1]), ε=$(r[2])" for r in eachrow(rows)
        name, ε, b = String(r[1]), Float64(r[2]), Symbol(r[3])
        ps = pspec(name, ε)
        gs = G.example(name; noise=ε)
        @test gs.P == ps.P && gs.τ == ps.τ && gs.sample == ps.sample
        θ = vec(readdlm(joinpath(PROTO, "phase4", "results", "theta_$(b)_$(san(name))_$ε.csv"), ','))
        mp, mg = pmake(b, ps).build(θ), gbuild(b, gs)(θ)
        @test same(mp, mg)
        @test runs_equal(simulate_frames(mp; T=10sum(ps.τ), seeds=1:4, ps.sample), G.simulate(mg, gs; T=10sum(ps.τ), seeds=1:4))
        Tnoisy, Tfree = 20sum(ps.τ), 60sum(ps.τ)
        kp = classify(mp; spec=ps, Tnoisy, Tfree)
        kg = G.classify(mg, gs; Tnoisy, Tfree)
        @test kp[1] === kg.class && kp[2] == kg.outcomes
    end
    # An optional argument limits the full refits to one noise level, e.g. `… simplex_cylinder.jl 0.1`.
    refit = isempty(ARGS) ? rows : rows[rows[:, 2] .== parse(Float64, ARGS[1]), :]
    @testset "stored suite: $(r[3]) $(r[1]), ε=$(r[2])" for r in eachrow(refit)
        name, ε, b = String(r[1]), Float64(r[2]), Symbol(r[3])
        b === :simplex && name == "foraging" && ε == 0.1 && get(ENV, "SKIP_DONE", "") == "1" && continue
        ps = pspec(name, ε)
        gs = G.example(name; noise=ε)
        t = @elapsed m = G.fit(gs, b === :simplex ? G.Simplex(; seconds=Inf) : G.Cylinder(; seconds=Inf))
        println("  $b $name ε=$ε: fit $(round(t; digits=1)) s (prototype $(round(r[16]; digits=1)) s, stage $(r[4]))")
        θ = vec(readdlm(joinpath(PROTO, "phase4", "results", "theta_$(b)_$(san(name))_$ε.csv"), ','))
        model_ok = same(pmake(b, ps).build(θ), m)
        v = G.validate(m, gs; T=250sum(gs.τ))
        again = G.validate(m, gs; T=250sum(gs.τ), seeds=9_200_001:9_200_064)
        numbers_ok = maximum(abs, v.err) == r[7] && v.rms == r[8] && v.off == r[9] && v.dwell_ok == r[10] &&
                     v.branch_ok == r[11] && again.branch_ok == r[12] && v.pass == r[14]
        println("    same model: $model_ok, same held-out numbers: $numbers_ok")
        flush(stdout)
        @test model_ok
        @test numbers_ok
    end
end
