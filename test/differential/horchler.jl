# Differential tests for the Horchler backend against the prototype (phases 2–3). Local only, like run.jl.
# From the repository root:
#   julia --project=test/differential -t 7 test/differential/horchler.jl
using Test, DelimitedFiles
const PROTO = joinpath(@__DIR__, "..", "..", "research", "benchmarks")
include(joinpath(PROTO, "phase5", "classes.jl"))           # phases 2–5, with classify(::NamedTuple)
include(joinpath(PROTO, "phase5", "twoway_cases.jl"))
import GenerateDynamicalNetworks as G

pspec(name, ε; form=:u) = (c = only(x for x in [NAMED; TWOWAY] if x[1] == name);
                           (; P=float(c[2]), τ=c[3], form, ε₀=ε, sample=startswith(name, "worm") ? WORM_SPEC.sample : minimum(c[3]) / 10))
gspec(s) = G.TransitionSpec(s.P, s.τ; noise=s.ε₀, sample=s.sample)
same(p::NamedTuple, g::G.LVModel) = p.form === g.form && p.P == g.P && p.visible == g.visible && p.κ == g.κ && p.β == g.β &&
                                    p.α == g.α && p.ε == g.ε && p.ρ == g.ρ
runs_equal(a, b) = length(a) == length(b) && all(k -> a[k].state == b[k].state && a[k].t == b[k].t && a[k].failed == b[k].failed, eachindex(a))
attempt(f) = try f() catch err; err isa ArgumentError ? sprint(showerror, err) : rethrow() end
traces_equal(a, b) = length(a) == length(b) && all(eachindex(a)) do k
    p, g = a[k], b[k]
    p.pass == g.pass && p.request == g.request && isequal(p.dwell, g.dwell) && isequal(p.off, g.off) && p.f == g.f && p.note == g.note &&
        (p.model === nothing ? g.design === nothing : same(p.model, G.lv_model(g.design)))
end

@testset "Horchler backend == prototype" begin
    @testset "repair and design: $name, $form, ε=$ε" for name in G.example_names(), form in (:u, :x), ε in (1e-5, 1e-3, 0.1)
        ps = pspec(name, ε; form)
        @test repair(ps.P, ps.τ) == G.repair(ps.P, ps.τ)
        p = attempt(() -> design(ps.P, ps.τ; form, ε₀=ε))
        g = attempt(() -> G.horchler_design(ps.P, ps.τ; form, noise=ε))
        @test p == g
        p isa NamedTuple || continue
        m = G.lv_model(g)
        @test same(p, m)
        pr = sde_problem(p; T=1.0)
        q = G.parameters(m)
        @test pr.p.α == q.α && pr.p.ρ == q.ρ && pr.p.ε == q.ε && pr.p.visible == q.visible
        @test pr.u0 == G.initial_state(m) && G.speed(m) == maximum(p.α)
        @test pr.f.f === (form === :x ? drift_x : drift_u)
        us = [SVector{length(p.α)}(abs.(randn(Xoshiro(k), length(p.α)))) for k in 1:5]
        @test all(u -> pr.f.f(u, pr.p, 0.0) == G.drift_function(m)(u, q, 0.0), us)
    end
    @testset "simulation and class: $name, $form" for name in ("foraging", "Δ-clique", "flip-flop"), form in (:u, :x)
        ε = form === :x ? 1e-5 : 0.1
        ps = pspec(name, ε; form)
        gs = gspec(ps)
        p = design(ps.P, ps.τ; form, ε₀=ε)
        m = G.lv_model(G.horchler_design(ps.P, ps.τ; form, noise=ε))
        T = 10sum(ps.τ)
        @test runs_equal(simulate_frames(p; T, seeds=1:4, ps.sample), G.simulate(m, gs; T, seeds=1:4))
        @test runs_equal(simulate_frames(p; T, seeds=5:8, ps.sample, dtscale=0.005), G.simulate(m, gs; T, seeds=5:8, dtscale=0.005))
        Tnoisy, Tfree = 40sum(ps.τ), 150sum(ps.τ)
        # the prototype's classify(::NamedTuple) always used the u-form drift, so an x-form model goes through the generic one
        kp = form === :x ?
             classify(drift_x, sde_problem(p; T=1.0).p, SVector{length(p.α)}(ntuple(i -> i == 1 ? 0.95 : 1e-3, length(p.α))),
                      u -> argmax(abs.(u)); noise=ε, Tnoisy, Tfree, dt=0.02 / maximum(p.α)) :
             classify(p; spec=ps, Tnoisy, Tfree)
        kg = G.classify(m, gs; Tnoisy, Tfree)
        @test kp[1] === kg.class && kp[2] == kg.outcomes
    end
    @testset "correction passes and polish: $name, ε=$ε" for (name, ε) in (("foraging", 0.1), ("Δ-clique", 1e-3), ("flip-flop", 0.1))
        ps = pspec(name, ε)
        gs = gspec(ps)
        T = 30sum(ps.τ)
        tp = correct(ps; passes=3, runs=8, T)
        tg = G.correct_requests(gs, :u; passes=3, runs=8, T)
        @test traces_equal(tp, tg)
        d = first(tg).design
        mp, trp = correct_rates(first(tp).model, ps; passes=2, runs=8, T)
        dg, trg = G.polish_saddles(d, gs; passes=2, runs=8, T)
        @test same(mp, G.lv_model(dg)) && mp.ν == dg.ν
        @test all(k -> trp[k].ν == trg[k].ν && isequal(trp[k].dwell, trg[k].dwell) && trp[k].f == trg[k].f, eachindex(trp))
        θ = [log.(d.α) .+ 0.1; log.(d.ν) .- 0.2]
        @test same(raw_model(first(tp).model, θ), G.lv_model(G.raw_design(d, θ)))
    end
    @testset "search and short fit: foraging, ε=0.1" begin
        ps = pspec("foraging", 0.1)
        gs = gspec(ps)
        seed = design(ps.P, ps.τ; form=:u, ε₀=0.1)
        n = length(seed.α)
        lower, upper = [fill(log(0.05), n); fill(log(0.5), n)], [fill(log(200.0), n); fill(log(1e4), n)]
        θ0 = clamp.([log.(seed.α); log.(seed.ν)], lower .+ 1e-6, upper .- 1e-6)
        kw = (; budget=36, popsize=12, runs=4, T=30sum(ps.τ))
        a = search(θ -> raw_model(seed, θ), ps, θ0, 0.3, lower, upper; kw...)
        b = G.cma_search(θ -> G.lv_model(G.raw_design(seed, θ)), gs, θ0, 0.3, lower, upper; kw...)
        c = G.cma_search(θ -> G.lv_model(G.raw_design(seed, θ)), gs, θ0, 0.3, lower, upper; kw..., maxtime=1e9)
        @test a.θ == b.θ == c.θ && a.evals == b.evals
        fp = fit(ps; budget=36, T=30sum(ps.τ))
        println("  short fit stage: ", fp.stage)
        @test same(fp.model, G.fit(gs, G.Horchler(; budget=36, T=30sum(ps.τ), seconds=Inf)))
        @test same(fp.model, G.fit(gs, G.Horchler(; budget=36, T=30sum(ps.τ))))
    end
    @testset "stored worm-ring fits rebuilt" begin
        R3 = joinpath(PROTO, "phase3", "results")
        θ(name) = vec(readdlm(joinpath(R3, "worm_$(name)_theta.csv"), ','))
        gs = G.example("worm ring"; noise=0.1)
        seed = design(WORM_P, WORM_SPEC.τ; form=:u, ε₀=0.1)
        gseed = G.horchler_design(gs.P, gs.τ; form=:u, noise=0.1)
        @test seed == gseed && WORM_SPEC.sample == gs.sample
        models = [request_model(WORM_SPEC, θ("request")) => G.lv_model(G.horchler_design(gs.P, exp.(θ("request")); noise=0.1));
                  [raw_model(seed, θ(k)) => G.lv_model(G.raw_design(gseed, θ(k))) for k in ("raw", "raw_polished", "repeat_2", "repeat_4")]]
        for (p, g) in models
            @test same(p, g)
            @test runs_equal(simulate_frames(p; T=200.0, seeds=1:4, WORM_SPEC.sample), G.simulate(g, gs; T=200.0, seeds=1:4))
        end
    end
    # The stored phase 3 branching suite (results/branching.csv): full default fits, then the held-out check on
    # 64 runs of 250Στ. The package's fit (search unlimited by time) must give the same numbers.
    @testset "stored branching suite: $(r[1]), ε=$(r[2])" for r in eachrow(readdlm(joinpath(PROTO, "phase3", "results", "branching.csv"), ','; header=true)[1])
        name, ε = String(r[1]), Float64(r[2])
        gs = G.example(name; noise=ε)
        t = @elapsed m = G.fit(gs, G.Horchler(; seconds=Inf))
        println("  $name ε=$ε: fit $(round(t; digits=1)) s (prototype $(round(r[11]; digits=1)) s, stage $(r[3]))")
        @test m.visible == G.nstates(gs) && size(m.P, 1) - m.visible == r[4]
        v = G.validate(m, gs)
        @test maximum(abs, v.err) == r[5] && v.rms == r[6] && v.off == r[7] && v.branch_ok == r[8] && v.pass == r[9]
    end
end
