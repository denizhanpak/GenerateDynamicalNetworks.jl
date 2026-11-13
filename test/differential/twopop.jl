# Differential tests for the shared fitting code (dwell-formula seed, edge correction, CMA-ES search) and the
# two-population backend, against the prototype (phases 2–5). Local only, like run.jl. From the repository root:
#   julia --project=test/differential -t 7 test/differential/twopop.jl
using Test, DelimitedFiles
const PROTO = joinpath(@__DIR__, "..", "..", "research", "benchmarks")
include(joinpath(PROTO, "phase5", "classes.jl"))           # phases 2–5 and fits()
include(joinpath(PROTO, "phase5", "twoway_cases.jl"))
import GenerateDynamicalNetworks as G

pspec(name, ε) = (c = only(x for x in [NAMED; TWOWAY] if x[1] == name);
                  (; P=float(c[2]), τ=c[3], form=:u, ε₀=ε, sample=startswith(name, "worm") ? WORM_SPEC.sample : minimum(c[3]) / 10))
same(a::TwoPopModel, b::G.TwoPopModel) = a.P == b.P && a.visible == b.visible && a.α == b.α && a.s == b.s && a.B == b.B &&
                                         a.from == b.from && a.to == b.to && a.ε == b.ε
attempt(f) = try f() catch err; err isa ArgumentError ? :no_design : rethrow() end

@testset "shared fitting code and two-population backend == prototype" begin
    @testset "dwell formulas" begin
        for (δ, ε, λu, λs) in ((0.3, 1e-3, 2.0, 1.0), (0.1, 0.05, 0.5, 3.0), (sqrt(0.1), 0.1, 7.5, 7.5))
            @test G.stone_holmes_mean(δ, ε, λu, λs) == passagetime(δ, ε, λu, λs)    # SHCTools.passagetime itself
        end
        for εs in ([0.1], [0.1, 0.1], [0.05, 0.2], [1e-3, 2e-3, 5e-3])
            @test G.effective_noise(εs) == effective_noise(εs)
        end
        for name in G.example_names(), ε in (1e-3, 0.1), form in (:u, :x)
            gs = G.example(name; noise=ε)
            n = length(gs.τ)
            ε_eff = [G.effective_noise(fill(ε, length(successors(gs.P, i)))) for i in 1:n]
            @test attempt(() -> G.dwell_rates(form, gs.τ, ε_eff, ones(n))) == attempt(() -> dwell_rates(form, gs.τ, ε_eff, ones(n)))
        end
    end
    @testset "edge seed: $name, ε=$ε" for name in G.example_names(), ε in (1e-3, 0.1)
        ps, gs = pspec(name, ε), G.example(name; noise=ε)
        @test seed_rates(ps, edge_list(ps.P)) == G.edge_seed_rates(gs, G.edge_list(gs.P))
    end
    ps, gs = pspec("foraging", 0.1), G.example("foraging"; noise=0.1)
    edges = edge_list(ps.P)
    ne = length(edges)
    llo, lhi = [log(0.05); fill(log(1e-4), ne)], [log(200.0); fill(log(5.0), ne)]
    r = seed_rates(ps, edges)
    θ0 = clamp.([log(r.α / TWO.contract); r.logs .+ log(TWO.contract)], llo, lhi)
    T = 20sum(ps.τ)
    @testset "edge correction passes" begin
        tp, θp = correct_edges(θ -> two_log(ps, edges, θ), θ0, ps, edges, llo, lhi; passes=3, runs=8, T)
        tg, θg = G.correct_edges(θ -> G.twopop_log(gs, edges, θ), θ0, gs, edges, llo, lhi; passes=3, runs=8, T)
        @test θp == θg && [p.f for p in tp] == [p.f for p in tg] && θp != θ0
    end
    @testset "CMA-ES searches" begin
        slo, shi = [log(0.05); fill(asinh(-5.0 / TWO.ν0), ne)], [log(200.0); fill(asinh(0.3 / TWO.ν0), ne)]
        start = clamp.([θ0[1]; asinh.(-exp.(θ0[2:end]) ./ TWO.ν0)], slo .+ 1e-6, shi .- 1e-6)
        kw = (; budget=30, popsize=6, runs=4, T)
        a = timed_search(θ -> two_signed(ps, edges, θ), ps, start, 0.3, slo, shi; kw..., maxtime=1e9)
        b = G.timed_search(θ -> G.twopop_signed(gs, edges, θ), gs, start, 0.3, slo, shi; kw..., maxtime=1e9)
        @test a.θ == b.θ && a.evals == b.evals
        c = search(θ -> two_signed(ps, edges, θ), ps, start, 0.3, slo, shi; budget=60, restarts=1, popsize=6, runs=4, T)
        d = G.cma_search(θ -> G.twopop_signed(gs, edges, θ), gs, start, 0.3, slo, shi; budget=60, restarts=1, popsize=6, runs=4, T)
        @test c.θ == d.θ && c.candidates == d.candidates && c.evals == d.evals
    end
    @testset "fit pipeline: $name" for name in ("foraging", "Δ-clique")
        ps, gs = pspec(name, 0.1), G.example(name; noise=0.1)
        T = 20sum(ps.τ)
        mp = fit_two(ps; passes=2, runs=8, T, budget=30, popsize=6, seconds=1e9).model
        @test same(mp, G.fit(gs, G.TwoPopulation(; passes=2, runs=8, T, budget=30, popsize=6, seconds=Inf)))
    end
    @testset "stored fits, rebuilt and simulated" begin
        checked = 0
        for (label, spec, make) in fits()
            startswith(label, "two ") || continue
            name = only(n for n in G.example_names() if occursin(" $n ", label))
            ε = spec.ε₀
            gs = G.example(name; noise=ε)
            θ = vec(readdlm(joinpath(PROTO, "phase5", "results", "theta_two_default_$(san(name))_$ε.csv"), ','))
            mp, mg = make(), G.TwoPopModel(gs, θ)
            @test same(mp, mg)
            rp = simulate_frames(mp; T=10sum(spec.τ), seeds=1:4, spec.sample)
            rg = G.simulate(mg, gs; T=10sum(spec.τ), seeds=1:4)
            @test all(k -> rp[k].state == rg[k].state && rp[k].t == rg[k].t, 1:4)
            checked += 1
        end
        @test checked == 12
    end
end
