# Differential tests for the synchrony backend, the race seed and the out-rate correction, against the
# prototype (phase 6). Local only, like run.jl. From the repository root:
#   julia --project=test/differential -t 7 test/differential/synchrony.jl
using Test, Serialization, StaticArrays
const PROTO = joinpath(@__DIR__, "..", "..", "research", "benchmarks")
include(joinpath(PROTO, "phase6", "seed1d.jl"))            # synchrony.jl, phases 2–5 and the race
include(joinpath(PROTO, "phase5", "twoway_cases.jl"))
import GenerateDynamicalNetworks as G

pspec(name, ε) = (c = only(x for x in [NAMED; TWOWAY] if x[1] == name);
                  (; P=float(c[2]), τ=c[3], form=:u, ε₀=ε, sample=startswith(name, "worm") ? WORM_SPEC.sample : minimum(c[3]) / 10))
san6(s) = replace(s, r"[^A-Za-z0-9]+" => "_")             # the suite's file names
same(a::SyncModel, b::G.SyncModel) = a.P == b.P && a.visible == b.visible && a.α[1] == b.α && a.M == b.M && a.Λ == b.Λ &&
                                     a.node_of == b.node_of && a.corner_of == b.corner_of && a.ε == b.ε && a.ω == 0
same(a::SyncLayout, b::G.SyncLayout) = a.M == b.M && a.corner_of == b.corner_of && a.paths == b.paths
runs_equal(a, b) = length(a) == length(b) && all(k -> a[k].state == b[k].state && a[k].t == b[k].t && a[k].failed == b[k].failed, eachindex(a))

@testset "synchrony backend == prototype" begin
    @testset "layout, rates and model: $name, ε=$ε" for name in G.example_names(), ε in (1e-3, 0.1)
        ps, gs = pspec(name, ε), G.example(name; noise=ε)
        Lp, Lg = sync_layout(ps.P), G.sync_layout(gs.P)
        @test same(Lp, Lg)
        rough, fast = rough_rates(ps)
        @test (rough, fast) == G.rough_rates(gs)
        mp, mg = sync_model(ps, Lp, rough; fast, contract=fast), G.sync_model(gs, Lg, rough; fast, contract=fast)
        @test same(mp, mg)
        @test same(mp, G.SyncModel(gs, rough))
        qp, qg = sync_params(mp), G.parameters(mg)
        @test qp.G == qg.G && qp.K == qg.K && qp.node_of == qg.node_of && qp.ε == qg.ε
        @test sync_start(mp) == G.initial_state(mg)
        θs = [SVector{2Lp.M}(2π .* rand(Xoshiro(k), 2Lp.M)) for k in 1:5]
        @test all(θ -> drift_sync(θ, qp, 0.0) == G.sync_drift(θ, qg, 0.0), θs)
        for k in 1:4                                           # the starting Broyden matrices
            @test initial_response(k, log(1 / ε)) == G.initial_response(k, log(1 / ε))
        end
    end
    @testset "race" begin
        for (λs, c, η) in (([2.0], 8.0, 0.1), ([1.0, 3.0], 12.0, 0.1), ([-0.5, 0.7, 2.0], 9.0, 0.05), ([0.4, 0.6], [5.0, 7.0], 1e-3))
            @test race(λs, c, η) == G.race(λs, c, η)
            @test race_outputs(λs, c, η) == G.race_outputs(λs, c, η)
        end
        f, g = race_response_full(10.0, 0.1), G.race_response(10.0, 0.1)
        @test f([1, 2], [1.0, 2.0, 3.0], 0.05) == g([1, 2], [1.0, 2.0, 3.0], 0.05)
    end
    @testset "race seed: $name, ε=$ε" for name in G.example_names(), ε in (1e-3, 0.1)
        ps, gs = pspec(name, ε), G.example(name; noise=ε)
        rough, fast = rough_rates(ps)
        sp, cp = seed_out_race(ps; start=rough)
        @test cp == fast
        @test sp == G.race_seed_rates(gs, rough, fast, ε)
    end
    @testset "correction passes, $seed seed" for seed in (:rough, :race)
        ps, gs = pspec("Δ-clique", 0.1), G.example("Δ-clique"; noise=0.1)
        L = sync_layout(ps.P)
        rough, fast = rough_rates(ps)
        kw = seed === :race ? (; seed=seed_out_race(ps; start=rough)[1], response=race_response_full(fast, 0.1)) : (;)
        fp = fit_sync(ps, L; passes=4, kw...)
        out0 = seed === :race ? G.race_seed_rates(gs, rough, fast, 0.1) : rough
        build = out -> G.sync_model(gs, G.sync_layout(gs.P), out; fast, contract=fast)
        tg = G.correct_rates(gs, build, out0; passes=4, response=seed === :race ? G.race_response(fast, 0.1) : nothing)
        @test same(fp.model, build(tg.out))
        @test [t.out for t in tg.trace] == [t.out for t in fp.trace] && [t.f for t in tg.trace] == [t.f for t in fp.trace] &&
              [t.miss for t in tg.trace] == [t.miss for t in fp.trace]
        # fit, stopped by a deadline and resumed from the checkpoint: the same rates
        file = tempname()
        @test_throws G.OutOfTime G.fit(gs, G.Synchrony(; seed, passes=4); checkpoint=file, deadline=time())
        @test same(fp.model, G.fit(gs, G.Synchrony(; seed, passes=4); checkpoint=file))
        rm(file)
    end
    @testset "stored fits ($(basename(dir))), rebuilt, simulated and classified" for dir in
            (joinpath(PROTO, "phase6", "results"), joinpath(PROTO, "phase6", "results", "seed1d"))
        checked = 0
        for file in filter(f -> startswith(f, "fit_") && endswith(f, ".jls"), readdir(dir))
            st = deserialize(joinpath(dir, file))
            k = findfirst(n -> "fit_$(san6(n))_$(st.spec.ε₀).jls" == file, G.example_names())
            k === nothing && continue                          # the measured worm graph isn't an example
            name = G.example_names()[k]
            ps, gs = st.spec, G.example(name; noise=st.spec.ε₀)
            fast = G.rough_rates(gs)[2]
            @test same(st.layout, G.sync_layout(gs.P)) && st.fast == fast && st.contract == fast
            mp, mg = sync_model(ps, st.layout, st.out; st.fast, st.contract, st.arrive, st.floor), G.SyncModel(gs, st.out)
            @test same(mp, mg)
            T = 10sum(ps.τ)
            @test runs_equal(simulate_frames(mp; T, seeds=1:4, ps.sample), G.simulate(mg, gs; T, seeds=1:4))
            @test runs_equal(simulate_frames(mp; T, seeds=1:4, ps.sample, dtscale=0.005), G.simulate(mg, gs; T, seeds=1:4, dtscale=0.005))
            if name in ("foraging", "Δ-clique")
                cp, cg = classify(mp; spec=ps, Tnoisy=40sum(ps.τ), Tfree=150sum(ps.τ)), G.classify(mg, gs)
                @test cp[1] == cg.class && cp[2] == cg.outcomes
            end
            checked += 1
        end
        @test checked == 12
    end
    @testset "full refit of the stored Δ-clique fit at 0.1, $seed seed" for (seed, sub) in ((:rough, ""), (:race, "seed1d"))
        st = deserialize(joinpath(PROTO, "phase6", "results", sub, "fit__clique_0.1.jls"))
        gs = G.example("Δ-clique"; noise=0.1)
        @test G.fit(gs, G.Synchrony(; seed)).Λ == G.SyncModel(gs, st.out).Λ
    end
end
