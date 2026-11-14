# Differential tests for the Hopfield backend against the prototype (phase 7). Local only, like run.jl.
# From the repository root:
#   julia --project=test/differential -t 7 test/differential/hopfield.jl
#
# The port fixes OSQP's adaptive_rho_interval at 25 (agreed 2026-11-13); OSQP's default sets it from
# measured setup time, so the prototype's designs vary from run to run by up to ~1e-9. The unpatched
# prototype's designs are checked against the port's within 1e-8; then the same fix is applied to an
# in-memory copy of the prototype's hop_design (the file is not changed), and everything else must match
# exactly.
using Test, Serialization, StaticArrays
const PROTO = joinpath(@__DIR__, "..", "..", "research", "benchmarks")
include(joinpath(PROTO, "phase7", "hopfield.jl"))           # synchrony, the race seed and phases 2–5
include(joinpath(PROTO, "phase5", "twoway_cases.jl"))
import GenerateDynamicalNetworks as G

pspec(name, ε) = (c = only(x for x in [NAMED; TWOWAY] if x[1] == name);
                  (; P=float(c[2]), τ=c[3], form=:u, ε₀=ε, sample=startswith(name, "worm") ? WORM_SPEC.sample : minimum(c[3]) / 10))
san6(s) = replace(s, r"[^A-Za-z0-9]+" => "_")             # the suite's file names
same(a::SyncLayout, b::G.SyncLayout) = a.M == b.M && a.corner_of == b.corner_of && a.paths == b.paths
same(a::HopModel, b::G.HopModel) = a.P == b.P && a.visible == b.visible && a.α[1] == b.α && a.n == b.n && a.J == b.J && a.b == b.b &&
                                   a.k == b.k && a.node_of == b.node_of && a.corner_of == b.corner_of && a.ε == b.ε
runs_equal(a, b) = length(a) == length(b) && all(k -> a[k].state == b[k].state && a[k].t == b[k].t && a[k].failed == b[k].failed, eachindex(a))
relgap(a, b) = maximum(abs, a .- b) / maximum(abs, a)

const R7 = joinpath(PROTO, "phase7", "results")
stored = Tuple{String,Any}[]
for f in readdir(R7)
    (startswith(f, "fit_") && endswith(f, ".jls")) || continue
    st = deserialize(joinpath(R7, f))
    k = findfirst(n -> "fit_$(san6(n))_$(st.spec.ε₀).jls" == f, G.example_names())
    k === nothing || push!(stored, (G.example_names()[k], st))      # the measured worm graph isn't an example
end
unpatched = [hop_design(st.layout, st.out; st.fast, st.contract) for (_, st) in stored]

let src = read(joinpath(PROTO, "phase7", "hopfield.jl"), String)
    a = findfirst("function hop_design", src)
    body = src[first(a):last(findnext("\nend\n", src, first(a)))]
    line = "set_attribute(model, \"polishing\", true)"
    @assert occursin(line, body)
    include_string(Main, replace(body, line => line * "\n        set_attribute(model, \"adaptive_rho_interval\", 25)"))
end

@testset "Hopfield backend == prototype (with the OSQP fix)" begin
    @testset "unpatched prototype designs within 1e-8" begin
        @test length(stored) == 12
        for ((name, st), d) in zip(stored, unpatched)
            g = G.hop_design(st.layout, st.out; st.fast, st.contract)
            @test d.k == g.k && relgap(d.J, g.J) < 1e-8 && relgap(d.b, g.b) < 1e-8
        end
    end
    @testset "layout, design, model and seed: $name, ε=$ε" for name in G.example_names(), ε in (1e-3, 0.1)
        ps, gs = pspec(name, ε), G.example(name; noise=ε)
        @test same(onehot_layout(ps.P), G.onehot_layout(gs.P))
        Lp, Lg = hop_layout(ps), G.hop_layout(gs)
        @test same(Lp, Lg)
        rough, c = rough_rates(ps)
        @test edge_contracts(Lp, rough; fast=c, contract=c) == G.edge_contracts(Lg, rough; fast=c, contract=c)
        (mp, wp), (mg, wg) = hop_model(ps, Lp, rough; fast=c, contract=c), G.hop_model(gs, Lg, rough; fast=c, contract=c)
        @test same(mp, mg) && wp == wg
        qp, qg = hop_params(mp), G.parameters(mg)
        @test qp.J == qg.J && qp.b == qg.b && qp.k == qg.k && qp.node_of == qg.node_of && qp.ε == qg.ε
        @test hop_start(mp) == G.initial_state(mg)
        ψs = [SVector{mp.n}(2π .* rand(Xoshiro(k), mp.n)) for k in 1:5]
        @test all(ψ -> drift_hop(ψ, qp, 0.0) == G.hop_drift(ψ, qg, 0.0), ψs)
        sp, _ = hop_seed(ps, Lp)
        @test sp == G.hop_seed(gs, Lg)
        o = [e for (e, (a, _)) in enumerate(edge_list(ps.P)) if a == 1]
        @test hop_response(Lp, c, ε / sqrt(2))(o, sp, 0.05c) == G.hop_response(Lg, c, ε / sqrt(2))(o, sp, 0.05c)
    end
    @testset "stored fits, rebuilt, simulated and classified" begin
        for (name, st) in stored
            ps, gs = st.spec, G.example(name; noise=st.spec.ε₀)
            @test same(st.layout, G.hop_layout(gs))
            mp, mg = first(hop_model(ps, st.layout, st.out; st.fast, st.contract)), G.HopModel(gs, st.out)
            @test same(mp, mg)
            T = 10sum(ps.τ)
            @test runs_equal(simulate_frames(mp; T, seeds=1:4, ps.sample), G.simulate(mg, gs; T, seeds=1:4))
            @test runs_equal(simulate_frames(mp; T, seeds=1:4, ps.sample, dtscale=0.005), G.simulate(mg, gs; T, seeds=1:4, dtscale=0.005))
            if name in ("foraging", "Δ-clique")
                cp, cg = classify(mp; spec=ps, Tnoisy=40sum(ps.τ), Tfree=150sum(ps.τ)), G.classify(mg, gs)
                @test cp[1] == cg.class && cp[2] == cg.outcomes
            end
        end
    end
    @testset "fit pipeline, 4 passes, with deadline and resume" begin
        ps, gs = pspec("foraging", 0.1), G.example("foraging"; noise=0.1)
        ck = tempname()
        fp = fit_hop(ps, hop_layout(ps); checkpoint=ck, passes=4)
        rm(ck; force=true); rm(ck * ".seed"; force=true)
        @test same(fp.model, G.fit(gs, G.Hopfield(; passes=4)))
        file = tempname()
        @test_throws G.OutOfTime G.fit(gs, G.Hopfield(; passes=4); checkpoint=file, deadline=time())
        @test same(fp.model, G.fit(gs, G.Hopfield(; passes=4); checkpoint=file))
        rm(file)
    end
    # A full refit can't reproduce a stored fit bit for bit: the prototype's own designs vary from run to run, and
    # refitting the stored Δ-clique 0.1 fit with the unpatched prototype gave rates up to 0.2% from the stored ones
    # (2026-11-13). So a full refit is checked against the patched prototype's. (The stored shared-solvable 0.1
    # fit, which fell back after pass 5, predates the rate cap; refitted now it runs all 12 passes. The fallback
    # itself is covered by failure injection in test/hopfield.jl.)
    @testset "full refit == patched prototype refit: Δ-clique, ε=0.1" begin
        st = only(st for (n, st) in stored if n == "Δ-clique" && st.spec.ε₀ == 0.1)
        ck = tempname()
        fp = fit_hop(st.spec, st.layout; checkpoint=ck)
        rm(ck; force=true); rm(ck * ".seed"; force=true)
        @test same(fp.model, G.fit(G.example("Δ-clique"; noise=0.1), G.Hopfield()))
    end
end
