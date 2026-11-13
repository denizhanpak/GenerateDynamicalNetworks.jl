# Differential tests for the CTRNN design backend against the prototype (research/benchmarks/phase5/ctrnn_design.jl).
# Local only, like run.jl. From the repository root:
#   julia --project=test/differential -t 7 test/differential/ctrnn.jl
using Test, DelimitedFiles
const PROTO = joinpath(@__DIR__, "..", "..", "research", "benchmarks")
include(joinpath(PROTO, "phase5", "classes.jl"))           # backends.jl and fits() through it
include(joinpath(PROTO, "phase5", "ctrnn_design.jl"))
include(joinpath(PROTO, "phase5", "twoway_cases.jl"))
import GenerateDynamicalNetworks as G

pspec(name, ε) = (c = only(x for x in [NAMED; TWOWAY] if x[1] == name);
                  (; P=float(c[2]), τ=c[3], form=:u, ε₀=ε, sample=startswith(name, "worm") ? WORM_SPEC.sample : minimum(c[3]) / 10))
same(a::CTRNNModel, b::G.CTRNNModel) = a.P == b.P && a.visible == b.visible && a.α == b.α && a.W == b.W && a.b == b.b &&
                                       a.τ == b.τ && a.g == b.g && a.ε == b.ε

@testset "CTRNN design backend == prototype" begin
    @testset "passage time and its inverse" begin
        # cells as the design builds them (on level 12 noise SDs), at offsets across the solver's range
        for name in ("foraging", "two-way and exit"), ε in (1e-3, 0.1)
            L = G.ctrnn_layout(G.example(name; noise=ε))
            for c in eachindex(L.τc), frac in (-0.3, 0.2, 0.6, 1.1)
                args = (L.A[c], L.g, L.τc[c], ε)
                @test G.ctrnn_passage_time(frac * L.A[c], args...) == passage_time(frac * L.A[c], args...)
            end
            @test G.ctrnn_solve_offset(2.0, L.A[1], L.g, L.τc[1], ε) == solve_offset(2.0, L.A[1], L.g, L.τc[1], ε)
        end
    end
    @testset "layout, seed and design: $name" for name in G.example_names(), ε in (1e-3, 0.1)
        ps, gs = pspec(name, ε), G.example(name; noise=ε)
        lp, lg = design_layout(ps), G.ctrnn_layout(gs)
        @test lp == lg
        dp, dg = seed_offsets(ps, lp), G.ctrnn_seed_offsets(gs, lg)
        @test dp == dg
        @test same(design_model(ps, lp, dp), G.ctrnn_design_model(gs, lg, dg))
    end
    @testset "correction passes: $name, by_state=$by_state" for (name, by_state) in
            (("two-way and exit", true), ("foraging", true), ("foraging", false), ("flip-flop", true))
        ps, gs = pspec(name, 0.1), G.example(name; noise=0.1)
        lp, lg = design_layout(ps), G.ctrnn_layout(gs)
        kw = (; by_state, passes=4, late_runs=16)
        dp = redirect_stdout(devnull) do
            correct_offsets(ps, lp, seed_offsets(ps, lp); kw...)
        end
        dg = G.ctrnn_correct_offsets(gs, lg, G.ctrnn_seed_offsets(gs, lg); kw...)
        @test dp == dg
        @test dp != seed_offsets(ps, lp)                       # the passes did move the offsets
    end
    @testset "fit pipeline" begin
        ps, gs = pspec("hub with two returns", 0.1), G.example("hub with two returns"; noise=0.1)
        mp = redirect_stdout(devnull) do
            fit_ctrnn_design(ps; passes=4, late_runs=16).model
        end
        @test same(mp, G.fit(gs, G.CTRNNDesign(; passes=4, late_runs=16)))
    end
    @testset "stored fits rebuilt from θ" begin
        dir = joinpath(PROTO, "phase5", "results")
        files = filter(f -> startswith(f, "theta_ctrnn_design_") || startswith(f, "theta_ctrnn_twoway_"), readdir(dir))
        checked = 0
        for f in files
            k = findfirst(n -> endswith(f, "_$(san(n))_0.1.csv"), G.example_names())
            k === nothing && continue                          # the real worm graph is not an example
            name = G.example_names()[k]
            ps, gs = pspec(name, 0.1), G.example(name; noise=0.1)
            θ = vec(readdlm(joinpath(dir, f), ','))
            mg = G.CTRNNModel(gs, θ)
            @test same(ctrnn_model(ps, θ; cap=Inf), mg)
            @test G.ctrnn_parameters(mg) ≈ θ
            # the stored fit's time constants and gain are the design's, so they follow from the spec alone
            L = G.ctrnn_layout(gs)
            @test mg.τ ≈ L.τc && mg.g ≈ L.g
            checked += 1
        end
        @test checked >= 9
    end
end
