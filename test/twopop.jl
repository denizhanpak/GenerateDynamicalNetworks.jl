# The shared fitting code (dwell-formula seed, edge correction, search) and the two-population backend.

@testset "dwell formulas" begin
    @test GDN.stone_holmes_mean(0.3, 1e-3, 2.0, 1.0) ≈ (2log(0.3 / 1e-3) + log(8) - log1p(2) + Base.MathConstants.γ) / 4
    @test GDN.stone_holmes_mean(0.3, 1e-4, 2.0, 1.0) > GDN.stone_holmes_mean(0.3, 1e-3, 2.0, 1.0)     # less noise, longer passage
    @test GDN.effective_noise([0.1]) == 0.1
    @test GDN.effective_noise([0.1, 0.1]) > 0.1 && GDN.effective_noise([0.1, 0.1, 0.1]) > GDN.effective_noise([0.1, 0.1])
    spec = example("foraging"; noise=1e-3)
    n = nstates(spec)
    ε_eff = [GDN.effective_noise(fill(1e-3, length(successors(spec, i)))) for i in 1:n]
    for form in (:u, :x)
        α, ν = GDN.dwell_rates(form, spec.τ, ε_eff, ones(n))
        @test allequal(α)
        @test minimum(ν) ≈ 2.0 rtol = 1e-6
        @test [GDN.designed_dwell(form, α[i], ν[i], ε_eff[i], 1.0, 0.1) for i in 1:n] ≈ spec.τ rtol = 1e-6
    end
    @test_throws ArgumentError GDN.dwell_rates(:u, [1.0, 1000.0], [0.1, 0.1], ones(2))
end

@testset "edge seed" begin
    spec = example("foraging"; noise=1e-3)
    edges = edge_list(spec)
    r = GDN.edge_seed_rates(spec, edges)
    @test r.seeded && r.α > 0 && length(r.logs) == length(edges)
    L = max(log(sqrt(0.1) / 1e-3), 1.0)
    k23, k24 = findfirst(==((2, 3)), edges), findfirst(==((2, 4)), edges)
    @test r.logs[k23] - r.logs[k24] ≈ log(0.7 / 0.3) / L                # branches split as p^(1/L)
    @test !GDN.edge_seed_rates(TransitionSpec(ring(2), [1.0, 1e6]; noise=0.1), [(1, 2), (2, 1)]).seeded
end

@testset "two-population model and fit" begin
    spec = TransitionSpec(ring(3), [1.0, 2.0, 3.0]; noise=0.1)
    θ = [log(4.0); fill(asinh(-1.0 / 0.01), 3)]
    m = TwoPopModel(spec, θ)
    @test m.s ≈ 4.0 && all(m.B .≈ 2.5) && m.from == (1, 2, 3) && m.to == (2, 3, 1) && m.α ≈ [10.0]
    runs = simulate(m, spec; T=60.0, seeds=1:4)
    @test !any(r -> r.failed, runs) && all(r -> all(in(1:3), r.state), runs)
    f = fit(spec, TwoPopulation(; passes=2, runs=4, T=20.0, budget=12, popsize=4, seconds=Inf))
    @test f isa TwoPopModel && length(f.B) == 3
    @test !any(r -> r.failed, simulate(f, spec; T=60.0, seeds=1:4))
end
