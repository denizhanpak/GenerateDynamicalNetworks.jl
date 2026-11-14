# The Horchler backend: graph repair, the closed-form design, the Lotka–Volterra model, a short fit.

@testset "repair" begin
    for (name, hidden) in (("foraging", 0), ("shared, solvable", 0), ("Δ-clique", 1), ("shared, conflicting", 1), ("flip-flop", 1),
                           ("two-way and exit", 2))
        spec = example(name)
        n = nstates(spec)
        r = GDN.repair(spec.P, spec.τ)
        @test r.ok && r.visible == n && size(r.P, 1) == n + hidden && length(r.reasons) == hidden
        @test r.P[1:n, 1:n] == spec.P .* (r.P[1:n, 1:n] .> 0)
        @test vec(sum(r.P; dims=2)) ≈ ones(size(r.P, 1))
        @test isempty(GDN.two_cycles(r.P)) && isempty(GDN.delta_cliques(r.P)) && GDN.noise_residual(r.P) < 1e-9
        # a state gives up as much dwell as its hidden states take on its edges
        for a in 1:n
            @test r.τ[a] + sum(r.P[a, h] * r.τ[h] for h in n+1:size(r.P, 1); init=0.0) ≈ spec.τ[a]
        end
    end
    @test GDN.repair(example("flip-flop").P, [1.0, 0.3]).reasons == ["2-cycle 1 ⇄ 2"]
    @test GDN.repair(example("Δ-clique").P, example("Δ-clique").τ).reasons == ["Δ-clique 1 → 2 → 3"]
    # an uneven three-way branch has no noise formula and no hidden state fixes it
    P = [0 0.2 0.3 0.5 0; 0 0 0 0 1; 0 0 0 0 1; 0 0 0 0 1; 1 0 0 0 0]
    @test !GDN.repair(P, ones(5)).ok
    @test_throws ArgumentError GDN.horchler_design(P, ones(5); noise=0.1)
    # a two-way branch's odds come from the ratio of its targets' noise
    ε = GDN.branch_noise(example("foraging").P, 0.1)
    @test ε[3] / ε[4] ≈ tan(0.35π) && ε[[1, 2, 5]] ≈ fill(0.1, 3)
end

@testset "design" begin
    # the SHCTools notebook's x-form design of the foraging graph at noise 1e-5 (research/benchmarks/phase2/reference)
    P = [0 1 0 0 0; 0 0 0.7 0.3 0; 0 0 0 0 1; 0 0 0 0 1; 1 0 0 0 0]
    d = GDN.horchler_design(P, [2.0, 1.0, 0.5, 1.5, 1.0]; form=:x, noise=1e-5)
    @test d.ε ≈ [1.0e-5, 1.0e-5, 1.4009320131630766e-5, 7.138105137180518e-6, 1.0e-5] rtol = 1e-12
    @test d.ε_eff ≈ [1.0e-5, 1.9211595769946658e-5, 1.0e-5, 1.0e-5, 1.0e-5] rtol = 1e-12
    @test d.α ≈ fill(55.38881046430524, 5) rtol = 1e-12
    @test d.ν ≈ [9.656996495032827, 4.77959126038457, 2.0, 7.053340706512751, 4.493898938642102] rtol = 1e-12
    ρ = [55.38881046430524 110.77762092861047 110.77762092861047 110.77762092861047 43.063475333150336
         49.6531958254478 55.38881046430524 110.77762092861047 110.77762092861047 110.77762092861047
         110.77762092861047 43.80020226607019 55.38881046430524 110.77762092861047 110.77762092861047
         110.77762092861047 43.80020226607019 110.77762092861047 55.38881046430524 110.77762092861047
         110.77762092861047 110.77762092861047 27.69440523215262 47.53596274731326 55.38881046430524]
    @test d.ρ ≈ ρ rtol = 1e-12
    # every state's designed dwell is the one asked for, in both forms, hidden states included
    for name in ("foraging", "Δ-clique", "flip-flop"), form in (:u, :x)
        spec = example(name; noise=1e-3)
        d = GDN.horchler_design(spec.P, spec.τ; form, noise=1e-3)
        @test allequal(d.α) && minimum(d.ν) ≈ 2.0
        @test [GDN.designed_dwell(form, d.α[i], d.ν[i], d.ε_eff[i], d.β[i], d.κ) for i in eachindex(d.τ)] ≈ d.τ rtol = 1e-8
        # each node is a saddle: it leaves at α/ν towards each node that follows it and contracts at α along the
        # others (along its own direction at α in the x form, 2α in the u form)
        m = GDN.lv_model(d)
        q = GDN.parameters(m)
        for i in eachindex(d.α)
            u = SVector{length(d.α)}(ntuple(j -> j == i ? 1.0 : 0.0, length(d.α)))
            @test maximum(abs, GDN.drift_function(m)(u, q, 0.0)) < 1e-12
            want = [j == i ? (form === :u ? -2 : -1) * d.α[i] : d.P[i, j] > 0 ? d.α[i] / d.ν[i] : -d.α[i] for j in eachindex(d.α)]
            @test sort(real.(GDN.eigvals(GDN.jac(GDN.drift_function(m), q, u)))) ≈ sort(want) rtol = 1e-6
        end
    end
    @test_throws ArgumentError GDN.horchler_design(example("foraging").P, example("foraging").τ; form=:x, noise=0.1)
end

@testset "model" begin
    spec = example("flip-flop"; noise=0.1)
    d = GDN.horchler_design(spec.P, spec.τ; noise=0.1)
    m = GDN.lv_model(d)
    q = GDN.parameters(m)
    @test m isa LVModel && m.visible == 2 && length(m.α) == 3
    @test GDN.initial_state(m) ≈ [sqrt(0.9), sqrt(0.1), sqrt(0.1)]
    @test GDN.frame_state(m, q, SVector(0.9, 0.1, 0.2)) == 1 && GDN.frame_state(m, q, SVector(0.1, 0.2, -0.9)) == 0
    @test GDN.project(m, SVector(-0.1, 0.2, 0.3)) === nothing
    x = GDN.lv_model(GDN.horchler_design(spec.P, spec.τ; form=:x, noise=0.1))
    @test GDN.project(x, SVector(-0.1, 0.2, 0.3)) == SVector(0.0, 0.2, 0.3) && GDN.initial_state(x)[1] ≈ 0.9
    @test_throws ArgumentError LVModel(:w, m.P, 2, 0.1, m.β, m.α, m.ε, m.ρ)
    @test_throws ArgumentError LVModel(:u, m.P, 4, 0.1, m.β, m.α, m.ε, m.ρ)
    for model in (m, x)
        runs = simulate(model, spec; T=40.0, seeds=1:4)
        @test !any(r -> r.failed, runs) && all(r -> length(r.state) > 10 && all(in(1:2), r.state), runs)
    end
    @test classify(m, spec).class === :heteroclinic
end

@testset "fit" begin
    @test_throws ArgumentError Horchler(:w)
    @test_throws ArgumentError Horchler(; passes=0)
    @test Horchler().form === :u && Horchler(:x).form === :x && Horchler().seconds == 300
    @test_throws ArgumentError fit(example("foraging"; noise=0.1), Horchler(:x))
    spec = TransitionSpec(ring(3), [1.0, 2.0, 3.0]; noise=0.1)
    b = Horchler(; passes=1, runs=4, T=30sum(spec.τ), budget=12, popsize=6, seconds=60)
    m = fit(spec, b)
    @test m isa LVModel && m.visible == 3
    @test fit(spec, b).ρ == m.ρ                         # a fit depends on the seeds, not on time or threads
    runs = simulate(m, spec; T=60.0, seeds=1:4)
    @test !any(r -> r.failed, runs) && all(r -> all(in(1:3), r.state), runs)
end
