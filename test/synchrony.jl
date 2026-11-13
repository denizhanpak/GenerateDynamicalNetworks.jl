# The synchrony backend, the race seed and the out-rate correction.

@testset "corner design" begin
    spec = example("foraging"; noise=0.1)
    L = GDN.sync_layout(spec.P)
    m = GDN.sync_model(spec, L, [1.0, -0.5, 2.0, 3.0, 0.7, 1.5]; fast=5.0, contract=7.0)
    q = GDN.parameters(m)
    # every corner is an equilibrium whose Jacobian has eigenvalues Λ[:, v] and one zero per pair (its common phase)
    for v in 1:2^L.M
        θ = SVector{2L.M}(ntuple(i -> iseven(i) && GDN.corner_bit(v, i ÷ 2) == 1 ? π : 0.0, 2L.M))
        @test maximum(abs, GDN.sync_drift(θ, q, 0.0)) < 1e-12
        @test sort(real.(GDN.eigvals(GDN.jac(GDN.sync_drift, q, θ)))) ≈ sort([m.Λ[:, v]; zeros(L.M)]) atol = 1e-6
        @test GDN.frame_state(m, q, θ) == m.node_of[v]
    end
    @test m.α == 7.0 && all(m.node_of[m.corner_of] .== 1:nstates(spec))
end

@testset "layouts" begin
    for name in example_names()
        spec = example(name)
        L = GDN.sync_layout(spec.P)
        edges = edge_list(spec)
        @test length(L.paths) == length(edges) && 2^L.M >= nstates(spec)
        interior = reduce(vcat, [p[2:end-1] for p in L.paths])
        @test allunique(interior) && isempty(intersect(interior, L.corner_of)) && allunique(L.corner_of)
        steps = Tuple{Int,Int}[]
        for (p, (i, j)) in zip(L.paths, edges)
            @test p[1] == L.corner_of[i] && p[end] == L.corner_of[j]
            @test all(k -> count_ones((p[k] - 1) ⊻ (p[k+1] - 1)) == 1, 1:length(p)-1)     # single-pair flips
            append!(steps, [(p[k], p[k+1]) for k in 1:length(p)-1])
        end
        @test allunique(steps) && !any(s -> reverse(s) in steps, steps)    # a cube edge carries one direction
        for i in 1:nstates(spec)                                          # exits of a state leave in different directions
            @test allunique([L.paths[e][2] for (e, (a, _)) in enumerate(edges) if a == i])
        end
    end
    @test_throws ArgumentError GDN.sync_layout(example("Δ-clique").P; Mmax=2)
end

@testset "race" begin
    # one exit against a Monte Carlo of dψ = sin ψ (g + k cos ψ) dt + √2 η dW, absorbed at |ψ| = π/2
    λ, c, η = 2.0, 8.0, 0.1
    g, k = (λ + c) / 2, (λ - c) / 2
    rng, dt, n = Xoshiro(3), 1e-4, 2000
    times = map(1:n) do _
        ψ, t = 0.0, 0.0
        while abs(ψ) < π / 2
            ψ += sin(ψ) * (g + k * cos(ψ)) * dt + sqrt(2dt) * η * randn(rng)
            t += dt
        end
        t
    end
    d, shares = GDN.race([λ], c, η)
    @test shares == [1.0]
    @test abs(d - mean(times)) < 4std(times) / sqrt(n) + 0.01d
    d2, s2 = GDN.race([λ, λ], c, η)
    @test s2 ≈ [0.5, 0.5] && d2 < d
    @test GDN.race([1.0, 3.0], c, η)[2][2] > 0.5                          # the faster exit wins more often
    # the seed solves every state's race for its dwell and shares
    spec = example("foraging"; noise=0.1)
    rough, fast = GDN.rough_rates(spec)
    out = GDN.race_seed_rates(spec, rough, fast, 0.1)
    for i in 1:nstates(spec)
        o = [e for (e, (a, _)) in enumerate(edge_list(spec)) if a == i]
        r = GDN.race(out[o], fast, 0.1)
        @test r[1] ≈ spec.τ[i] rtol = 1e-3
        @test r[2] ≈ [spec.P[i, j] for (_, j) in edge_list(spec)[o]] rtol = 1e-3
    end
end

@testset "synchrony model and fit" begin
    spec = TransitionSpec(ring(3), [1.0, 2.0, 3.0]; noise=0.1)
    rough, fast = GDN.rough_rates(spec)
    @test rough ≈ log(10) ./ [1.0, 2.0, 3.0] && fast ≈ 4log(10)
    m = SyncModel(spec, rough)
    @test m.M == 2 && size(m.Λ) == (2, 4)
    runs = simulate(m, spec; T=60.0, seeds=1:4)
    @test !any(r -> r.failed, runs) && all(r -> all(in(1:3), r.state), runs)
    @test classify(m, spec).class === :heteroclinic
    @test_throws ArgumentError Synchrony(; seed=:other)
    # a short correction, then a fit stopped by its deadline that resumes from the checkpoint
    build = out -> GDN.sync_model(spec, GDN.sync_layout(spec.P), out; fast, contract=fast)
    r = GDN.correct_rates(spec, build, rough; passes=2, T=30sum(spec.τ))
    @test length(r.trace) == 2 && r.out != rough && all(isfinite, r.out)
    b = Synchrony(; passes=1)
    f = fit(spec, b)
    @test f isa SyncModel && !any(r -> r.failed, simulate(f, spec; T=60.0, seeds=1:4))
    file = tempname()
    @test_throws OutOfTime fit(spec, b; checkpoint=file, deadline=time())
    @test isfile(file)
    @test fit(spec, b; checkpoint=file).Λ == f.Λ
    rm(file)
end
