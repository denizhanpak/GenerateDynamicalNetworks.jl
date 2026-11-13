# The CTRNN design backend.

const GDN = GenerateDynamicalNetworks

@testset "passage time" begin
    L = GDN.ctrnn_layout(example("flip-flop"; noise=0.1))
    A, g, τ, ε = L.A[1], L.g, L.τc[1], 0.1
    d = GDN.ctrnn_solve_offset(8τ, A, g, τ, ε)
    @test GDN.ctrnn_passage_time(d, A, g, τ, ε) ≈ 8τ rtol = 1e-4
    @test GDN.ctrnn_passage_time(d + 0.1A, A, g, τ, ε) > GDN.ctrnn_passage_time(d, A, g, τ, ε)
    # Monte Carlo of the same cell, dy = (−y − d + A σ(g y)) / τ dt + ε dW from −d to A/2
    rng, dt, runs = Xoshiro(3), τ / 5000, 2000
    times = map(1:runs) do _
        y, t = -d, 0.0
        while y < A / 2
            y += (-y - d + A / (1 + exp(-g * y))) / τ * dt + ε * sqrt(dt) * randn(rng)
            t += dt
        end
        t
    end
    @test abs(mean(times) - 8τ) < 3std(times) / sqrt(runs) + 0.02 * 8τ
end

@testset "layout and design" begin
    flip = example("flip-flop")
    L = GDN.ctrnn_layout(flip)
    @test L.n == 2 && L.rel == [(2, 1)] && L.receiver == [2, 3]          # 1 → 2 fires state 2, 2 → 1 fires the relay
    @test L.τc ≈ fill(0.25 * 0.3, 3)
    @test L.A ≈ 12 .* 0.1 .* sqrt.(L.τc ./ 2) && L.D ≈ 1.5 .* L.A
    m = GDN.ctrnn_design_model(flip, L, [0.1, 0.2])
    @test size(m.W) == (3, 3) && m.visible == 2 && m.b == zeros(3)
    @test m.W[2, 1] == -0.1 && m.W[3, 2] == -0.2                       # the two edges' offsets
    @test m.W[1, 3] ≈ L.D[1] + 0.5L.A[1] && m.W[2, 3] == -L.D[2]        # the relay drives 1 on and holds 2 off
    @test all(i -> m.W[i, i] == L.A[i], 1:3)
    fora = GDN.ctrnn_layout(example("foraging"))
    @test isempty(fora.rel) && fora.receiver == [j for (_, j) in edge_list(example("foraging").P)]
    @test all(isfinite, GDN.ctrnn_seed_offsets(example("foraging"), fora))
end

@testset "parameter vector" begin
    spec = example("two-way and exit")
    L = GDN.ctrnn_layout(spec)
    m = GDN.ctrnn_design_model(spec, L, GDN.ctrnn_seed_offsets(spec, L))
    θ = GDN.ctrnn_parameters(m)
    @test length(θ) == (size(m.W, 1) + 1)^2
    m2 = CTRNNModel(spec, θ)
    @test m2.W == m.W && m2.τ ≈ m.τ && m2.g ≈ m.g && m2.α ≈ m.α
    @test_throws ArgumentError CTRNNModel(spec, θ; cap=1.0)
    @test_throws DimensionMismatch CTRNNModel(spec, θ[1:end-1])
end

@testset "fit, checkpoint and deadline" begin
    spec = example("flip-flop")
    b = CTRNNDesign(; passes=2, late_runs=16)
    m = fit(spec, b)
    @test m isa CTRNNModel && size(m.W) == (3, 3)
    c = validate(m, spec; seeds=1:8, T=100sum(spec.τ))
    @test c.failed == 0 && c.off == 0 && all(isfinite, c.dwell)
    # a fit stopped by its deadline resumes from the checkpoint to the same model
    file = tempname()
    @test_throws OutOfTime fit(spec, b; checkpoint=file, deadline=time())
    @test isfile(file)
    @test fit(spec, b; checkpoint=file).W == m.W
    rm(file)
end
